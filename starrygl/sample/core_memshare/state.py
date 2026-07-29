"""Bounded last-fetched state used as the residual prediction baseline."""

from collections import OrderedDict
from dataclasses import dataclass
import threading
from typing import Dict, Optional, Tuple

import torch


@dataclass
class _MemoryEntry:
    value: torch.Tensor
    timestamp: float
    uncertainty: float
    owner: int


@dataclass
class PredictionBatch:
    value: torch.Tensor
    valid: torch.Tensor
    uncertainty: torch.Tensor
    age: torch.Tensor
    source: torch.Tensor


class LastFetchedMemoryPredictor:
    """An LRU-bounded memory predictor with owner-centroid fallback.

    Source codes in :class:`PredictionBatch` are:

    * 2: exact last-fetched state;
    * 1: owner centroid;
    * 0: zero fallback.
    """

    def __init__(
        self,
        dimension: int,
        capacity: int,
        *,
        uncertainty_ema: float = 0.9,
        centroid_ema: float = 0.99,
        default_uncertainty: float = 1.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if capacity < 0:
            raise ValueError("capacity cannot be negative")
        if not 0.0 <= uncertainty_ema < 1.0:
            raise ValueError("uncertainty_ema must be in [0, 1)")
        if not 0.0 <= centroid_ema < 1.0:
            raise ValueError("centroid_ema must be in [0, 1)")
        self.dimension = int(dimension)
        self.capacity = int(capacity)
        self.uncertainty_ema = float(uncertainty_ema)
        self.centroid_ema = float(centroid_ema)
        self.default_uncertainty = float(default_uncertainty)
        self.dtype = dtype
        self._entries: "OrderedDict[int, _MemoryEntry]" = OrderedDict()
        self._centroids: Dict[int, torch.Tensor] = {}
        self._centroid_counts: Dict[int, int] = {}
        self._lock = threading.RLock()
        self.observations = 0
        self.residual_sum = 0.0
        self.last_hit_count = 0
        self.centroid_hit_count = 0
        self.zero_fallback_count = 0

    @staticmethod
    def _flatten_ids(value: torch.Tensor) -> torch.Tensor:
        if value is None:
            raise ValueError("node_ids are required")
        return value.detach().reshape(-1).cpu().long()

    def _owners(
        self, node_ids: torch.Tensor, owner_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if owner_ids is None:
            return torch.zeros_like(node_ids)
        owners = owner_ids.detach().reshape(-1).cpu().long()
        if owners.numel() != node_ids.numel():
            raise ValueError("owner_ids must align with node_ids")
        return owners

    def observe(
        self,
        node_ids: torch.Tensor,
        memory: torch.Tensor,
        timestamps: torch.Tensor,
        *,
        owner_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Store fetched memories and return observed RMS prediction residuals."""

        ids = self._flatten_ids(node_ids)
        values = memory.detach().reshape(ids.numel(), -1).cpu().to(self.dtype)
        times = timestamps.detach().reshape(-1).cpu().float()
        owners = self._owners(ids, owner_ids)
        if values.shape[1] != self.dimension:
            raise ValueError("memory dimension does not match predictor")
        if times.numel() != ids.numel():
            raise ValueError("timestamps must align with node_ids")
        residuals = torch.empty(ids.numel(), dtype=torch.float32)
        with self._lock:
            for index, (node_id, value, timestamp, owner) in enumerate(
                zip(ids.tolist(), values, times.tolist(), owners.tolist())
            ):
                previous = self._entries.pop(int(node_id), None)
                if previous is None:
                    residual = self.default_uncertainty
                    uncertainty = residual
                else:
                    residual = float(
                        torch.sqrt(torch.mean(torch.square(value - previous.value))).item()
                    )
                    uncertainty = (
                        self.uncertainty_ema * previous.uncertainty
                        + (1.0 - self.uncertainty_ema) * residual
                    )
                residuals[index] = residual
                if self.capacity > 0:
                    self._entries[int(node_id)] = _MemoryEntry(
                        value=value.clone(),
                        timestamp=float(timestamp),
                        uncertainty=float(uncertainty),
                        owner=int(owner),
                    )
                    while len(self._entries) > self.capacity:
                        self._entries.popitem(last=False)

                centroid = self._centroids.get(int(owner))
                if centroid is None:
                    self._centroids[int(owner)] = value.clone()
                    self._centroid_counts[int(owner)] = 1
                else:
                    centroid.mul_(self.centroid_ema).add_(
                        value, alpha=1.0 - self.centroid_ema
                    )
                    self._centroid_counts[int(owner)] += 1
                self.observations += 1
                self.residual_sum += residual
        return residuals

    def predict(
        self,
        node_ids: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        *,
        owner_ids: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> PredictionBatch:
        ids = self._flatten_ids(node_ids)
        owners = self._owners(ids, owner_ids)
        if timestamps is None:
            times = torch.full((ids.numel(),), float("nan"))
        else:
            times = timestamps.detach().reshape(-1).cpu().float()
            if times.numel() != ids.numel():
                raise ValueError("timestamps must align with node_ids")
        values = torch.zeros((ids.numel(), self.dimension), dtype=self.dtype)
        valid = torch.zeros(ids.numel(), dtype=torch.bool)
        uncertainty = torch.full(
            (ids.numel(),), self.default_uncertainty, dtype=torch.float32
        )
        age = torch.full((ids.numel(),), float("inf"), dtype=torch.float32)
        source = torch.zeros(ids.numel(), dtype=torch.int8)

        with self._lock:
            for index, (node_id, timestamp, owner) in enumerate(
                zip(ids.tolist(), times.tolist(), owners.tolist())
            ):
                entry = self._entries.pop(int(node_id), None)
                if entry is not None:
                    self._entries[int(node_id)] = entry
                    values[index] = entry.value
                    valid[index] = True
                    uncertainty[index] = entry.uncertainty
                    if timestamp == timestamp:
                        age[index] = max(0.0, float(timestamp) - entry.timestamp)
                    else:
                        age[index] = 0.0
                    source[index] = 2
                    self.last_hit_count += 1
                    continue
                centroid = self._centroids.get(int(owner))
                if centroid is not None:
                    values[index] = centroid
                    valid[index] = True
                    source[index] = 1
                    self.centroid_hit_count += 1
                else:
                    self.zero_fallback_count += 1

        target_device = device if device is not None else node_ids.device
        return PredictionBatch(
            value=values.to(target_device),
            valid=valid.to(target_device),
            uncertainty=uncertainty.to(target_device),
            age=age.to(target_device),
            source=source.to(target_device),
        )

    def native_state(
        self, node_ids: torch.Tensor, timestamps: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return uncertainty and age tensors suitable for sampler updates."""

        prediction = self.predict(node_ids, timestamps)
        return prediction.uncertainty.cpu(), prediction.age.cpu()

    def stats(self) -> Dict[str, float]:
        with self._lock:
            return {
                "entries": float(len(self._entries)),
                "owners": float(len(self._centroids)),
                "observations": float(self.observations),
                "mean_observed_residual": (
                    self.residual_sum / self.observations
                    if self.observations
                    else 0.0
                ),
                "last_hit_count": float(self.last_hit_count),
                "centroid_hit_count": float(self.centroid_hit_count),
                "zero_fallback_count": float(self.zero_fallback_count),
            }
