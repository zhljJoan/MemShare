"""Probability- and variance-aware immutable edge-feature cache."""

from dataclasses import dataclass
import math
import threading
from typing import Dict, Optional, Sequence, Tuple

import torch


@dataclass
class CacheLookup:
    values: torch.Tensor
    hit_mask: torch.Tensor
    slots: torch.Tensor


class ProbabilityAwareEdgeCache:
    """Fixed-capacity opportunistic cache for immutable edge features.

    Admission value combines expected communication bytes, observed reuse, and
    a caller-provided residual-variance reduction.  A minimum residency and
    hysteresis avoid cache thrashing.
    """

    def __init__(
        self,
        capacity: int,
        feature_shape: Sequence[int],
        *,
        dtype: torch.dtype,
        device: torch.device,
        variance_weight: float = 1.0,
        score_ema: float = 0.9,
        min_residency: int = 8,
        admission_hysteresis: float = 0.05,
        max_replacements_per_batch: int = 64,
    ) -> None:
        if capacity < 0:
            raise ValueError("capacity cannot be negative")
        if not 0.0 <= score_ema < 1.0:
            raise ValueError("score_ema must be in [0, 1)")
        if min_residency < 0:
            raise ValueError("min_residency cannot be negative")
        if admission_hysteresis < 0:
            raise ValueError("admission_hysteresis cannot be negative")
        if max_replacements_per_batch < 0:
            raise ValueError("max_replacements_per_batch cannot be negative")
        self.capacity = int(capacity)
        self.feature_shape = tuple(int(size) for size in feature_shape)
        self.dtype = dtype
        self.device = torch.device(device)
        self.variance_weight = float(variance_weight)
        self.score_ema = float(score_ema)
        self.min_residency = int(min_residency)
        self.admission_hysteresis = float(admission_hysteresis)
        self.max_replacements_per_batch = int(max_replacements_per_batch)
        self.feature_bytes = max(
            1,
            math.prod(self.feature_shape)
            * torch.empty((), dtype=self.dtype).element_size(),
        )

        self.values = torch.empty(
            (self.capacity,) + self.feature_shape,
            dtype=self.dtype,
            device=self.device,
        )
        self.ids = torch.full((self.capacity,), -1, dtype=torch.long)
        self.scores = torch.zeros(self.capacity, dtype=torch.float64)
        self.inserted_at = torch.full((self.capacity,), -1, dtype=torch.long)
        self.last_access = torch.full((self.capacity,), -1, dtype=torch.long)
        self.valid = torch.zeros(self.capacity, dtype=torch.bool)
        self._id_to_slot: Dict[int, int] = {}
        self._step = 0
        self._lock = threading.RLock()

        self.requests = 0
        self.hits = 0
        self.bytes_saved = 0
        self.insertions = 0
        self.evictions = 0
        self.rejected = 0

    def __len__(self) -> int:
        return len(self._id_to_slot)

    @staticmethod
    def _cpu_ids(edge_ids: torch.Tensor) -> torch.Tensor:
        return edge_ids.detach().reshape(-1).cpu().long()

    def contains(self, edge_ids: torch.Tensor) -> torch.Tensor:
        ids = self._cpu_ids(edge_ids)
        with self._lock:
            mask = [int(edge_id) in self._id_to_slot for edge_id in ids.tolist()]
        return torch.tensor(mask, dtype=torch.bool, device=edge_ids.device).reshape(
            edge_ids.shape
        )

    def lookup(self, edge_ids: torch.Tensor, *, count: bool = True) -> CacheLookup:
        ids = self._cpu_ids(edge_ids)
        hit_mask_cpu = torch.zeros(ids.numel(), dtype=torch.bool)
        slots_cpu = torch.full((ids.numel(),), -1, dtype=torch.long)
        with self._lock:
            self._step += 1
            for index, edge_id in enumerate(ids.tolist()):
                slot = self._id_to_slot.get(int(edge_id))
                if slot is not None:
                    hit_mask_cpu[index] = True
                    slots_cpu[index] = slot
                    self.last_access[slot] = self._step
            hit_count = int(hit_mask_cpu.sum().item())
            if count:
                self.requests += ids.numel()
                self.hits += hit_count
                self.bytes_saved += hit_count * self.feature_bytes

        output = torch.zeros(
            (ids.numel(),) + self.feature_shape,
            dtype=self.dtype,
            device=self.device,
        )
        if bool(hit_mask_cpu.any()):
            output[hit_mask_cpu.to(self.device)] = self.values[
                slots_cpu[hit_mask_cpu].to(self.device)
            ]
        return CacheLookup(
            values=output.reshape(tuple(edge_ids.shape) + self.feature_shape),
            hit_mask=hit_mask_cpu.to(edge_ids.device).reshape(edge_ids.shape),
            slots=slots_cpu.to(edge_ids.device).reshape(edge_ids.shape),
        )

    @staticmethod
    def _signal(
        value: Optional[torch.Tensor], count: int, default: float
    ) -> torch.Tensor:
        if value is None:
            return torch.full((count,), default, dtype=torch.float64)
        flattened = value.detach().reshape(-1).cpu().double()
        if flattened.numel() == 1 and count != 1:
            flattened = flattened.expand(count)
        if flattened.numel() != count:
            raise ValueError("cache admission signals must align with edge_ids")
        return flattened

    def _candidate_score(
        self,
        access_probability: torch.Tensor,
        communication_cost: torch.Tensor,
        variance_reduction: torch.Tensor,
        reuse_count: torch.Tensor,
    ) -> torch.Tensor:
        communication_value = (
            access_probability.clamp_min(0.0)
            * communication_cost.clamp_min(0.0)
            * reuse_count.clamp_min(1.0)
            * self.feature_bytes
        )
        return (
            communication_value
            + self.variance_weight * variance_reduction.clamp_min(0.0)
        ) / self.feature_bytes

    def _free_slot(self) -> Optional[int]:
        free = (~self.valid).nonzero().reshape(-1)
        return int(free[0].item()) if free.numel() else None

    def _victim(self) -> Optional[int]:
        if self.capacity == 0:
            return None
        eligible = self.valid & ((self._step - self.inserted_at) >= self.min_residency)
        slots = eligible.nonzero().reshape(-1)
        if slots.numel() == 0:
            return None
        scores = self.scores[slots]
        oldest_bias = self.last_access[slots].double() * 1e-15
        return int(slots[torch.argmin(scores + oldest_bias)].item())

    def admit(
        self,
        edge_ids: torch.Tensor,
        features: torch.Tensor,
        *,
        access_probability: Optional[torch.Tensor] = None,
        communication_cost: Optional[torch.Tensor] = None,
        variance_reduction: Optional[torch.Tensor] = None,
        reuse_count: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Observe fetched features and opportunistically admit valuable edges."""

        ids = self._cpu_ids(edge_ids)
        feature = features.detach().reshape((ids.numel(),) + self.feature_shape)
        if feature.shape[1:] != self.feature_shape:
            raise ValueError("feature shape does not match cache")
        probability = self._signal(access_probability, ids.numel(), 1.0)
        cost = self._signal(communication_cost, ids.numel(), 1.0)
        variance = self._signal(variance_reduction, ids.numel(), 0.0)
        reuse = self._signal(reuse_count, ids.numel(), 1.0)
        candidate_scores = self._candidate_score(probability, cost, variance, reuse)
        admitted = torch.zeros(ids.numel(), dtype=torch.bool)

        with self._lock:
            self._step += 1
            replacements = 0
            for index, (edge_id, candidate_score) in enumerate(
                zip(ids.tolist(), candidate_scores.tolist())
            ):
                edge_id = int(edge_id)
                slot = self._id_to_slot.get(edge_id)
                if slot is not None:
                    self.scores[slot] = (
                        self.score_ema * self.scores[slot]
                        + (1.0 - self.score_ema) * float(candidate_score)
                    )
                    self.last_access[slot] = self._step
                    admitted[index] = True
                    continue
                if self.capacity == 0:
                    self.rejected += 1
                    continue
                slot = self._free_slot()
                victim_id = None
                if slot is None:
                    if replacements >= self.max_replacements_per_batch:
                        self.rejected += 1
                        continue
                    slot = self._victim()
                    if slot is None:
                        self.rejected += 1
                        continue
                    threshold = float(self.scores[slot].item()) * (
                        1.0 + self.admission_hysteresis
                    )
                    if float(candidate_score) <= threshold:
                        self.rejected += 1
                        continue
                    victim_id = int(self.ids[slot].item())

                if victim_id is not None:
                    del self._id_to_slot[victim_id]
                    self.evictions += 1
                    replacements += 1
                self.values[slot].copy_(feature[index].to(self.device))
                self.ids[slot] = edge_id
                self.scores[slot] = float(candidate_score)
                self.inserted_at[slot] = self._step
                self.last_access[slot] = self._step
                self.valid[slot] = True
                self._id_to_slot[edge_id] = slot
                self.insertions += 1
                admitted[index] = True
        return admitted.to(edge_ids.device).reshape(edge_ids.shape)

    def stats(self) -> Dict[str, float]:
        with self._lock:
            return {
                "capacity": float(self.capacity),
                "size": float(len(self._id_to_slot)),
                "requests": float(self.requests),
                "hits": float(self.hits),
                "object_hit_rate": self.hits / self.requests if self.requests else 0.0,
                "bytes_saved": float(self.bytes_saved),
                "insertions": float(self.insertions),
                "evictions": float(self.evictions),
                "rejected": float(self.rejected),
                "memory_bytes": float(self.capacity * self.feature_bytes),
            }
