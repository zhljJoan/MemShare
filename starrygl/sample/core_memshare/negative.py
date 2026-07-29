"""Communication-aware hard negatives with exact two-stage correction."""

from collections import OrderedDict
import math
import threading
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from ..sample_core.base import NegativeSampling
from .sampling import effective_sample_size


def _owner(dist_ids: torch.Tensor) -> torch.Tensor:
    return (dist_ids.long() >> 50) & 0xFFFF


def importance_weighted_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    importance_weight: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Binary cross entropy under an importance-sampled proposal."""

    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be one of: none, mean, sum")
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weight = importance_weight.to(device=loss.device, dtype=loss.dtype)
    while weight.ndim < loss.ndim:
        weight = weight.unsqueeze(-1)
    weighted = loss * weight
    if reduction == "none":
        return weighted
    if reduction == "sum":
        return weighted.sum()
    return weighted.mean()


class CommunicationAwareNegativeSampling(NegativeSampling):
    r"""Two-stage hard-negative proposal with conditional importance weights.

    For every negative slot, ``pool_factor`` candidates are drawn independently
    from a localism proposal.  One candidate occurrence is selected using the
    hardness/cost proposal ``q``.  The returned weight combines the exact
    target/localism density ratio with ``(1 / pool_factor) / q``.  With the
    default ``localism_probability=1`` this reduces to global-uniform candidate
    generation and the second-stage correction alone.
    """

    def __init__(
        self,
        mode: str = "triplet",
        amount: int = 1,
        *,
        dst_node_list: torch.Tensor,
        id_mapper: Optional[torch.Tensor] = None,
        local_part: Optional[int] = None,
        pool_factor: int = 4,
        hardness_beta: float = 1.0,
        exploration: float = 0.1,
        temperature: float = 1.0,
        local_cost: float = 0.05,
        cached_cost: float = 0.02,
        reuse_cost: float = 0.01,
        remote_cost: float = 1.0,
        cost_floor: float = 1e-6,
        embedding_capacity: int = 100000,
        max_importance_weight: Optional[float] = None,
        localism_probability: float = 1.0,
        validity_filter: Optional[
            Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor]
        ] = None,
        seed: int = 6773,
    ) -> None:
        super().__init__(mode=mode, amount=amount)
        if not self.is_triplet():
            raise ValueError("communication-aware sampling currently supports triplet mode")
        if dst_node_list is None or dst_node_list.numel() == 0:
            raise ValueError("dst_node_list must contain at least one node")
        if pool_factor <= 0:
            raise ValueError("pool_factor must be positive")
        if hardness_beta < 0:
            raise ValueError("hardness_beta cannot be negative")
        if not 0.0 < exploration <= 1.0:
            raise ValueError("exploration must be in (0, 1]")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if min(local_cost, cached_cost, reuse_cost, remote_cost, cost_floor) <= 0:
            raise ValueError("all communication costs must be positive")
        if embedding_capacity < 0:
            raise ValueError("embedding_capacity cannot be negative")
        if max_importance_weight is not None and max_importance_weight <= 0:
            raise ValueError("max_importance_weight must be positive")
        if not 0.0 < localism_probability <= 1.0:
            raise ValueError("localism_probability must be in (0, 1]")

        self.dst_node_list = torch.unique(dst_node_list.detach().cpu().long())
        self.id_mapper = id_mapper.detach().cpu().long() if id_mapper is not None else None
        if local_part is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                local_part = torch.distributed.get_rank()
            else:
                local_part = 0
        self.local_part = int(local_part)
        self.pool_factor = int(pool_factor)
        self.hardness_beta = float(hardness_beta)
        self.exploration = float(exploration)
        self.temperature = float(temperature)
        self.local_cost = float(local_cost)
        self.cached_cost = float(cached_cost)
        self.reuse_cost = float(reuse_cost)
        self.remote_cost = float(remote_cost)
        self.cost_floor = float(cost_floor)
        self.embedding_capacity = int(embedding_capacity)
        self.max_importance_weight = max_importance_weight
        self.localism_probability = float(localism_probability)
        self.validity_filter = validity_filter
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed) ^ self.local_part)
        destination_dist = self._map_ids(self.dst_node_list)
        local_mask = _owner(destination_dist) == self.local_part
        self.local_dst_node_list = self.dst_node_list[local_mask]
        if (
            self.localism_probability < 1.0
            and self.local_dst_node_list.numel() == 0
        ):
            raise ValueError(
                "localism_probability below one requires local destinations"
            )
        self.local_destination_fraction = (
            self.local_dst_node_list.numel() / self.dst_node_list.numel()
        )

        self._embeddings: "OrderedDict[int, torch.Tensor]" = OrderedDict()
        self._embedding_dimension = 0
        self._lock = threading.RLock()
        self.last_importance_weight = torch.empty(0)
        self.last_proposal_probability = torch.empty(0)
        self.last_hardness = torch.empty(0)
        self.last_communication_cost = torch.empty(0)
        self.last_effective_sample_size = 0.0
        self.last_clipped_fraction = 0.0
        self.total_samples = 0
        self.total_remote_samples = 0
        self.total_reused_samples = 0
        self.total_hardness = 0.0
        self.total_communication_cost = 0.0
        self.total_importance_weight = 0.0
        self.total_squared_importance_weight = 0.0
        self.total_clipped_samples = 0

    def _draw_target(self, shape: Tuple[int, ...]) -> torch.Tensor:
        global_index = torch.randint(
            self.dst_node_list.numel(), shape, generator=self.generator
        )
        global_candidate = self.dst_node_list[global_index]
        if self.localism_probability >= 1.0:
            return global_candidate
        local_index = torch.randint(
            self.local_dst_node_list.numel(), shape, generator=self.generator
        )
        local_candidate = self.local_dst_node_list[local_index]
        choose_global = (
            torch.rand(shape, generator=self.generator)
            < self.localism_probability
        )
        return torch.where(choose_global, global_candidate, local_candidate)

    def _map_ids(self, node_ids: torch.Tensor) -> torch.Tensor:
        ids = node_ids.detach().cpu().long()
        if self.id_mapper is None:
            return ids
        return self.id_mapper[ids]

    def update_embeddings(
        self, dist_node_ids: torch.Tensor, embeddings: torch.Tensor
    ) -> None:
        """Update the bounded last-seen embedding cache.

        ``dist_node_ids`` are encoded distributed IDs, matching ``nids_mapper``.
        """

        ids = dist_node_ids.detach().reshape(-1).cpu().long()
        values = embeddings.detach().reshape(ids.numel(), -1).cpu().float()
        if ids.numel() != values.shape[0]:
            raise ValueError("dist_node_ids and embeddings must align")
        if self._embedding_dimension not in (0, values.shape[1]):
            raise ValueError("embedding dimension changed during training")
        self._embedding_dimension = int(values.shape[1])
        with self._lock:
            for node_id, value in zip(ids.tolist(), values):
                self._embeddings.pop(int(node_id), None)
                if self.embedding_capacity > 0:
                    self._embeddings[int(node_id)] = value.clone()
            while len(self._embeddings) > self.embedding_capacity:
                self._embeddings.popitem(last=False)

    def _lookup_embeddings(
        self, dist_node_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = dist_node_ids.detach().reshape(-1).cpu().long()
        dimension = self._embedding_dimension
        values = torch.zeros((ids.numel(), dimension), dtype=torch.float32)
        valid = torch.zeros(ids.numel(), dtype=torch.bool)
        with self._lock:
            for index, node_id in enumerate(ids.tolist()):
                value = self._embeddings.get(int(node_id))
                if value is not None:
                    values[index] = value
                    valid[index] = True
        return values, valid

    def _draw_candidate_pool(
        self,
        source_slots: torch.Tensor,
        positive_slots: torch.Tensor,
        timestamp_slots: Optional[torch.Tensor],
    ) -> torch.Tensor:
        shape = (source_slots.numel(), self.pool_factor)
        candidates = self._draw_target(shape)

        source_matrix = source_slots[:, None].expand_as(candidates)
        positive_matrix = positive_slots[:, None].expand_as(candidates)
        timestamp_matrix = (
            timestamp_slots[:, None].expand_as(candidates)
            if timestamp_slots is not None
            else None
        )
        for _ in range(16):
            invalid = candidates == positive_matrix
            if self.validity_filter is not None:
                custom_invalid = ~self.validity_filter(
                    source_matrix, candidates, timestamp_matrix
                ).bool()
                if custom_invalid.shape != candidates.shape:
                    raise ValueError("validity_filter returned an invalid shape")
                invalid |= custom_invalid
            if not bool(invalid.any()):
                return candidates
            candidates[invalid] = self._draw_target(
                (int(invalid.sum().item()),)
            )
        invalid = candidates == positive_matrix
        if self.validity_filter is not None:
            custom_invalid = ~self.validity_filter(
                source_matrix, candidates, timestamp_matrix
            ).bool()
            if custom_invalid.shape != candidates.shape:
                raise ValueError("validity_filter returned an invalid shape")
            invalid |= custom_invalid
        if bool(invalid.any()):
            raise RuntimeError("unable to draw a temporally valid negative pool")
        return candidates

    def sample_for_batch(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        timestamps: Optional[torch.Tensor],
        num_samples: int,
        num_nodes: Optional[int] = None,
        *,
        already_requested: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del num_nodes
        src = src.detach().reshape(-1).cpu().long()
        dst = dst.detach().reshape(-1).cpu().long()
        if src.numel() != dst.numel() or src.numel() == 0:
            raise ValueError("src and dst must be aligned non-empty tensors")
        repeats = int(math.ceil(num_samples / src.numel()))
        source_slots = src.repeat(repeats)[:num_samples]
        positive_slots = dst.repeat(repeats)[:num_samples]
        timestamp_slots = None
        if timestamps is not None:
            base_time = timestamps.detach().reshape(-1).cpu().float()
            if base_time.numel() != src.numel():
                raise ValueError("timestamps must align with src")
            timestamp_slots = base_time.repeat(repeats)[:num_samples]

        candidates = self._draw_candidate_pool(
            source_slots, positive_slots, timestamp_slots
        )
        candidate_dist = self._map_ids(candidates.reshape(-1)).reshape_as(candidates)
        source_dist = self._map_ids(source_slots)

        source_embedding, source_valid = self._lookup_embeddings(source_dist)
        candidate_embedding, candidate_valid = self._lookup_embeddings(
            candidate_dist.reshape(-1)
        )
        candidate_embedding = candidate_embedding.reshape(
            num_samples, self.pool_factor, -1
        )
        candidate_valid = candidate_valid.reshape(num_samples, self.pool_factor)
        if self._embedding_dimension:
            source_normalized = F.normalize(source_embedding, dim=-1)
            candidate_normalized = F.normalize(candidate_embedding, dim=-1)
            similarity = torch.sum(
                source_normalized[:, None, :] * candidate_normalized, dim=-1
            )
            hardness = torch.sigmoid(similarity / self.temperature)
            hardness[~(source_valid[:, None] & candidate_valid)] = 0.5
        else:
            hardness = torch.full(
                (num_samples, self.pool_factor), 0.5, dtype=torch.float32
            )

        owners = _owner(candidate_dist)
        cost = torch.full_like(hardness, self.remote_cost)
        cost[owners == self.local_part] = self.local_cost
        cost[candidate_valid] = torch.minimum(
            cost[candidate_valid],
            torch.full_like(cost[candidate_valid], self.cached_cost),
        )
        reused = torch.zeros_like(candidate_valid)
        if already_requested is not None and already_requested.numel():
            requested_dist = set(
                self._map_ids(already_requested.reshape(-1)).cpu().tolist()
            )
            reused = torch.tensor(
                [
                    int(candidate) in requested_dist
                    for candidate in candidate_dist.reshape(-1).tolist()
                ],
                dtype=torch.bool,
            ).reshape_as(candidate_dist)
            cost[reused] = self.reuse_cost

        value = hardness.clamp_min(1e-12).pow(self.hardness_beta)
        value = value / torch.sqrt(cost + self.cost_floor)
        hard_proposal = value / value.sum(dim=1, keepdim=True)
        proposal = (
            (1.0 - self.exploration) * hard_proposal
            + self.exploration / self.pool_factor
        )
        chosen = torch.multinomial(
            proposal, num_samples=1, replacement=True, generator=self.generator
        )
        row = torch.arange(num_samples)
        selected = candidates[row, chosen[:, 0]]
        selected_probability = proposal[row, chosen[:, 0]]
        if self.localism_probability < 1.0:
            local_fraction = self.local_destination_fraction
            local_target_weight = local_fraction / (
                1.0
                - self.localism_probability
                + self.localism_probability * local_fraction
            )
            target_weight = torch.where(
                owners == self.local_part,
                torch.full_like(proposal, local_target_weight),
                torch.full_like(
                    proposal, 1.0 / self.localism_probability
                ),
            )
        else:
            target_weight = torch.ones_like(proposal)
        importance = (
            target_weight[row, chosen[:, 0]]
            / self.pool_factor
            / selected_probability
        )
        unclipped = importance.clone()
        if self.max_importance_weight is not None:
            importance.clamp_(max=float(self.max_importance_weight))

        self.last_importance_weight = importance
        self.last_proposal_probability = selected_probability
        self.last_hardness = hardness[row, chosen[:, 0]]
        self.last_communication_cost = cost[row, chosen[:, 0]]
        self.last_effective_sample_size = effective_sample_size(importance)
        self.last_clipped_fraction = float(
            (importance != unclipped).float().mean().item()
        )
        selected_dist = candidate_dist[row, chosen[:, 0]]
        selected_owner = owners[row, chosen[:, 0]]
        selected_reused = reused[row, chosen[:, 0]]
        self.total_samples += num_samples
        self.total_remote_samples += int(
            (selected_owner != self.local_part).sum().item()
        )
        self.total_reused_samples += int(selected_reused.sum().item())
        self.total_hardness += float(self.last_hardness.sum().item())
        self.total_communication_cost += float(
            self.last_communication_cost.sum().item()
        )
        self.total_importance_weight += float(
            self.last_importance_weight.sum().item()
        )
        self.total_squared_importance_weight += float(
            self.last_importance_weight.square().sum().item()
        )
        self.total_clipped_samples += int(
            (self.last_importance_weight != unclipped).sum().item()
        )
        # Keep this for callers that want to update the request planner directly.
        self.last_selected_dist_ids = selected_dist
        return selected

    def sample(
        self, num_samples: int, num_nodes: Optional[int] = None
    ) -> torch.Tensor:
        del num_nodes
        index = torch.randint(
            self.dst_node_list.numel(), (num_samples,), generator=self.generator
        )
        self.last_importance_weight = torch.ones(num_samples)
        return self.dst_node_list[index]

    def diagnostics(self) -> Dict[str, float]:
        return {
            "pool_factor": float(self.pool_factor),
            "target_localism_probability": float(
                self.localism_probability
            ),
            "local_destination_fraction": float(
                self.local_destination_fraction
            ),
            "embedding_entries": float(len(self._embeddings)),
            "last_ess": float(self.last_effective_sample_size),
            "last_mean_hardness": (
                float(self.last_hardness.mean().item())
                if self.last_hardness.numel()
                else 0.0
            ),
            "last_mean_cost": (
                float(self.last_communication_cost.mean().item())
                if self.last_communication_cost.numel()
                else 0.0
            ),
            "last_clipped_fraction": float(self.last_clipped_fraction),
            "total_samples": float(self.total_samples),
            "total_remote_samples": float(self.total_remote_samples),
            "total_reused_samples": float(self.total_reused_samples),
            "total_hardness": float(self.total_hardness),
            "total_communication_cost": float(self.total_communication_cost),
            "total_importance_weight": float(self.total_importance_weight),
            "total_squared_importance_weight": float(
                self.total_squared_importance_weight
            ),
            "total_clipped_samples": float(self.total_clipped_samples),
            "cumulative_mean_hardness": (
                self.total_hardness / self.total_samples
                if self.total_samples
                else 0.0
            ),
            "cumulative_mean_cost": (
                self.total_communication_cost / self.total_samples
                if self.total_samples
                else 0.0
            ),
            "cumulative_mean_importance_weight": (
                self.total_importance_weight / self.total_samples
                if self.total_samples
                else 0.0
            ),
            "cumulative_ess": (
                self.total_importance_weight**2
                / self.total_squared_importance_weight
                if self.total_squared_importance_weight
                else 0.0
            ),
            "cumulative_clipped_fraction": (
                self.total_clipped_samples / self.total_samples
                if self.total_samples
                else 0.0
            ),
            "remote_sample_ratio": (
                self.total_remote_samples / self.total_samples
                if self.total_samples
                else 0.0
            ),
            "reuse_sample_ratio": (
                self.total_reused_samples / self.total_samples
                if self.total_samples
                else 0.0
            ),
        }
