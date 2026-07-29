"""Unified positive/negative remote request planning and accounting."""

from dataclasses import dataclass
import threading
from typing import Dict, Optional

import torch


def _owner(dist_ids: torch.Tensor) -> torch.Tensor:
    return (dist_ids.long() >> 50) & 0xFFFF


@dataclass
class RequestPlan:
    node_ids: torch.Tensor
    edge_ids: torch.Tensor
    static_node_ids: torch.Tensor
    dynamic_memory_ids: torch.Tensor
    missing_edge_ids: torch.Tensor
    cached_edge_ids: torch.Tensor
    nodes_by_owner: Dict[int, torch.Tensor]
    edges_by_owner: Dict[int, torch.Tensor]
    raw_node_requests: int
    raw_edge_requests: int

    @property
    def unique_node_requests(self) -> int:
        return int(self.node_ids.numel())

    @property
    def unique_edge_requests(self) -> int:
        return int(self.edge_ids.numel())


class UnifiedRequestPlanner:
    """Deduplicate typed requests before owner bucketing."""

    def __init__(self, local_rank: int = 0) -> None:
        self.local_rank = int(local_rank)
        self._lock = threading.RLock()
        self.batches = 0
        self.raw_node_requests = 0
        self.raw_edge_requests = 0
        self.unique_node_requests = 0
        self.unique_edge_requests = 0
        self.remote_node_requests = 0
        self.remote_edge_requests = 0
        self.cached_edge_requests = 0

    @staticmethod
    def _unique(ids: torch.Tensor) -> torch.Tensor:
        return torch.unique(ids.reshape(-1), sorted=True)

    @staticmethod
    def _bucket(ids: torch.Tensor) -> Dict[int, torch.Tensor]:
        owners = _owner(ids)
        result = {}
        for owner in torch.unique(owners).tolist():
            result[int(owner)] = ids[owners == int(owner)]
        return result

    def plan(
        self,
        node_ids: torch.Tensor,
        edge_ids: torch.Tensor,
        *,
        raw_node_ids: Optional[torch.Tensor] = None,
        raw_edge_ids: Optional[torch.Tensor] = None,
        raw_node_count: Optional[int] = None,
        raw_edge_count: Optional[int] = None,
        edge_cached: Optional[torch.Tensor] = None,
    ) -> RequestPlan:
        nodes = self._unique(node_ids)
        edges = self._unique(edge_ids)
        if edge_cached is None:
            cached = torch.zeros(edges.numel(), dtype=torch.bool, device=edges.device)
        else:
            cached = edge_cached.reshape(-1).to(device=edges.device, dtype=torch.bool)
            if cached.numel() != edges.numel():
                raise ValueError("edge_cached must align with unique edge_ids")

        node_owner = _owner(nodes)
        edge_owner = _owner(edges)
        remote_nodes = nodes[node_owner != self.local_rank]
        remote_edges = edges[edge_owner != self.local_rank]
        remote_cached = cached[edge_owner != self.local_rank]
        cached_edges = remote_edges[remote_cached]
        missing_edges = remote_edges[~remote_cached]
        raw_nodes = int(
            raw_node_count
            if raw_node_count is not None
            else raw_node_ids.numel()
            if raw_node_ids is not None
            else node_ids.numel()
        )
        raw_edges = int(
            raw_edge_count
            if raw_edge_count is not None
            else raw_edge_ids.numel()
            if raw_edge_ids is not None
            else edge_ids.numel()
        )
        if raw_nodes < nodes.numel() or raw_edges < edges.numel():
            raise ValueError("raw request counts cannot be smaller than unique counts")
        plan = RequestPlan(
            node_ids=nodes,
            edge_ids=edges,
            static_node_ids=remote_nodes,
            dynamic_memory_ids=remote_nodes,
            missing_edge_ids=missing_edges,
            cached_edge_ids=cached_edges,
            nodes_by_owner=self._bucket(remote_nodes),
            edges_by_owner=self._bucket(missing_edges),
            raw_node_requests=raw_nodes,
            raw_edge_requests=raw_edges,
        )
        with self._lock:
            self.batches += 1
            self.raw_node_requests += raw_nodes
            self.raw_edge_requests += raw_edges
            self.unique_node_requests += nodes.numel()
            self.unique_edge_requests += edges.numel()
            self.remote_node_requests += remote_nodes.numel()
            self.remote_edge_requests += missing_edges.numel()
            self.cached_edge_requests += cached_edges.numel()
        return plan

    def stats(self) -> Dict[str, float]:
        with self._lock:
            node_dedup = (
                1.0 - self.unique_node_requests / self.raw_node_requests
                if self.raw_node_requests
                else 0.0
            )
            edge_dedup = (
                1.0 - self.unique_edge_requests / self.raw_edge_requests
                if self.raw_edge_requests
                else 0.0
            )
            return {
                "batches": float(self.batches),
                "raw_node_requests": float(self.raw_node_requests),
                "raw_edge_requests": float(self.raw_edge_requests),
                "unique_node_requests": float(self.unique_node_requests),
                "unique_edge_requests": float(self.unique_edge_requests),
                "remote_node_requests": float(self.remote_node_requests),
                "remote_edge_requests": float(self.remote_edge_requests),
                "cached_edge_requests": float(self.cached_edge_requests),
                "node_dedup_ratio": node_dedup,
                "edge_dedup_ratio": edge_dedup,
            }
