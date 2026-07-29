import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from starrygl.distributed.context import _set_default_dist_context
from starrygl.distributed.utils import DistIndex, DistributedTensor
from starrygl.sample.batch_data import (
    get_edge_feature_by_dist,
    get_edge_feature_by_dist_async_cached,
)
from starrygl.sample.core_memshare import ProbabilityAwareEdgeCache


class _TestContext:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self._use_rpc = False
        self.memory_gloo_group = dist.group.WORLD
        self.memory_nccl_group = dist.group.WORLD


class _TestGraph:
    def __init__(self, local_feature):
        self.efeat = DistributedTensor(local_feature)
        self.feature_device = torch.device("cpu")
        self.use_pin = False

    def get_local_efeat(self, index, out_device=torch.device("cpu")):
        location = DistIndex(index).loc
        return self.efeat.accessor.data[location].to(out_device)

    def get_dist_efeat(
        self,
        idx,
        *,
        group,
        out_device,
        is_async=False,
    ):
        return self.efeat.all_to_all_get(
            dist_index=idx,
            group=group,
            device=out_device,
            is_async=is_async,
        )


def _dist_id(part, local):
    return (int(part) << 50) | int(local)


def _cache_worker(rank, world_size, rendezvous, result_queue):
    try:
        dist.init_process_group(
            "gloo",
            init_method="file://{}".format(rendezvous),
            rank=rank,
            world_size=world_size,
        )
        _set_default_dist_context(_TestContext(rank, world_size))
        local_feature = torch.tensor(
            [
                [100.0 * rank, 100.0 * rank + 10.0],
                [100.0 * rank + 1.0, 100.0 * rank + 11.0],
            ]
        )
        graph = _TestGraph(local_feature)
        cache = ProbabilityAwareEdgeCache(
            capacity=4,
            feature_shape=(2,),
            dtype=torch.float32,
            device=torch.device("cpu"),
            min_residency=0,
        )
        other = 1 - rank
        query = torch.tensor(
            sorted([_dist_id(rank, 0), _dist_id(other, 1)]),
            dtype=torch.long,
        )
        expected = torch.stack(
            [
                local_feature[0]
                if DistIndex(item.reshape(1)).part.item() == rank
                else torch.tensor([100.0 * other + 1.0, 100.0 * other + 11.0])
                for item in query
            ]
        )

        first = get_edge_feature_by_dist(
            graph,
            query,
            is_local=False,
            out_device=torch.device("cpu"),
            cache=cache,
            access_probability=torch.ones(2),
            communication_cost=2.0,
        )
        assert torch.equal(first, expected)
        assert len(cache) == 1

        # The second request is a cache hit on both ranks, but every rank must
        # still enter the empty miss collective.
        second, work = get_edge_feature_by_dist_async_cached(
            graph,
            query,
            cache,
            out_device=torch.device("cpu"),
            access_probability=torch.ones(2),
            communication_cost=2.0,
        )
        work.wait()
        assert torch.equal(second, expected)
        assert cache.stats()["hits"] == 1.0
        result_queue.put(None)
    except Exception as error:
        result_queue.put(repr(error))
    finally:
        _set_default_dist_context(None)
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="gloo distributed backend is required",
)
def test_probability_cache_miss_only_all_to_all(tmp_path):
    world_size = 2
    rendezvous = str(tmp_path / "core-cache-rendezvous")
    context = mp.get_context("fork")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_cache_worker,
            args=(rank, world_size, rendezvous, result_queue),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    errors = [result_queue.get(timeout=30) for _ in processes]
    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join()
    assert all(process.exitcode == 0 for process in processes)
    assert errors == [None, None]
