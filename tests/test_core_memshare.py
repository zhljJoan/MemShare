import math

import dgl
import pytest
import torch

from starrygl.module.layers import TransfomerAttentionLayer
from starrygl.sample.batch_data import to_block
from starrygl.sample.data_loader import DistributedDataLoader
from starrygl.sample.core_memshare import (
    CommunicationAwareNegativeSampling,
    LastFetchedMemoryPredictor,
    ProbabilityAwareEdgeCache,
    UnifiedRequestPlanner,
    communication_aware_probabilities,
    coordinated_bernoulli,
    residual_correct,
    residual_ratio_aggregate,
)
from starrygl.sample.part_utils.transformer_from_speed import _load_partition_ids


def _dist_id(part, local):
    return (int(part) << 50) | int(local)


def test_communication_aware_probabilities_obey_budget_and_order():
    residual = torch.tensor([1.0, 2.0, 1.0])
    cost = torch.tensor([1.0, 1.0, 4.0])
    probability = communication_aware_probabilities(
        residual,
        cost,
        budget=2.0,
        min_probability=0.05,
    )
    assert probability[1] > probability[0] > probability[2]
    assert torch.all(probability >= 0.05)
    assert math.isclose(float((probability * cost).sum()), 2.0, rel_tol=1e-5)


def test_residual_correction_is_unbiased():
    generator = torch.Generator().manual_seed(7)
    prediction = torch.tensor([[0.5, -1.0], [2.0, 0.25]])
    actual = torch.tensor([[2.0, 3.0], [-1.0, 4.0]])
    probability = torch.tensor([0.2, 0.65])
    draws = 100000
    selected = (
        torch.rand((draws, 2), generator=generator)
        < probability.unsqueeze(0)
    )
    estimate = residual_correct(
        prediction.unsqueeze(0).expand(draws, -1, -1),
        actual.unsqueeze(0).expand(draws, -1, -1),
        selected,
        probability.unsqueeze(0).expand(draws, -1),
    )
    assert torch.allclose(estimate.mean(dim=0), actual, atol=0.035, rtol=0.02)


def test_coordinated_sampling_preserves_marginals_and_nests_decisions():
    ids = torch.tensor([11, 11])
    groups = torch.tensor([3, 3])
    probabilities = torch.tensor([0.2, 0.6])
    samples = []
    for batch_id in range(5000):
        decision = coordinated_bernoulli(
            ids, probabilities, groups, seed=19, batch_id=batch_id
        )
        # Sharing one uniform makes the low-probability decision a subset.
        assert not bool(decision[0]) or bool(decision[1])
        samples.append(decision)
    empirical = torch.stack(samples).float().mean(dim=0)
    assert torch.allclose(empirical, probabilities, atol=0.025)
    union_rate = torch.stack(samples).any(dim=1).float().mean()
    assert abs(float(union_rate) - 0.6) < 0.025


def test_residual_ratio_aggregate_corrects_components_before_ratio():
    actual_numerator = torch.tensor([[2.0], [6.0], [4.0]])
    actual_denominator = torch.tensor([[1.0], [2.0], [1.0]])
    zeros_numerator = torch.zeros_like(actual_numerator)
    zeros_denominator = torch.zeros_like(actual_denominator)
    selected = torch.ones(3, dtype=torch.bool)
    probability = torch.ones(3)
    destination = torch.tensor([0, 0, 1])
    ratio, numerator, denominator = residual_ratio_aggregate(
        actual_numerator,
        actual_denominator,
        zeros_numerator,
        zeros_denominator,
        selected,
        probability,
        destination,
        2,
    )
    assert torch.equal(numerator, torch.tensor([[8.0], [4.0]]))
    assert torch.equal(denominator, torch.tensor([[3.0], [1.0]]))
    assert torch.allclose(ratio, torch.tensor([[8.0 / 3.0], [4.0]]))


def test_last_fetched_predictor_uses_lru_then_owner_fallback():
    predictor = LastFetchedMemoryPredictor(
        dimension=2, capacity=1, uncertainty_ema=0.0, centroid_ema=0.0
    )
    first_residual = predictor.observe(
        torch.tensor([10]),
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([5.0]),
        owner_ids=torch.tensor([2]),
    )
    assert torch.equal(first_residual, torch.tensor([1.0]))
    predictor.observe(
        torch.tensor([11]),
        torch.tensor([[3.0, 4.0]]),
        torch.tensor([7.0]),
        owner_ids=torch.tensor([2]),
    )
    prediction = predictor.predict(
        torch.tensor([11, 10, 99]),
        torch.tensor([9.0, 9.0, 9.0]),
        owner_ids=torch.tensor([2, 2, 7]),
    )
    assert prediction.source.tolist() == [2, 1, 0]
    assert torch.equal(prediction.value[0], torch.tensor([3.0, 4.0]))
    assert torch.equal(prediction.value[1], torch.tensor([3.0, 4.0]))
    assert prediction.age[0] == 2.0
    assert math.isinf(float(prediction.age[1]))


def test_probability_cache_admits_high_value_and_evicts_low_value():
    cache = ProbabilityAwareEdgeCache(
        capacity=2,
        feature_shape=(2,),
        dtype=torch.float32,
        device=torch.device("cpu"),
        min_residency=0,
        max_replacements_per_batch=2,
    )
    cache.admit(
        torch.tensor([1, 2]),
        torch.tensor([[1.0, 1.0], [2.0, 2.0]]),
        access_probability=torch.tensor([1.0, 0.1]),
    )
    cache.admit(
        torch.tensor([3]),
        torch.tensor([[3.0, 3.0]]),
        access_probability=torch.tensor([1.0]),
        communication_cost=torch.tensor([2.0]),
    )
    lookup = cache.lookup(torch.tensor([1, 2, 3]))
    assert lookup.hit_mask.tolist() == [True, False, True]
    assert torch.equal(lookup.values[0], torch.tensor([1.0, 1.0]))
    assert torch.equal(lookup.values[2], torch.tensor([3.0, 3.0]))
    stats = cache.stats()
    assert stats["evictions"] == 1.0
    assert stats["object_hit_rate"] == 2.0 / 3.0


def test_two_stage_negative_importance_correction_recovers_target_mean():
    sampler = CommunicationAwareNegativeSampling(
        amount=1,
        dst_node_list=torch.tensor([1, 2, 3]),
        id_mapper=torch.arange(4),
        pool_factor=4,
        exploration=0.1,
        hardness_beta=4.0,
        seed=123,
    )
    sampler.update_embeddings(
        torch.tensor([0, 1, 2, 3]),
        torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
            ]
        ),
    )
    draws = 30000
    selected = sampler.sample_for_batch(
        torch.tensor([0]),
        torch.tensor([0]),
        None,
        num_samples=draws,
    )
    loss_by_node = torch.tensor([0.0, 1.0, 5.0, 9.0])
    estimate = (
        loss_by_node[selected] * sampler.last_importance_weight
    ).mean()
    assert abs(float(estimate) - 5.0) < 0.12
    assert float(sampler.last_importance_weight.max()) <= 10.0 + 1e-5
    assert sampler.last_effective_sample_size > 0


def test_localism_and_reranking_corrections_recover_global_target():
    mapper = torch.tensor(
        [
            _dist_id(0, 0),
            _dist_id(0, 1),
            _dist_id(0, 2),
            _dist_id(1, 0),
            _dist_id(1, 1),
        ]
    )
    sampler = CommunicationAwareNegativeSampling(
        amount=1,
        dst_node_list=torch.tensor([1, 2, 3, 4]),
        id_mapper=mapper,
        local_part=0,
        pool_factor=4,
        localism_probability=0.2,
        local_cost=1.0,
        cached_cost=1.0,
        reuse_cost=1.0,
        remote_cost=1.0,
        exploration=0.1,
        hardness_beta=4.0,
        seed=321,
    )
    sampler.update_embeddings(
        mapper,
        torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [-1.0, 0.0],
                [0.5, 0.5],
            ]
        ),
    )
    draws = 60000
    selected = sampler.sample_for_batch(
        torch.tensor([0]),
        torch.tensor([0]),
        None,
        num_samples=draws,
    )
    value = torch.tensor([0.0, 1.0, 3.0, 7.0, 9.0])
    estimate = (value[selected] * sampler.last_importance_weight).mean()
    assert abs(float(estimate) - 5.0) < 0.15
    assert sampler.diagnostics()["target_localism_probability"] == 0.2


def test_hard_negative_rejects_pool_when_validity_filter_cannot_be_satisfied():
    sampler = CommunicationAwareNegativeSampling(
        amount=1,
        dst_node_list=torch.tensor([1, 2]),
        pool_factor=2,
        validity_filter=lambda source, candidate, timestamp: torch.zeros_like(
            candidate, dtype=torch.bool
        ),
        seed=123,
    )
    with pytest.raises(RuntimeError, match="temporally valid"):
        sampler.sample_for_batch(
            torch.tensor([0]),
            torch.tensor([3]),
            torch.tensor([1.0]),
            num_samples=1,
        )


def test_request_planner_deduplicates_and_splits_cached_edges():
    node_ids = torch.tensor(
        [_dist_id(0, 1), _dist_id(1, 2), _dist_id(1, 2), _dist_id(2, 3)]
    )
    edge_ids = torch.tensor(
        [_dist_id(1, 7), _dist_id(1, 7), _dist_id(2, 8)]
    )
    planner = UnifiedRequestPlanner(local_rank=0)
    plan = planner.plan(
        node_ids,
        edge_ids,
        raw_node_ids=node_ids,
        raw_edge_ids=edge_ids,
        edge_cached=torch.tensor([True, False]),
    )
    assert plan.unique_node_requests == 3
    assert plan.unique_edge_requests == 2
    assert plan.cached_edge_ids.tolist() == [_dist_id(1, 7)]
    assert plan.missing_edge_ids.tolist() == [_dist_id(2, 8)]
    assert set(plan.nodes_by_owner) == {1, 2}
    stats = planner.stats()
    assert stats["node_dedup_ratio"] == 0.25
    assert math.isclose(stats["edge_dedup_ratio"], 1.0 / 3.0)


def test_empty_edge_observation_is_a_noop():
    class Sampler:
        policy = "boundery_recent_core"

        def update_core_edge_state(self, *args):
            raise AssertionError("empty edge state must not reach the sampler")

    loader = object.__new__(DistributedDataLoader)
    loader.sampler = Sampler()
    batch_data = (None, None, {"core_global_eids": torch.empty(0, dtype=torch.long)})
    loader._observe_edge_state(
        batch_data,
        torch.empty(0, dtype=torch.long),
        torch.empty((0, 172)),
    )


def test_memory_predictor_only_observes_remote_fetches(monkeypatch):
    class Predictor:
        def observe(self, node_ids, memory, timestamps, *, owner_ids):
            self.observed = (
                node_ids.clone(),
                memory.clone(),
                timestamps.clone(),
                owner_ids.clone(),
            )
            return torch.ones(node_ids.numel())

    class Sampler:
        def observe_core_fetch(self, node_ids, residual, timestamps):
            self.observed = (
                node_ids.clone(),
                residual.clone(),
                timestamps.clone(),
            )

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    loader = object.__new__(DistributedDataLoader)
    loader.memory_predictor = Predictor()
    loader.sampler = Sampler()
    dist_nids = torch.tensor(
        [_dist_id(0, 1), _dist_id(1, 2), _dist_id(2, 3)]
    )
    batch_data = (
        None,
        None,
        {"core_global_nids": torch.tensor([10, 11, 12])},
    )
    memory = (
        torch.tensor([[1.0], [2.0], [3.0]]),
        torch.tensor([4.0, 5.0, 6.0]),
    )
    loader._observe_fetched_memory(batch_data, dist_nids, memory)
    observed = loader.memory_predictor.observed
    assert observed[0].tolist() == [11, 12]
    assert observed[1].reshape(-1).tolist() == [2.0, 3.0]
    assert observed[2].tolist() == [5.0, 6.0]
    assert observed[3].tolist() == [1, 2]
    assert loader.sampler.observed[0].tolist() == [11, 12]


def test_attention_uses_inverse_inclusion_probability():
    block = dgl.create_block(
        (torch.tensor([1, 2]), torch.tensor([0, 0])),
        num_src_nodes=3,
        num_dst_nodes=1,
    )
    block.edata['sample_prob'] = torch.tensor([0.5, 1.0])
    layer = TransfomerAttentionLayer(
        dim_node_feat=1,
        dim_edge_feat=0,
        dim_time=0,
        num_head=1,
        dropout=0.0,
        att_dropout=0.0,
        dim_out=1,
    )
    attention = layer._edge_attention(block, torch.zeros(2, 1))
    assert torch.allclose(
        attention.reshape(-1), torch.tensor([2.0 / 3.0, 1.0 / 3.0])
    )


def test_mfg_builder_only_remaps_index_metadata():
    class FakeRet:
        def eid(self):
            return torch.tensor([0])

        def sample_nodes(self):
            return torch.tensor([3])

        def sample_nodes_ts(self):
            return torch.tensor([9])

        def delta_ts(self):
            return torch.tensor([1])

        def src_index(self):
            return torch.tensor([0])

        def sample_weight(self):
            return torch.tensor([0.25])

    class FakeGraph:
        eids_mapper = torch.tensor([0])
        nids_mapper = torch.arange(4)

    class FakeData:
        pass

    data = FakeData()
    metadata = {
        'seed': torch.tensor([0, 1, 2]),
        'seed_ts': torch.tensor([10, 10, 10]),
        'src_pos_index': torch.tensor([0]),
        'dst_pos_index': torch.tensor([1]),
        'dst_neg_index': torch.tensor([2]),
        'src_neg_index': torch.empty(0, dtype=torch.long),
        'neg_importance_weight': torch.tensor([2.5]),
        'neg_hardness': torch.tensor([0.75]),
    }
    (result, dist_nid, dist_eid) = to_block(
        FakeGraph(),
        data,
        ([FakeRet()], metadata),
        device=torch.device('cpu'),
    )
    _, mfgs, remapped = result
    assert remapped['neg_importance_weight'].tolist() == [2.5]
    assert remapped['neg_hardness'].tolist() == [0.75]
    assert remapped['dst_neg_index'].numel() == 1
    assert remapped['core_global_nids'].tolist() == [0, 1, 2, 3]
    assert remapped['core_global_eids'].tolist() == [0]
    assert mfgs[0][0].edata['sample_prob'].tolist() == [0.25]
    assert mfgs[0][0].srcdata['NID'].tolist() == [0, 1, 2, 3]
    assert dist_nid.tolist() == [0, 1, 2, 3]
    assert dist_eid.tolist() == [0]


def test_partition_id_loader_handles_empty_and_invalid_files(tmp_path):
    valid = tmp_path / "valid.txt"
    valid.write_text("3\n\n7\n", encoding="utf-8")
    assert _load_partition_ids(valid).tolist() == [3, 7]

    missing = tmp_path / "missing.txt"
    assert _load_partition_ids(missing, optional=True).numel() == 0
    with pytest.raises(FileNotFoundError, match="partition ID"):
        _load_partition_ids(missing)

    invalid = tmp_path / "invalid.txt"
    invalid.write_text("3\nnot-an-id\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid ID"):
        _load_partition_ids(invalid)
