import importlib.util
import os
from pathlib import Path

import pytest
import torch


def _load_sampler_module():
    library = os.environ.get("COORDINATED_SAMPLER_NATIVE_LIB")
    if library:
        path = Path(library)
        spec = importlib.util.spec_from_file_location(
            "libstarrygl_sampler", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    import starrygl

    module = getattr(starrygl, "sampler_ops", None)
    if module is None or not hasattr(
        module.ParallelSampler, "configure_coordinated_sampling"
    ):
        pytest.skip(
            "native sampler is stale; build it and set "
            "COORDINATED_SAMPLER_NATIVE_LIB"
        )
    return module


def _make_sampler(module, graph_name, policy, *, group_size=8, seed=19):
    # Both copies of root 0 see the same single remote node. Its occurrence
    # probability is exactly the boundary probability.
    neighbors = module.get_neighbors(
        graph_name,
        torch.tensor([1], dtype=torch.long),
        torch.tensor([0], dtype=torch.long),
        2,
        0,
        torch.tensor([0], dtype=torch.long),
        None,
        None,
        torch.tensor([1], dtype=torch.long),
    )
    sampler = module.ParallelSampler(
        neighbors,
        2,
        1,
        1,
        [1],
        1,
        policy,
        0,
        torch.ones(1, dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        0.4,
    )
    if policy == "boundery_recent_decay_coordinated":
        sampler.configure_coordinated_sampling(group_size, seed)
    return sampler


def _duplicate_root_rates(sampler, draws=4000):
    occurrence_hits = torch.zeros(2, dtype=torch.long)
    union_hits = 0
    mismatches = 0
    for _ in range(draws):
        sampler.neighbor_sample_from_nodes(
            torch.tensor([0, 0]),
            torch.tensor([2, 2]),
            None,
            0.4,
        )
        result = sampler.get_ret()[0]
        selected = torch.zeros(2, dtype=torch.bool)
        source_index = result.src_index()
        if source_index.numel():
            selected[source_index.unique()] = True
        occurrence_hits += selected.long()
        union_hits += int(selected.any())
        mismatches += int(selected[0] != selected[1])
    return occurrence_hits.float() / draws, union_hits / draws, mismatches


def test_coordinated_sampling_preserves_occurrence_marginals_and_reduces_union():
    module = _load_sampler_module()
    coordinated = _make_sampler(
        module,
        "coordinated-{}".format(os.getpid()),
        "boundery_recent_decay_coordinated",
    )
    baseline = _make_sampler(
        module,
        "baseline-{}".format(os.getpid()),
        "boundery_recent_decay",
    )

    coordinated_rate, coordinated_union, mismatches = _duplicate_root_rates(
        coordinated
    )
    baseline_rate, baseline_union, _ = _duplicate_root_rates(baseline)

    expected_occurrence = torch.full((2,), 0.4)
    assert torch.allclose(
        coordinated_rate, expected_occurrence, atol=0.025
    )
    assert torch.allclose(baseline_rate, expected_occurrence, atol=0.025)
    assert mismatches == 0
    assert abs(coordinated_union - 0.4) < 0.025
    assert abs(baseline_union - 0.64) < 0.035
    assert coordinated_union < baseline_union - 0.18


def test_different_probabilities_share_one_uniform_without_inflating_the_lower_one():
    module = _load_sampler_module()
    # Root 0 sees shared node 2 as its newest neighbor; root 1 sees the same
    # node as its older neighbor. The two occurrence probabilities differ,
    # but both compare their own probability against one shared U.
    neighbors = module.get_neighbors(
        "nested-{}".format(os.getpid()),
        torch.tensor([3, 2, 2, 4], dtype=torch.long),
        torch.tensor([0, 0, 1, 1], dtype=torch.long),
        5,
        0,
        torch.arange(4, dtype=torch.long),
        None,
        None,
        torch.tensor([8, 9, 8, 9], dtype=torch.long),
    )
    sampler = module.ParallelSampler(
        neighbors,
        5,
        4,
        1,
        [2],
        1,
        "boundery_recent_decay_coordinated",
        0,
        torch.ones(4, dtype=torch.int32),
        torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32),
        0.5,
    )
    sampler.configure_coordinated_sampling(8, 23)

    draws = 5000
    hits = torch.zeros(2, dtype=torch.long)
    for _ in range(draws):
        sampler.neighbor_sample_from_nodes(
            torch.tensor([0, 1]),
            torch.tensor([10, 10]),
            None,
            0.5,
        )
        result = sampler.get_ret()[0]
        eids = result.eid()
        hits[0] += int(bool((eids == 1).any()))
        hits[1] += int(bool((eids == 2).any()))
        # The lower-probability occurrence (eid 2) is a subset of eid 1.
        assert not bool((eids == 2).any()) or bool((eids == 1).any())

    temporal_new = torch.exp(torch.tensor(-0.5))
    temporal_old = torch.exp(torch.tensor(-1.0))
    expected = torch.tensor(
        [
            float(temporal_new / (temporal_new + temporal_old)),
            float(temporal_old / (temporal_new + temporal_old)),
        ]
    )
    empirical = hits.float() / draws
    assert torch.allclose(empirical, expected, atol=0.025)
