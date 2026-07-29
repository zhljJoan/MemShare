import importlib.util
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest
import torch


def _load_sampler_module():
    library = os.environ.get("CORE_MEMSHARE_NATIVE_LIB")
    if library:
        path = pathlib.Path(library)
        spec = importlib.util.spec_from_file_location("libstarrygl_sampler", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    import starrygl

    module = getattr(starrygl, "sampler_ops", None)
    if module is None or not hasattr(
        module.ParallelSampler, "configure_core_sampling"
    ):
        pytest.skip(
            "installed sampler predates CoRe-MemShare; build it and set "
            "CORE_MEMSHARE_NATIVE_LIB"
        )
    return module


def test_native_core_sampler_exports_probabilities_and_coordinates_roots():
    library = os.environ.get("CORE_MEMSHARE_NATIVE_LIB")
    if library:
        script = textwrap.dedent(
            """
            import importlib.util
            import os
            import pathlib
            import sys
            import torch

            path = pathlib.Path(sys.argv[1])
            spec = importlib.util.spec_from_file_location("libstarrygl_sampler", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            row = torch.tensor([1, 2, 3, 4], dtype=torch.long)
            col = torch.tensor([0, 0, 0, 0], dtype=torch.long)
            edge_ids = torch.arange(4, dtype=torch.long)
            timestamp = torch.tensor([1, 2, 3, 4], dtype=torch.long)
            neighbors = module.get_neighbors(
                "core-native-subprocess-{}".format(os.getpid()),
                row, col, 5, 0, edge_ids, None, None, timestamp
            )
            sampler = module.ParallelSampler(
                neighbors, 5, 4, 1, [4], 1, "boundery_recent_core", 0,
                torch.ones(4, dtype=torch.int32),
                torch.tensor([0, 1, 1, 1, 1], dtype=torch.int32),
                0.5,
            )
            sampler.configure_core_sampling(
                0.05, 1.0, 0.25, 0.5, 1.0, 1.0, 1.0, 0.0, 0.05,
                0, 8, 19
            )
            sampler.update_core_node_state(
                torch.tensor([1, 2, 3, 4]),
                torch.tensor([0.1, 0.2, 1.0, 2.0]),
                torch.tensor([9.0, 9.0, 9.0, 9.0]),
            )
            sampler.update_core_edge_state(
                edge_ids,
                torch.tensor([1.0, 1.0, 0.0, 0.0]),
                torch.tensor([False, False, True, True]),
            )
            sampler.neighbor_sample_from_nodes(
                torch.tensor([0, 0]), torch.tensor([10, 10]), None, 0.5
            )
            result = sampler.get_ret()[0]
            sampled_eids = result.eid()
            source_index = result.src_index()
            probability = result.sample_weight()
            assert probability.numel() == sampled_eids.numel()
            assert bool(((probability > 0) & (probability <= 1)).all())
            assert set(sampled_eids[source_index == 0].tolist()) == set(
                sampled_eids[source_index == 1].tolist()
            )

            coordinated = module.ParallelSampler(
                neighbors, 5, 4, 1, [4], 1,
                "boundery_recent_decay_coordinated", 0,
                torch.ones(4, dtype=torch.int32),
                torch.tensor([0, 1, 1, 1, 1], dtype=torch.int32),
                0.5,
            )
            coordinated.configure_core_sampling(
                0.05, 1.0, 0.25, 0.5, 1.0,
                1.0, 1.0, 0.0, 0.05, 0, 8, 19
            )
            coordinated.neighbor_sample_from_nodes(
                torch.tensor([0, 0]), torch.tensor([10, 10]), None, 0.5
            )
            coordinated_result = coordinated.get_ret()[0]
            coordinated_eids = coordinated_result.eid()
            coordinated_sources = coordinated_result.src_index()
            assert set(
                coordinated_eids[coordinated_sources == 0].tolist()
            ) == set(
                coordinated_eids[coordinated_sources == 1].tolist()
            )
            assert coordinated_result.sample_weight().numel() == 0

            # A rejected recent remote candidate must not be replaced by an
            # older local edge: that replacement would have a random,
            # unreported inclusion probability and invalidate HT correction.
            no_backfill_neighbors = module.get_neighbors(
                "core-native-no-backfill-{}".format(os.getpid()),
                torch.tensor([1, 2], dtype=torch.long),
                torch.tensor([0, 0], dtype=torch.long),
                3,
                0,
                torch.tensor([0, 1], dtype=torch.long),
                None,
                None,
                torch.tensor([1, 2], dtype=torch.long),
            )
            no_backfill = module.ParallelSampler(
                no_backfill_neighbors, 3, 2, 1, [1], 1,
                "boundery_recent_core", 0,
                torch.tensor([0, 1], dtype=torch.int32),
                torch.tensor([0, 0, 1], dtype=torch.int32),
                1e-9,
            )
            no_backfill.configure_core_sampling(
                1e-9, 1.0, 0.25, 0.5, 1.0,
                1.0, 1.0, 0.0, 0.05, 0, 8, 19
            )
            no_backfill.neighbor_sample_from_nodes(
                torch.tensor([0]), torch.tensor([3]), None, 1e-9
            )
            assert 0 not in no_backfill.get_ret()[0].eid().tolist()

            # The deterministic prefix protects the newest candidate even
            # when the residual tail budget is effectively zero.
            top_one = module.ParallelSampler(
                no_backfill_neighbors, 3, 2, 1, [2], 1,
                "boundery_recent_core", 0,
                torch.tensor([0, 1], dtype=torch.int32),
                torch.tensor([0, 0, 1], dtype=torch.int32),
                1e-9,
            )
            top_one.configure_core_sampling(
                1e-9, 1.0, 0.25, 0.5, 1.0,
                1.0, 1.0, 0.0, 0.05, 1, 8, 19
            )
            top_one.neighbor_sample_from_nodes(
                torch.tensor([0]), torch.tensor([3]), None, 1e-9
            )
            top_one_result = top_one.get_ret()[0]
            newest = top_one_result.eid() == 1
            assert newest.sum().item() == 1
            assert top_one_result.sample_weight()[newest].item() == 1.0
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", script, library],
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        return

    module = _load_sampler_module()
    row = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    col = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    edge_ids = torch.arange(4, dtype=torch.long)
    timestamp = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    graph_name = "core-native-test-{}".format(os.getpid())
    neighbors = module.get_neighbors(
        graph_name, row, col, 5, 0, edge_ids, None, None, timestamp
    )
    sampler = module.ParallelSampler(
        neighbors,
        5,
        4,
        1,
        [4],
        1,
        "boundery_recent_core",
        0,
        torch.ones(4, dtype=torch.int32),
        torch.tensor([0, 1, 1, 1, 1], dtype=torch.int32),
        0.5,
    )
    sampler.configure_core_sampling(
        0.05, 1.0, 0.25, 0.5, 1.0, 1.0, 1.0, 0.0, 0.05,
        0, 8, 19
    )
    sampler.update_core_node_state(
        torch.tensor([1, 2, 3, 4]),
        torch.tensor([0.1, 0.2, 1.0, 2.0]),
        torch.tensor([9.0, 9.0, 9.0, 9.0]),
    )
    sampler.update_core_edge_state(
        edge_ids,
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
        torch.tensor([False, False, True, True]),
    )
    sampler.neighbor_sample_from_nodes(
        torch.tensor([0, 0]), torch.tensor([10, 10]), None, 0.5
    )
    result = sampler.get_ret()[0]
    sampled_eids = result.eid()
    source_index = result.src_index()
    probability = result.sample_weight()
    assert probability.numel() == sampled_eids.numel()
    assert bool(((probability > 0) & (probability <= 1)).all())
    # Both roots are in one coordinated group, so their selected object sets match.
    assert set(sampled_eids[source_index == 0].tolist()) == set(
        sampled_eids[source_index == 1].tolist()
    )

    coordinated = module.ParallelSampler(
        neighbors,
        5,
        4,
        1,
        [4],
        1,
        "boundery_recent_decay_coordinated",
        0,
        torch.ones(4, dtype=torch.int32),
        torch.tensor([0, 1, 1, 1, 1], dtype=torch.int32),
        0.5,
    )
    coordinated.configure_core_sampling(
        0.05, 1.0, 0.25, 0.5, 1.0, 1.0, 1.0, 0.0, 0.05,
        0, 8, 19
    )
    coordinated.neighbor_sample_from_nodes(
        torch.tensor([0, 0]), torch.tensor([10, 10]), None, 0.5
    )
    coordinated_result = coordinated.get_ret()[0]
    coordinated_eids = coordinated_result.eid()
    coordinated_sources = coordinated_result.src_index()
    assert set(
        coordinated_eids[coordinated_sources == 0].tolist()
    ) == set(
        coordinated_eids[coordinated_sources == 1].tolist()
    )
    assert coordinated_result.sample_weight().numel() == 0

    no_backfill_neighbors = module.get_neighbors(
        "core-native-no-backfill-{}".format(os.getpid()),
        torch.tensor([1, 2], dtype=torch.long),
        torch.tensor([0, 0], dtype=torch.long),
        3,
        0,
        torch.tensor([0, 1], dtype=torch.long),
        None,
        None,
        torch.tensor([1, 2], dtype=torch.long),
    )
    no_backfill = module.ParallelSampler(
        no_backfill_neighbors,
        3,
        2,
        1,
        [1],
        1,
        "boundery_recent_core",
        0,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0, 0, 1], dtype=torch.int32),
        1e-9,
    )
    no_backfill.configure_core_sampling(
        1e-9, 1.0, 0.25, 0.5, 1.0, 1.0, 1.0, 0.0, 0.05,
        0, 8, 19
    )
    no_backfill.neighbor_sample_from_nodes(
        torch.tensor([0]), torch.tensor([3]), None, 1e-9
    )
    assert 0 not in no_backfill.get_ret()[0].eid().tolist()

    top_one = module.ParallelSampler(
        no_backfill_neighbors,
        3,
        2,
        1,
        [2],
        1,
        "boundery_recent_core",
        0,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0, 0, 1], dtype=torch.int32),
        1e-9,
    )
    top_one.configure_core_sampling(
        1e-9, 1.0, 0.25, 0.5, 1.0, 1.0, 1.0, 0.0, 0.05,
        1, 8, 19
    )
    top_one.neighbor_sample_from_nodes(
        torch.tensor([0]), torch.tensor([3]), None, 1e-9
    )
    top_one_result = top_one.get_ret()[0]
    newest = top_one_result.eid() == 1
    assert newest.sum().item() == 1
    assert top_one_result.sample_weight()[newest].item() == 1.0
