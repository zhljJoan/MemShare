from types import SimpleNamespace

import torch

from starrygl.distributed.utils import DistIndex
from starrygl.lib.libstarrygl_sampler import ParallelSampler, get_neighbors
from starrygl.sample.batch_data import to_block
from starrygl.sample.sample_core.neighbor_sampler import NeighborSampler


def test_sampler_override_disables_hierarchical_compaction(monkeypatch):
    monkeypatch.setenv('MEMSHARE_HIERARCHICAL_COMPACTION', 'on')

    class LegacySampler:
        def neighbor_sample_from_nodes(self, *_args):
            self.called = True

        def get_ret(self):
            return 'legacy'

    sampler = NeighborSampler.__new__(NeighborSampler)
    sampler.hierarchical_compaction = False
    sampler.policy = 'recent'
    sampler.ada_param = SimpleNamespace(beta=0.1)
    sampler.p_sampler = LegacySampler()

    output, metadata = sampler.sample_from_nodes(
        torch.tensor([0]), torch.tensor([1]))

    assert output == 'legacy'
    assert metadata == {}
    assert sampler.p_sampler.called


def test_hierarchical_compaction_matches_legacy_block(monkeypatch):
    monkeypatch.setenv('MEMSHARE_FUSED_SEGMENT_ATTENTION', 'off')
    row = torch.tensor([1, 2, 3, 1, 4, 2, 4])
    col = torch.tensor([0, 0, 0, 2, 2, 3, 3])
    ts = torch.tensor([1, 2, 3, 2, 4, 3, 5])
    eid = torch.arange(row.numel())
    tnb = get_neighbors(
        'test_hierarchical_compaction', row, col, 5, 0,
        eid, None, None, ts)
    edge_part = torch.tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.int32)
    node_part = torch.tensor([0, 1, 0, 1, 0], dtype=torch.int32)
    nid_mapper = torch.tensor([
        (int(node_part[i]) << 50) | i for i in range(5)
    ])
    eid_mapper = torch.tensor([
        (int(edge_part[i]) << 50) | i for i in range(7)
    ])
    seed = torch.tensor([0, 2, 0])
    seed_ts = torch.tensor([4, 5, 4])

    def make_sampler():
        return ParallelSampler(
            tnb, 5, 7, 2, [3], 1, 'recent', 0,
            edge_part, node_part, 1.0)

    legacy_sampler = make_sampler()
    legacy_sampler.neighbor_sample_from_nodes(seed, seed_ts, None, 1.0)
    legacy_ret = legacy_sampler.get_ret()
    compact = make_sampler().sample_and_compact_hierarchical(
        seed, seed_ts, 1.0, nid_mapper, eid_mapper, 2, False)

    metadata = {
        'seed': seed.clone(),
        'seed_ts': seed_ts.clone(),
        'src_pos_index': torch.tensor([0]),
        'dst_pos_index': torch.tensor([1]),
        'dst_neg_index': torch.tensor([2]),
        'train_ratio_pos': 0.5,
    }
    graph = SimpleNamespace(
        nids_mapper=nid_mapper,
        eids_mapper=eid_mapper,
        sample_graph={'ts': ts},
    )
    data = SimpleNamespace()
    legacy, legacy_nids, legacy_eids = to_block(
        graph, data, (legacy_ret, dict(metadata)), torch.device('cpu'))
    raw, raw_nids, raw_eids = to_block(
        graph, data, (legacy_ret, dict(metadata)), torch.device('cpu'),
        unique=False)
    state_raw, state_raw_nids, state_raw_eids = to_block(
        graph, data, (legacy_ret, dict(metadata)), torch.device('cpu'),
        unique=True, state_unique=False)
    actual, actual_nids, actual_eids = to_block(
        graph, data,
        (None, {'_hierarchical_compaction': compact, **metadata}),
        torch.device('cpu'))
    legacy_block = legacy[1][0][0]
    raw_block = raw[1][0][0]
    state_raw_block = state_raw[1][0][0]
    actual_block = actual[1][0][0]

    parts = DistIndex(actual_nids).part
    assert torch.all(parts[:-1] <= parts[1:])
    assert actual_block._memshare_hierarchical_compaction
    assert raw_block.num_src_nodes() > legacy_block.num_src_nodes()
    assert raw_nids.numel() == raw_block.num_src_nodes()
    assert raw_eids.numel() == raw_block.num_edges()
    assert state_raw_block.num_src_nodes() == raw_block.num_src_nodes()
    assert state_raw_nids.numel() == legacy_nids.numel()
    assert state_raw_eids.numel() == legacy_eids.numel()
    assert torch.all(DistIndex(raw_nids).part[:-1]
                     <= DistIndex(raw_nids).part[1:])
    expected_raw_nids = nid_mapper[torch.cat([
        seed, *[ret.sample_nodes() for ret in legacy_ret]])]
    expected_raw_eids = eid_mapper[torch.cat([
        ret.eid() for ret in legacy_ret])]
    torch.testing.assert_close(
        raw_nids[raw_block.srcdata['__ID']], expected_raw_nids)
    torch.testing.assert_close(
        raw_eids[raw_block.edata['__ID']], expected_raw_eids)
    torch.testing.assert_close(
        state_raw_nids[state_raw_block.srcdata['__ID']], expected_raw_nids)
    torch.testing.assert_close(
        state_raw_eids[state_raw_block.edata['__ID']], expected_raw_eids)
    for key in ('src_pos_index', 'dst_pos_index', 'dst_neg_index'):
        torch.testing.assert_close(raw[2][key], metadata[key])

    torch.testing.assert_close(
        legacy_nids[legacy_block.srcdata['__ID']],
        actual_nids[actual_block.srcdata['__ID']])
    torch.testing.assert_close(
        legacy_eids[legacy_block.edata['__ID']],
        actual_eids[actual_block.edata['__ID']])
    torch.testing.assert_close(
        legacy_block.srcdata['ts'], actual_block.srcdata['ts'])
    torch.testing.assert_close(
        legacy_block.edata['dt'], actual_block.edata['dt'])
    for expected, observed in zip(
            legacy_block.edges(order='eid'), actual_block.edges(order='eid')):
        torch.testing.assert_close(expected, observed)
    for key in ('src_pos_index', 'dst_pos_index', 'dst_neg_index'):
        torch.testing.assert_close(legacy[2][key], actual[2][key])
    assert actual[2]['train_ratio_pos'] == 0.5
