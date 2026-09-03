import pytest
import torch

from starrygl.module.compact_memory_ops import (
    attach_node_features,
    latest_object_selection,
    normalize_transition,
    shared_object_plan,
)
from starrygl.module.memorys import AsyncMemeoryUpdater, TransformerMemoryUpdater
from starrygl.sample.batch_data import prepare_input
from starrygl.sample.memory.shared_mailbox import SharedMailBox


class _Block:
    def __init__(self, state_to_uid, hierarchical=False):
        self.srcdata = {'__ID': state_to_uid}
        self.edata = {}
        if hierarchical:
            self._memshare_hierarchical_compaction = True

    def num_src_nodes(self):
        return self.srcdata['__ID'].shape[0]


class _Mailbox:
    def __init__(self, device):
        self.shared_nodes_index = torch.tensor([0, 2, 4], device=device)
        self.is_shared_mask = torch.tensor([0, -1, 1, -1, 2], device=device)
        self.deliver_to = 'self'


def _legacy_normalize(values):
    values = values.clone()
    if values.numel() > 0 and not values.max().item() == 0:
        values -= values.min()
        values /= values.max()
        values = 2 * values - 1
    return values


def test_prepare_input_exposes_compact_memory_objects(monkeypatch):
    monkeypatch.setenv('MEMSHARE_COMPACT_MEMORY_OPS', 'on')
    monkeypatch.setenv('MEMSHARE_COMPACT_GRU_UPDATE', 'off')
    state_to_uid = torch.tensor([0, 1, 0, 2])
    block = _Block(state_to_uid, hierarchical=True)
    node_features = torch.randn(3, 7)
    memory = torch.randn(3, 5)
    memory_ts = torch.randn(3)
    mailbox = torch.randn(3, 1, 9)
    mailbox_ts = torch.randn(3, 1)
    object_ids = torch.tensor([11, 12, 13])

    prepare_input(
        node_features, None,
        (memory, memory_ts, mailbox, mailbox_ts),
        [[block]], object_ids,
    )

    assert 'h' not in block.srcdata
    assert block._memshare_node_feat_object is node_features
    assert block._memshare_memory_object is memory
    torch.testing.assert_close(
        block._memshare_mailbox_object, mailbox.flatten(1))
    torch.testing.assert_close(
        block._memshare_mailbox_ts_object, mailbox_ts.flatten(1))
    assert block._memshare_dist_nid_object is object_ids
    torch.testing.assert_close(block.srcdata['mem'], memory[state_to_uid])
    torch.testing.assert_close(
        block.srcdata['mem_input'], mailbox[state_to_uid].flatten(1))


def test_latest_object_selection_matches_legacy_memory_and_mail():
    state_to_uid = torch.tensor([0, 1, 0, 2, 1])
    object_ids = torch.tensor([10, 11, 12])
    src = torch.tensor([0, 1, 3])
    dst = torch.tensor([2, 4, 1])
    states = torch.cat((src, dst))
    ts = torch.tensor([2.0, 5.0, 4.0])
    candidate_ts = torch.cat((ts, ts))

    ids, selected = latest_object_selection(
        state_to_uid, states, candidate_ts, object_ids)

    mailbox = SharedMailBox.__new__(SharedMailBox)
    mailbox.device = torch.device('cpu')
    mailbox.mailbox = torch.empty(0, dtype=torch.float32)
    mailbox.mailbox_ts = torch.empty(0, dtype=torch.float32)
    mailbox.deliver_to = 'self'
    state_ids = object_ids[state_to_uid]
    state_memory = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    edge_features = torch.arange(6, dtype=torch.float32).reshape(3, 2)

    expected_ids, expected_mail, expected_mail_ts = mailbox.get_update_mail(
        state_ids, src, dst, ts, edge_features, state_memory)
    actual_ids, actual_mail, actual_mail_ts = mailbox.get_update_mail(
        state_ids, src, dst, ts, edge_features, state_memory,
        selection=(ids, selected))

    expected_order = expected_ids.argsort()
    actual_order = actual_ids.argsort()
    torch.testing.assert_close(
        actual_ids[actual_order], expected_ids[expected_order])
    torch.testing.assert_close(
        actual_mail[actual_order], expected_mail[expected_order])
    torch.testing.assert_close(
        actual_mail_ts[actual_order], expected_mail_ts[expected_order])

    expected_ids, expected_memory, expected_ts = mailbox.get_update_memory(
        state_ids[states], state_memory[states], candidate_ts, None)
    selected_states = states[selected]
    actual_memory = state_memory[selected_states]
    actual_ts = candidate_ts[selected]
    expected_order = expected_ids.argsort()
    actual_order = ids.argsort()
    torch.testing.assert_close(ids[actual_order], expected_ids[expected_order])
    torch.testing.assert_close(
        actual_memory[actual_order], expected_memory[expected_order])
    torch.testing.assert_close(
        actual_ts[actual_order], expected_ts[expected_order])


@pytest.mark.parametrize('values', (
    torch.zeros(2, 3),
    torch.tensor([[-2.0, 0.0], [-1.0, -0.5]]),
    torch.full((2, 3), 3.0),
    torch.full((2, 3), -3.0),
))
def test_normalize_transition_preserves_legacy_edges(values):
    torch.testing.assert_close(
        normalize_transition(values), _legacy_normalize(values),
        rtol=0, atol=0, equal_nan=True)


def test_shared_prediction_and_feature_attach_match_state_expansion():
    torch.manual_seed(3)
    state_to_uid = torch.tensor([2, 0, 2, 3, 1, 0, 3])
    shared_object_mask = torch.tensor([True, False, True, True])
    shared_values = torch.randn(shared_object_mask.sum(), 5)
    used_shared, state_mask, rows = shared_object_plan(
        state_to_uid, shared_object_mask)
    compact_values = shared_values.index_select(0, rows)

    object_values = torch.zeros(4, 5)
    object_values[used_shared] = shared_values
    torch.testing.assert_close(
        compact_values, object_values[state_to_uid][state_mask])

    node_features = torch.randn(4, 7)
    updated_expected = torch.randn(7, 5, requires_grad=True)
    updated_actual = updated_expected.detach().clone().requires_grad_()
    expected_map = torch.nn.Linear(7, 5)
    actual_map = torch.nn.Linear(7, 5)
    actual_map.load_state_dict(expected_map.state_dict())

    expected = updated_expected + expected_map(node_features[state_to_uid])
    actual = attach_node_features(
        updated_actual, node_features, state_to_uid, actual_map)
    torch.testing.assert_close(actual, expected)

    grad = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(
        expected,
        (updated_expected, expected_map.weight, expected_map.bias),
        grad,
    )
    actual_grads = torch.autograd.grad(
        actual,
        (updated_actual, actual_map.weight, actual_map.bias),
        grad,
    )
    for expected_grad, actual_grad in zip(expected_grads, actual_grads):
        torch.testing.assert_close(
            actual_grad, expected_grad, rtol=1e-5, atol=1e-5)


def test_shared_object_plan_excludes_unused_objects():
    state_to_uid = torch.tensor([0, 1, 0])
    shared_object_mask = torch.tensor([True, True, True])

    used_shared, state_mask, rows = shared_object_plan(
        state_to_uid, shared_object_mask)

    torch.testing.assert_close(
        used_shared, torch.tensor([True, True, False]))
    torch.testing.assert_close(state_mask, torch.ones(3, dtype=torch.bool))
    torch.testing.assert_close(rows, torch.tensor([0, 1, 0]))


def test_memory_and_mail_keep_separate_timestamp_selectors():
    state_to_uid = torch.tensor([0, 0])
    states = torch.tensor([0, 1])
    object_ids = torch.tensor([10])
    memory_ts = torch.tensor([2048.0, 2049.0], dtype=torch.float32)
    mail_ts = memory_ts.to(torch.float16)

    _, memory_selected = latest_object_selection(
        state_to_uid, states, memory_ts, object_ids)
    _, mail_selected = latest_object_selection(
        state_to_uid, states, mail_ts, object_ids)

    assert memory_selected.item() == 1
    assert mail_selected.item() == 0


def test_prepare_input_uses_state_fallback_without_hierarchical_block(
        monkeypatch):
    monkeypatch.setenv('MEMSHARE_COMPACT_MEMORY_OPS', 'on')
    monkeypatch.setenv('MEMSHARE_COMPACT_GRU_UPDATE', 'off')
    state_to_uid = torch.tensor([0, 1, 0])
    block = _Block(state_to_uid)
    node_features = torch.randn(2, 4)
    memory = torch.randn(2, 3)
    memory_ts = torch.randn(2)
    mailbox = torch.randn(2, 1, 5)
    mailbox_ts = torch.randn(2, 1)

    prepare_input(
        node_features, None,
        (memory, memory_ts, mailbox, mailbox_ts),
        [[block]], torch.tensor([10, 11]),
    )

    torch.testing.assert_close(block.srcdata['h'], node_features[state_to_uid])
    assert not hasattr(block, '_memshare_node_feat_object')
    assert block._memshare_memory_object is memory
    torch.testing.assert_close(
        block._memshare_mailbox_object, mailbox.flatten(1))


@pytest.mark.parametrize('nonlinearity', ('tanh', 'relu'))
def test_compact_rnn_matches_state_path_on_legacy_block(nonlinearity):
    torch.manual_seed(5)
    state_to_uid = torch.tensor([0, 1, 0, 2])
    object_ids = torch.tensor([10, 11, 12])
    states, hidden, mail_dim, time_dim = 4, 3, 5, 2
    mailbox = _Mailbox(torch.device('cpu'))

    def make_updater(compact):
        def rnn_cell(input_size, hidden_size):
            return torch.nn.RNNCell(
                input_size, hidden_size, nonlinearity=nonlinearity)

        updater = AsyncMemeoryUpdater(
            {
                'memory_update': 'rnn',
                'combine_node_feature': True,
                'compact_gru_update': False,
                'compact_memory_ops': compact,
            },
            mail_dim, hidden, time_dim, hidden,
            updater=rnn_cell, mode='all_update',
            mailbox=mailbox, ada_param=None,
        )
        updater.eval()
        return updater

    legacy = make_updater(False)
    compact = make_updater(True)
    compact.load_state_dict(legacy.state_dict())
    memory_object = torch.randn(3, hidden)
    memory = memory_object[state_to_uid]
    memory_ts = torch.randn(states)
    state_ts = memory_ts + torch.rand(states)
    mailbox_object = torch.randn(3, mail_dim)
    mail = mailbox_object[state_to_uid]
    node_features = torch.randn(3, hidden)

    def make_block(is_compact):
        block = _Block(state_to_uid)
        block.srcdata.update({
            'ID': object_ids[state_to_uid],
            'mem': memory,
            'mem_ts': memory_ts,
            'mem_input': mail,
            'ts': state_ts,
            'h': node_features[state_to_uid],
        })
        if is_compact:
            block._memshare_memory_object = memory_object
            block._memshare_mailbox_object = mailbox_object
        return block

    legacy_block = make_block(False)
    compact_block = make_block(True)
    legacy([legacy_block])
    compact([compact_block])

    torch.testing.assert_close(
        compact_block.srcdata['h'], legacy_block.srcdata['h'])
    legacy_block.srcdata['h'].sum().backward()
    compact_block.srcdata['h'].sum().backward()
    for (legacy_name, legacy_param), (compact_name, compact_param) in zip(
            legacy.named_parameters(), compact.named_parameters()):
        assert legacy_name == compact_name
        torch.testing.assert_close(
            compact_param.grad, legacy_param.grad, rtol=2e-5, atol=2e-6)

    torch.testing.assert_close(
        compact.rnn_updater(make_block(False)),
        legacy.rnn_updater(make_block(False)),
    )


def test_compact_transformer_matches_state_path_and_dropout_rng():
    torch.manual_seed(11)
    state_to_uid = torch.tensor([2, 0, 2, 1, 0])
    object_ids = torch.tensor([10, 11, 12])
    states, objects = state_to_uid.shape[0], object_ids.shape[0]
    hidden, mail_dim, time_dim, mailbox_size = 4, 6, 3, 3
    mailbox = _Mailbox(torch.device('cpu'))
    memory_param = {
        'memory_update': 'transformer',
        'attention_head': 2,
        'mailbox_size': mailbox_size,
        'combine_node_feature': False,
        'compact_gru_update': False,
    }
    train_param = {'dropout': 0.2, 'att_dropout': 0.3}

    def make_updater(compact):
        params = dict(memory_param, compact_memory_ops=compact)
        updater = AsyncMemeoryUpdater(
            params, mail_dim, hidden, time_dim, 0,
            updater=TransformerMemoryUpdater, mode='all_update', mailbox=mailbox,
            train_param=train_param, ada_param=None,
        )
        updater.eval()
        return updater

    legacy = make_updater(False)
    compact = make_updater(True)
    compact.load_state_dict(legacy.state_dict())
    memory_object = torch.randn(objects, hidden)
    mailbox_object = torch.randn(objects, mailbox_size, mail_dim)
    mailbox_ts_object = torch.randn(objects, mailbox_size)
    state_memory = memory_object[state_to_uid]
    state_ts = torch.randn(states)

    def make_block(is_compact):
        block = _Block(state_to_uid)
        block.srcdata.update({
            'ID': object_ids[state_to_uid],
            'mem': state_memory,
            'mem_ts': torch.zeros(states),
            'mem_input': mailbox_object[state_to_uid].flatten(1),
            'mail_ts': mailbox_ts_object[state_to_uid],
            'ts': state_ts,
        })
        if is_compact:
            block._memshare_memory_object = memory_object
            block._memshare_mailbox_object = mailbox_object.flatten(1)
            block._memshare_mailbox_ts_object = mailbox_ts_object
        return block

    legacy_block = make_block(False)
    compact_block = make_block(True)
    legacy([legacy_block])
    compact([compact_block])
    torch.testing.assert_close(
        compact_block.srcdata['h'], legacy_block.srcdata['h'],
        rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        compact.ceil_updater(make_block(False), compact=True),
        legacy.ceil_updater(make_block(False), compact=False),
        rtol=2e-5, atol=2e-6,
    )

    legacy_block.srcdata['h'].sum().backward()
    compact_block.srcdata['h'].sum().backward()
    for (legacy_name, legacy_param), (compact_name, compact_param) in zip(
            legacy.named_parameters(), compact.named_parameters()):
        assert legacy_name == compact_name
        torch.testing.assert_close(
            compact_param.grad, legacy_param.grad, rtol=3e-5, atol=3e-6)

    legacy.ceil_updater.train()
    compact.ceil_updater.train()
    torch.manual_seed(23)
    expected = legacy.ceil_updater(make_block(False), compact=False)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(23)
    actual = compact.ceil_updater(make_block(True), compact=True)
    actual_rng = torch.get_rng_state()
    torch.testing.assert_close(actual_rng, expected_rng)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_compact_smooth_and_feature_attach_match_state_path():
    torch.manual_seed(7)
    device = torch.device('cuda')
    state_to_uid = torch.tensor([2, 0, 2, 3, 1, 0, 3], device=device)
    shared_bit = 1 << 48
    object_ids = torch.tensor(
        [shared_bit | 0, 1, shared_bit | 2, 3, shared_bit | 4],
        device=device)
    states = state_to_uid.numel()
    hidden, mail_dim, time_dim, node_dim = 4, 6, 3, 5
    mailbox = _Mailbox(device)

    def make_updater(compact):
        memory_param = {
            'memory_update': 'gru',
            'combine_node_feature': True,
            'compact_gru_update': False,
            'compact_memory_ops': compact,
        }
        updater = AsyncMemeoryUpdater(
            memory_param, mail_dim, hidden, time_dim, node_dim,
            updater=torch.nn.GRUCell, mode='historical',
            mailbox=mailbox, ada_param=None,
        ).to(device)
        updater.eval()
        return updater

    legacy = make_updater(False)
    compact = make_updater(True)
    compact.load_state_dict(legacy.state_dict())
    filter_count = torch.tensor([[2.0], [3.0], [1.0]], device=device)
    filter_increment = torch.randn(3, hidden, device=device)
    for updater in (legacy, compact):
        updater.filter.count.copy_(filter_count)
        updater.filter.incretment.copy_(filter_increment)

    memory_object = torch.randn(5, hidden, device=device)
    historical_object = torch.randn(5, hidden, device=device)
    historical_object[4] = 1000
    node_features = torch.randn(5, node_dim, device=device)
    mem_input = torch.randn(states, mail_dim, device=device)
    mem_ts = torch.randn(states, device=device)
    state_ts = mem_ts + torch.rand(states, device=device)

    def make_block(is_compact):
        block = _Block(state_to_uid)
        block.srcdata.update({
            'ID': object_ids[state_to_uid],
            'mem': memory_object[state_to_uid],
            'mem_ts': mem_ts,
            'mem_input': mem_input,
            'ts': state_ts,
        })
        if is_compact:
            block._memshare_dist_nid_object = object_ids
            block._memshare_historical_memory_object = historical_object
            block._memshare_node_feat_object = node_features
        else:
            block.srcdata['his_mem'] = historical_object[state_to_uid]
            block.srcdata['h'] = node_features[state_to_uid]
        return block

    legacy_block = make_block(False)
    compact_block = make_block(True)
    legacy([legacy_block])
    compact([compact_block])

    torch.testing.assert_close(
        compact_block.srcdata['h'], legacy_block.srcdata['h'],
        rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        compact.last_updated_memory, legacy.last_updated_memory,
        rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(compact.filter.count, legacy.filter.count)
    torch.testing.assert_close(
        compact.filter.incretment, legacy.filter.incretment,
        rtol=2e-5, atol=2e-5)

    legacy_block.srcdata['h'].sum().backward()
    compact_block.srcdata['h'].sum().backward()
    for (legacy_name, legacy_param), (compact_name, compact_param) in zip(
            legacy.named_parameters(), compact.named_parameters()):
        assert legacy_name == compact_name
        torch.testing.assert_close(
            compact_param.grad, legacy_param.grad, rtol=2e-4, atol=2e-5)
