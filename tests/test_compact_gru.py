import pytest
import torch

from starrygl.module.compact_gru import compact_gru_cell
from starrygl.sample.batch_data import prepare_input


class _Block:
    def __init__(self, state_to_uid):
        self.srcdata = {'__ID': state_to_uid}
        self.edata = {}


def test_prepare_input_keeps_compact_gru_objects(monkeypatch):
    monkeypatch.setenv('MEMSHARE_COMPACT_GRU_UPDATE', 'on')
    state_to_uid = torch.tensor([0, 1, 0, 2])
    block = _Block(state_to_uid)
    memory = torch.randn(3, 5)
    memory_ts = torch.randn(3)
    mailbox = torch.randn(3, 1, 7)
    mailbox_ts = torch.randn(3, 1)

    prepare_input(
        None,
        None,
        (memory, memory_ts, mailbox, mailbox_ts),
        [[block]],
        torch.arange(3),
    )

    assert 'mem_input' not in block.srcdata
    assert block._memshare_memory_object is memory
    torch.testing.assert_close(
        block._memshare_mailbox_object,
        mailbox.reshape(3, -1),
    )
    torch.testing.assert_close(block.srcdata['mem'], memory[state_to_uid])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA fused GRU is unavailable')
@pytest.mark.parametrize('dtype', (torch.float32, torch.float64))
def test_compact_gru_matches_gru_cell_forward_and_backward(dtype):
    torch.manual_seed(1)
    device = torch.device('cuda')
    states, objects, mail_dim, time_dim, hidden_dim = 37, 9, 12, 7, 5
    state_to_uid = torch.randint(objects, (states,), device=device)
    mailbox = torch.randn(objects, mail_dim, device=device)
    memory = torch.randn(objects, hidden_dim, device=device)
    state_memory = memory.index_select(0, state_to_uid)
    time_feat = torch.randn(
        states, time_dim, device=device, dtype=dtype, requires_grad=True,
    )
    mailbox = mailbox.to(dtype)
    memory = memory.to(dtype)
    state_memory = state_memory.to(dtype)
    cell = torch.nn.GRUCell(mail_dim + time_dim, hidden_dim).to(device, dtype)

    tolerance = {'rtol': 2e-4, 'atol': 2e-5} if dtype == torch.float32 else {}

    expected = cell(
        torch.cat((mailbox.index_select(0, state_to_uid), time_feat), dim=1),
        state_memory,
    )
    actual = compact_gru_cell(
        cell, mailbox, memory, time_feat, state_to_uid, state_memory,
    )
    torch.testing.assert_close(actual, expected, **tolerance)

    inputs = (
        time_feat,
        cell.weight_ih,
        cell.weight_hh,
        cell.bias_ih,
        cell.bias_hh,
    )
    grad = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(expected, inputs, grad, retain_graph=True)
    actual_grads = torch.autograd.grad(actual, inputs, grad)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, **tolerance)

    if dtype == torch.float32:
        state_memory = state_memory.detach().requires_grad_()
        expected = cell(
            torch.cat((mailbox.index_select(0, state_to_uid), time_feat), dim=1),
            state_memory,
        )
        actual = compact_gru_cell(
            cell, mailbox, memory, time_feat, state_to_uid, state_memory,
        )
        torch.testing.assert_close(
            torch.autograd.grad(actual.sum(), state_memory)[0],
            torch.autograd.grad(expected.sum(), state_memory)[0],
            **tolerance,
        )
