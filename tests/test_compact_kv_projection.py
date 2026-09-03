import torch

from starrygl.module.layers import TransfomerAttentionLayer


def test_compact_kv_matches_occurrence_projection():
    torch.manual_seed(0)
    layer = TransfomerAttentionLayer(7, 5, 3, 1, 0, 0, 9).double()
    state = torch.randn(11, 7, dtype=torch.double, requires_grad=True)
    edge = torch.randn(13, 5, dtype=torch.double, requires_grad=True)
    time = torch.randn(29, 3, dtype=torch.double, requires_grad=True)
    edge_src = torch.randint(state.shape[0], (time.shape[0],))
    edge_to_eid = torch.randint(edge.shape[0], (time.shape[0],))

    got = torch.cat(layer._compact_kv(
        state, edge, time, edge_src, edge_to_eid), dim=1)
    expected = torch.cat((
        layer.w_k(torch.cat((state[edge_src], edge[edge_to_eid], time), dim=1)),
        layer.w_v(torch.cat((state[edge_src], edge[edge_to_eid], time), dim=1)),
    ), dim=1)
    torch.testing.assert_close(got, expected)

    grad = torch.randn_like(got)
    got_grads = torch.autograd.grad(
        got, (state, edge, time, layer.w_k.weight, layer.w_k.bias,
              layer.w_v.weight, layer.w_v.bias), grad, retain_graph=True)
    expected_grads = torch.autograd.grad(
        expected, (state, edge, time, layer.w_k.weight, layer.w_k.bias,
                   layer.w_v.weight, layer.w_v.bias), grad)
    for got_grad, expected_grad in zip(got_grads, expected_grads):
        torch.testing.assert_close(got_grad, expected_grad)
