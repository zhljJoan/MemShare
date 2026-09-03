import pytest
import torch

from starrygl.module import triton_edge_projection


def test_triton_edge_projection_forward_backward_matches_baseline(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    monkeypatch.setenv("MEMSHARE_TRITON_EDGE_PROJECTION", "on")
    torch.manual_seed(0)
    device = torch.device("cuda")
    state = torch.randn(11, 7, device=device, requires_grad=True)
    edge = torch.randn(23, 5, device=device, requires_grad=True)
    time = torch.randn(23, 3, device=device, requires_grad=True)
    edge_src = torch.randint(state.shape[0], (edge.shape[0],), device=device)
    linear = torch.nn.Linear(15, 9, device=device)

    got = triton_edge_projection.linear(
        state, edge, time, linear.weight, linear.bias, edge_src)
    expected = linear(torch.cat([state[edge_src], edge, time], dim=1))
    torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-2)

    grad = torch.randn_like(got)
    got.backward(grad, retain_graph=True)
    got_grads = (
        state.grad.detach().clone(),
        edge.grad.detach().clone(),
        time.grad.detach().clone(),
        linear.weight.grad.detach().clone(),
        linear.bias.grad.detach().clone(),
    )
    state.grad = edge.grad = time.grad = None
    linear.weight.grad = linear.bias.grad = None
    expected.backward(grad)
    expected_grads = (
        state.grad,
        edge.grad,
        time.grad,
        linear.weight.grad,
        linear.bias.grad,
    )
    for got_grad, expected_grad in zip(got_grads, expected_grads):
        torch.testing.assert_close(got_grad, expected_grad, rtol=1e-4, atol=1e-4)


def test_triton_object_edge_projection_forward_backward_matches_baseline(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    monkeypatch.setenv("MEMSHARE_TRITON_EDGE_PROJECTION", "on")
    torch.manual_seed(1)
    device = torch.device("cuda")
    state = torch.randn(13, 7, device=device, requires_grad=True)
    edge = torch.randn(17, 5, device=device, requires_grad=True)
    time = torch.randn(29, 3, device=device, requires_grad=True)
    edge_src = torch.randint(state.shape[0], (time.shape[0],), device=device)
    edge_to_eid = torch.randint(edge.shape[0], (time.shape[0],), device=device)
    linear = torch.nn.Linear(15, 9, device=device)

    got = triton_edge_projection.object_linear(
        state, edge, time, linear.weight, linear.bias, edge_src, edge_to_eid)
    expected = linear(torch.cat([state[edge_src], edge[edge_to_eid], time], dim=1))
    torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-2)

    grad = torch.randn_like(got)
    got.backward(grad, retain_graph=True)
    got_grads = (
        state.grad.detach().clone(),
        edge.grad.detach().clone(),
        time.grad.detach().clone(),
        linear.weight.grad.detach().clone(),
        linear.bias.grad.detach().clone(),
    )
    state.grad = edge.grad = time.grad = None
    linear.weight.grad = linear.bias.grad = None
    expected.backward(grad)
    expected_grads = (
        state.grad,
        edge.grad,
        time.grad,
        linear.weight.grad,
        linear.bias.grad,
    )
    for got_grad, expected_grad in zip(got_grads, expected_grads):
        torch.testing.assert_close(got_grad, expected_grad, rtol=1e-4, atol=1e-4)
