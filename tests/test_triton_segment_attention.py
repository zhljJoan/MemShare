import copy

import dgl
import pytest
import torch

from starrygl.module import triton_segment_attention
from starrygl.module.layers import TransfomerAttentionLayer
from starrygl.sample import direct_segment_block


def _inputs(device, dtype=torch.float32, degree=(0, 3, 1, 0, 5, 2)):
    dst_ptr = torch.tensor((0, *torch.tensor(degree).cumsum(0).tolist()),
                           device=device, dtype=torch.int64)
    q = torch.randn(len(degree), 2, 7, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(sum(degree), 2, 7, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(sum(degree), 2, 5, device=device, dtype=dtype, requires_grad=True)
    return q, k, v, dst_ptr


@pytest.mark.parametrize("dropout_p", (0.0, 0.35))
def test_triton_forward_backward_matches_reference(dropout_p):
    if not torch.cuda.is_available() or triton_segment_attention.triton is None:
        pytest.skip("CUDA and Triton are required")
    torch.manual_seed(7)
    got_inputs = _inputs("cuda")
    expected_inputs = tuple(x.detach().clone().requires_grad_(x.is_floating_point())
                            for x in got_inputs)
    dropout_mask = torch.rand(got_inputs[1].shape[:2], device="cuda") > dropout_p
    edge_ids = None
    if dropout_p > 0.0:
        edge_ids = torch.roll(
            torch.arange(got_inputs[1].shape[0], device="cuda"), shifts=1)
    kwargs = dict(dropout_p=dropout_p, training=dropout_p > 0.0,
                  dropout_mask=dropout_mask, edge_ids=edge_ids)
    got = triton_segment_attention.attention(*got_inputs, **kwargs)
    expected = triton_segment_attention.reference(*expected_inputs, **kwargs)
    torch.testing.assert_close(got, expected, rtol=2e-5, atol=2e-5)

    grad = torch.randn_like(got)
    got.backward(grad)
    expected.backward(grad)
    for actual_input, expected_input in zip(got_inputs[:3], expected_inputs[:3]):
        torch.testing.assert_close(
            actual_input.grad, expected_input.grad, rtol=2e-4, atol=2e-4)


def test_cpu_fallback():
    torch.manual_seed(8)
    cpu_inputs = _inputs("cpu", dtype=torch.float64)
    cpu_out = triton_segment_attention.attention(*cpu_inputs)
    assert cpu_out.shape == (6, 2, 5)
    cpu_out.sum().backward()
    assert all(x.grad is not None for x in cpu_inputs[:3])


def test_generated_dropout_mask_matches_pytorch_rng():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    torch.manual_seed(11)
    x = torch.rand(127, 2, device="cuda") + 0.1
    rng_state = torch.cuda.get_rng_state()
    expected = torch.nn.functional.dropout(x, 0.2, training=True).ne(0)
    expected_rng_state = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(rng_state)
    k = torch.empty(127, 2, 7, device="cuda")
    actual = triton_segment_attention._prepare_dropout(
        k, 0.2, True, None)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.cuda.get_rng_state(), expected_rng_state)


def test_uniform_segment_forward_backward_matches_reference():
    if not torch.cuda.is_available() or triton_segment_attention.triton is None:
        pytest.skip("CUDA and Triton are required")
    torch.manual_seed(12)
    states, degree, heads = 17, 10, 2
    q = torch.randn(
        states, heads, 7, device="cuda", requires_grad=True)
    k = torch.randn(
        states * degree, heads, 7, device="cuda", requires_grad=True)
    v = torch.randn(
        states * degree, heads, 5, device="cuda", requires_grad=True)
    expected_inputs = tuple(
        x.detach().clone().requires_grad_() for x in (q, k, v))
    dst_ptr = torch.arange(states + 1, device="cuda") * degree
    dropout_mask = torch.rand(states * degree, heads, device="cuda") > 0.2

    actual = triton_segment_attention.uniform_attention(
        q, k, v, degree, dropout_p=0.2, training=True,
        dropout_mask=dropout_mask)
    expected = triton_segment_attention.reference(
        *expected_inputs, dst_ptr, dropout_p=0.2, training=True,
        dropout_mask=dropout_mask)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    for actual_input, expected_input in zip((q, k, v), expected_inputs):
        torch.testing.assert_close(
            actual_input.grad, expected_input.grad, rtol=2e-4, atol=2e-4)



def test_degree_630_forward_backward_matches_reference():
    if not torch.cuda.is_available() or triton_segment_attention.triton is None:
        pytest.skip("CUDA and Triton are required")
    torch.manual_seed(10)
    got_inputs = _inputs("cuda", degree=(630, 3, 0))
    expected_inputs = tuple(x.detach().clone().requires_grad_(x.is_floating_point())
                            for x in got_inputs)
    edge_ids = torch.randperm(got_inputs[1].shape[0], device="cuda")
    dropout_mask = torch.rand(got_inputs[1].shape[:2], device="cuda") > 0.2
    kwargs = dict(dropout_p=0.2, training=True, dropout_mask=dropout_mask,
                  edge_ids=edge_ids)
    got = triton_segment_attention.attention(*got_inputs, **kwargs)
    expected = triton_segment_attention.reference(*expected_inputs, **kwargs)
    torch.testing.assert_close(got, expected, rtol=3e-5, atol=3e-5)

    grad = torch.randn_like(got)
    got.backward(grad)
    expected.backward(grad)
    for actual_input, expected_input in zip(got_inputs[:3], expected_inputs[:3]):
        torch.testing.assert_close(
            actual_input.grad, expected_input.grad, rtol=5e-4, atol=5e-4)


def test_empty_edge_ids_means_identity():
    inputs = _inputs("cuda" if torch.cuda.is_available() else "cpu")
    empty = torch.empty(0, device=inputs[0].device, dtype=torch.int64)
    torch.testing.assert_close(
        triton_segment_attention.attention(
            *inputs, edge_ids=empty, validate=False),
        triton_segment_attention.attention(*inputs),
    )


@pytest.mark.parametrize("force_dgl_fallback", (False, True))
def test_layer_matches_dgl_with_non_segmented_edge_order(
        monkeypatch, force_dgl_fallback):
    if not torch.cuda.is_available() or triton_segment_attention.triton is None:
        pytest.skip("CUDA and Triton are required")
    torch.manual_seed(9)
    device = torch.device("cuda")
    src = torch.tensor([1, 3, 2, 0, 5, 4, 2], device=device)
    dst = torch.tensor([2, 0, 1, 0, 2, 1, 0], device=device)

    def make_block(h, edge, dt, direct=False):
        block = (direct_segment_block.create_block(src, dst, 6, 3)
                 if direct else dgl.create_block(
                     (src, dst), num_src_nodes=6, num_dst_nodes=3,
                     device=device))
        block.srcdata['h'] = h
        block.edata['f'] = edge
        block.edata['dt'] = dt
        return block

    baseline_layer = TransfomerAttentionLayer(6, 4, 3, 2, 0, 0, 8).to(device).eval()
    fused_layer = copy.deepcopy(baseline_layer)
    baseline_inputs = (
        torch.randn(6, 6, device=device, requires_grad=True),
        torch.randn(src.numel(), 4, device=device, requires_grad=True),
        torch.randn(src.numel(), device=device, requires_grad=True),
    )
    fused_inputs = tuple(x.detach().clone().requires_grad_() for x in baseline_inputs)

    monkeypatch.setenv("MEMSHARE_FUSED_SEGMENT_ATTENTION", "off")
    expected = baseline_layer(make_block(*baseline_inputs))
    monkeypatch.setenv("MEMSHARE_FUSED_SEGMENT_ATTENTION", "on")
    if force_dgl_fallback:
        monkeypatch.setattr(triton_segment_attention, "supported",
                            lambda q, k, v: False)
    actual = fused_layer(make_block(*fused_inputs, direct=True))
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)

    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    for actual_input, expected_input in zip(fused_inputs, baseline_inputs):
        torch.testing.assert_close(
            actual_input.grad, expected_input.grad, rtol=5e-4, atol=5e-4)
    for (actual_name, actual_param), (expected_name, expected_param) in zip(
            fused_layer.named_parameters(), baseline_layer.named_parameters()):
        assert actual_name == expected_name
        torch.testing.assert_close(
            actual_param.grad, expected_param.grad, rtol=5e-4, atol=5e-4)
