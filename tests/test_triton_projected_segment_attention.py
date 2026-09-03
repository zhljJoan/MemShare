import pytest
import torch

from starrygl.module import triton_projected_segment_attention as projected
from starrygl.module import triton_segment_attention as base


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_projected_segment_attention_matches_materialized():
    device = torch.device("cuda")
    torch.manual_seed(7)
    destinations, heads, head_dim = 3, 2, 4
    occurrences, states, edge_objects = 7, 5, 4
    state = torch.randn(states, 2 * heads * head_dim, device=device, requires_grad=True)
    edge = torch.randn(edge_objects, 2 * heads * head_dim, device=device, requires_grad=True)
    time = torch.randn(occurrences, 2 * heads * head_dim, device=device, requires_grad=True)
    q = torch.randn(destinations, heads, head_dim, device=device, requires_grad=True)
    edge_src = torch.tensor([0, 2, 0, 1, 2, 1, 0], device=device)
    edge_obj = torch.tensor([2, 0, 3, 1, 2, 0, 1], device=device)
    edge_ids = torch.tensor([4, 0, 6, 1, 5, 3, 2], device=device)
    dst_ptr = torch.tensor([0, 3, 5, 7], device=device, dtype=torch.int64)
    mask = torch.tensor(
        [[True, False], [True, True], [False, True], [True, True],
         [True, False], [True, True], [False, True]], device=device,
    )

    projected_out = projected.attention(
        q, state, edge, time, edge_src, dst_ptr,
        dropout_p=0.2, training=True, dropout_mask=mask, edge_obj=edge_obj,
        edge_ids=edge_ids, validate=True,
    )
    state_occ = state.index_select(0, edge_src)
    edge_occ = edge.index_select(0, edge_obj)
    components = state_occ + edge_occ + time
    total = heads * head_dim
    k = components[:, :total].view(occurrences, heads, head_dim)
    v = components[:, total:].view_as(k)
    reference_out = base.attention(
        q, k, v, dst_ptr, dropout_p=0.2, training=True,
        dropout_mask=mask, edge_ids=edge_ids, validate=True,
    )
    torch.testing.assert_close(projected_out, reference_out, rtol=2e-5, atol=2e-5)

    grad_out = torch.randn_like(projected_out)
    projected_grads = torch.autograd.grad(
        projected_out, (q, state, edge, time), grad_out, retain_graph=True,
    )
    reference_grads = torch.autograd.grad(
        reference_out, (q, state, edge, time), grad_out,
    )
    for got, expected in zip(projected_grads, reference_grads):
        torch.testing.assert_close(got, expected, rtol=2e-5, atol=2e-5)
