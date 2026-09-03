import pytest
import torch


def test_fused_edge_temporal_linear_forward_matches_concat():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    import starrygl

    if not hasattr(starrygl, "ops"):
        pytest.skip("starrygl.ops is not built")

    torch.manual_seed(0)
    device = torch.device("cuda")
    state = torch.randn(5, 7, device=device)
    edge = torch.randn(4, 3, device=device)
    time = torch.randn(9, 2, device=device)
    edge_src = torch.tensor([0, 1, 2, 3, 4, 0, 2, 1, 3], device=device)
    edge_to_eid = torch.tensor([0, 1, 2, 3, 0, 2, 1, 3, 1], device=device)
    linear = torch.nn.Linear(12, 6, device=device)

    got = starrygl.ops.fused_edge_temporal_linear_forward(
        state.contiguous(),
        edge.contiguous(),
        time.contiguous(),
        linear.weight.contiguous(),
        linear.bias.contiguous(),
        edge_src.contiguous(),
        edge_to_eid.contiguous(),
    )
    expected = linear(torch.cat([state[edge_src], edge[edge_to_eid], time], dim=1))

    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-4)
