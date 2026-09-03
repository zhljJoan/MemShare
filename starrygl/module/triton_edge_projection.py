import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


def enabled():
    return os.environ.get("MEMSHARE_TRITON_EDGE_PROJECTION", "off").lower() in {
        "1",
        "true",
        "on",
        "yes",
    }


if triton is not None:

    @triton.jit
    def _forward_kernel(
        state,
        edge,
        time,
        weight,
        bias,
        edge_src,
        out,
        ROWS,
        STATE_DIM: tl.constexpr,
        EDGE_DIM: tl.constexpr,
        TIME_DIM: tl.constexpr,
        OUT_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        src = tl.load(edge_src + offs_m, mask=offs_m < ROWS, other=0)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        total_dim: tl.constexpr = STATE_DIM + EDGE_DIM + TIME_DIM

        for k0 in range(0, total_dim, BLOCK_K):
            k = k0 + offs_k
            state_mask = k < STATE_DIM
            edge_mask = (k >= STATE_DIM) & (k < STATE_DIM + EDGE_DIM)
            time_mask = k >= STATE_DIM + EDGE_DIM

            a_state = tl.load(
                state + src[:, None] * STATE_DIM + k[None, :],
                mask=(offs_m[:, None] < ROWS) & state_mask[None, :],
                other=0.0,
            )
            a_edge = tl.load(
                edge + offs_m[:, None] * EDGE_DIM + (k - STATE_DIM)[None, :],
                mask=(offs_m[:, None] < ROWS) & edge_mask[None, :],
                other=0.0,
            )
            a_time = tl.load(
                time + offs_m[:, None] * TIME_DIM + (k - STATE_DIM - EDGE_DIM)[None, :],
                mask=(offs_m[:, None] < ROWS) & time_mask[None, :],
                other=0.0,
            )
            b = tl.load(
                weight + offs_n[None, :] * total_dim + k[:, None],
                mask=(offs_n[None, :] < OUT_DIM) & (k[:, None] < total_dim),
                other=0.0,
            )
            acc += tl.dot(a_state + a_edge + a_time, b, input_precision="tf32")

        acc += tl.load(bias + offs_n, mask=offs_n < OUT_DIM, other=0.0)[None, :]
        tl.store(
            out + offs_m[:, None] * OUT_DIM + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < ROWS) & (offs_n[None, :] < OUT_DIM),
        )

    @triton.jit
    def _object_forward_kernel(
        state,
        edge,
        time,
        weight,
        bias,
        edge_src,
        edge_to_eid,
        out,
        ROWS,
        STATE_DIM: tl.constexpr,
        EDGE_DIM: tl.constexpr,
        TIME_DIM: tl.constexpr,
        OUT_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        src = tl.load(edge_src + offs_m, mask=offs_m < ROWS, other=0)
        eid = tl.load(edge_to_eid + offs_m, mask=offs_m < ROWS, other=0)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        total_dim: tl.constexpr = STATE_DIM + EDGE_DIM + TIME_DIM

        for k0 in range(0, total_dim, BLOCK_K):
            k = k0 + offs_k
            state_mask = k < STATE_DIM
            edge_mask = (k >= STATE_DIM) & (k < STATE_DIM + EDGE_DIM)
            time_mask = k >= STATE_DIM + EDGE_DIM

            a_state = tl.load(
                state + src[:, None] * STATE_DIM + k[None, :],
                mask=(offs_m[:, None] < ROWS) & state_mask[None, :],
                other=0.0,
            )
            a_edge = tl.load(
                edge + eid[:, None] * EDGE_DIM + (k - STATE_DIM)[None, :],
                mask=(offs_m[:, None] < ROWS) & edge_mask[None, :],
                other=0.0,
            )
            a_time = tl.load(
                time + offs_m[:, None] * TIME_DIM + (k - STATE_DIM - EDGE_DIM)[None, :],
                mask=(offs_m[:, None] < ROWS) & time_mask[None, :],
                other=0.0,
            )
            b = tl.load(
                weight + offs_n[None, :] * total_dim + k[:, None],
                mask=(offs_n[None, :] < OUT_DIM) & (k[:, None] < total_dim),
                other=0.0,
            )
            acc += tl.dot(a_state + a_edge + a_time, b, input_precision="tf32")

        acc += tl.load(bias + offs_n, mask=offs_n < OUT_DIM, other=0.0)[None, :]
        tl.store(
            out + offs_m[:, None] * OUT_DIM + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < ROWS) & (offs_n[None, :] < OUT_DIM),
        )


def _baseline(state, edge, time, weight, bias, edge_src):
    x = torch.cat([state[edge_src], edge, time], dim=1)
    return torch.nn.functional.linear(x, weight, bias)


def _object_baseline(state, edge, time, weight, bias, edge_src, edge_to_eid):
    x = torch.cat([state[edge_src], edge[edge_to_eid], time], dim=1)
    return torch.nn.functional.linear(x, weight, bias)


class _TritonEdgeProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, edge, time, weight, bias, edge_src):
        ctx.save_for_backward(state, edge, time, weight, edge_src)
        if triton is None or not state.is_cuda:
            return _baseline(state, edge, time, weight, bias, edge_src)
        out = torch.empty((time.shape[0], weight.shape[0]), device=time.device, dtype=time.dtype)
        block_m, block_n, block_k = 32, 64, 32
        grid = (triton.cdiv(time.shape[0], block_m), triton.cdiv(weight.shape[0], block_n))
        _forward_kernel[grid](
            state.contiguous(),
            edge.contiguous(),
            time.contiguous(),
            weight.contiguous(),
            bias.contiguous(),
            edge_src.contiguous(),
            out,
            time.shape[0],
            state.shape[1],
            edge.shape[1],
            time.shape[1],
            weight.shape[0],
            block_m,
            block_n,
            block_k,
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        state, edge, time, weight, edge_src = ctx.saved_tensors
        x = torch.cat([state[edge_src], edge, time], dim=1)
        grad_weight = grad_out.t().matmul(x)
        grad_bias = grad_out.sum(0)
        grad_x = grad_out.matmul(weight)
        state_dim = state.shape[1]
        edge_dim = edge.shape[1]
        grad_state_rows = grad_x[:, :state_dim]
        grad_edge = grad_x[:, state_dim : state_dim + edge_dim]
        grad_time = grad_x[:, state_dim + edge_dim :]
        grad_state = torch.zeros_like(state)
        grad_state.index_add_(0, edge_src, grad_state_rows)
        return grad_state, grad_edge, grad_time, grad_weight, grad_bias, None


def linear(state, edge, time, weight, bias, edge_src):
    return _TritonEdgeProjection.apply(state, edge, time, weight, bias, edge_src)


class _TritonObjectEdgeProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, edge, time, weight, bias, edge_src, edge_to_eid):
        ctx.save_for_backward(state, edge, time, weight, edge_src, edge_to_eid)
        if triton is None or not state.is_cuda:
            return _object_baseline(state, edge, time, weight, bias, edge_src, edge_to_eid)
        out = torch.empty((time.shape[0], weight.shape[0]), device=time.device, dtype=time.dtype)
        block_m, block_n, block_k = 32, 64, 32
        grid = (triton.cdiv(time.shape[0], block_m), triton.cdiv(weight.shape[0], block_n))
        _object_forward_kernel[grid](
            state.contiguous(),
            edge.contiguous(),
            time.contiguous(),
            weight.contiguous(),
            bias.contiguous(),
            edge_src.contiguous(),
            edge_to_eid.contiguous(),
            out,
            time.shape[0],
            state.shape[1],
            edge.shape[1],
            time.shape[1],
            weight.shape[0],
            block_m,
            block_n,
            block_k,
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        state, edge, time, weight, edge_src, edge_to_eid = ctx.saved_tensors
        x = torch.cat([state[edge_src], edge[edge_to_eid], time], dim=1)
        grad_weight = grad_out.t().matmul(x)
        grad_bias = grad_out.sum(0)
        grad_x = grad_out.matmul(weight)
        state_dim = state.shape[1]
        edge_dim = edge.shape[1]
        grad_state_rows = grad_x[:, :state_dim]
        grad_edge_rows = grad_x[:, state_dim : state_dim + edge_dim]
        grad_time = grad_x[:, state_dim + edge_dim :]
        grad_state = torch.zeros_like(state)
        grad_edge = torch.zeros_like(edge)
        grad_state.index_add_(0, edge_src, grad_state_rows)
        grad_edge.index_add_(0, edge_to_eid, grad_edge_rows)
        return grad_state, grad_edge, grad_time, grad_weight, grad_bias, None, None


def object_linear(state, edge, time, weight, bias, edge_src, edge_to_eid):
    return _TritonObjectEdgeProjection.apply(
        state, edge, time, weight, bias, edge_src, edge_to_eid)
