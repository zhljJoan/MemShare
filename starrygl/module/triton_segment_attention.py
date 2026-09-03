import os

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


EDGE_TILE = 32
MAX_HEAD_DIM = 64


def enabled():
    return os.environ.get("MEMSHARE_FUSED_SEGMENT_ATTENTION", "off").lower() in {
        "1", "true", "on", "yes",
    }


def supported(q, k, v):
    return (
        triton is not None
        and q.is_cuda
        and q.dtype == torch.float32
        and q.shape[0] > 0
        and q.shape[2] > 0
        and v.shape[2] > 0
        and q.shape[2] <= MAX_HEAD_DIM
        and v.shape[2] <= MAX_HEAD_DIM
    )


def _prepare_dropout(k, dropout_p, training, dropout_mask):
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must be in [0, 1)")
    if not training or dropout_p == 0.0:
        return None
    if dropout_mask is None:
        # Only the native Philox mask is consumed; input values are irrelevant.
        return torch.ops.aten.native_dropout(
            torch.empty(k.shape[:2], device=k.device, dtype=k.dtype),
            dropout_p, True)[1]
    if (dropout_mask.shape != k.shape[:2] or dropout_mask.dtype != torch.bool
            or dropout_mask.device != k.device):
        raise ValueError("dropout_mask must be bool [edges, heads] on the input device")
    return dropout_mask


def _prepare_edge_ids(k, edge_ids):
    if edge_ids is None or edge_ids.numel() == 0:
        return None
    if (edge_ids.ndim != 1 or edge_ids.shape[0] != k.shape[0]
            or edge_ids.device != k.device
            or edge_ids.dtype not in (torch.int32, torch.int64)):
        raise ValueError("edge_ids must be integer [edges] on the input device")
    return edge_ids


def reference(q, k, v, dst_ptr, dropout_p=0.0, training=False,
              dropout_mask=None, edge_ids=None):
    """Destination-major attention reference; empty segments produce zeros."""
    edge_ids = _prepare_edge_ids(k, edge_ids)
    dropout_mask = _prepare_dropout(k, dropout_p, training, dropout_mask)
    degree = dst_ptr[1:] - dst_ptr[:-1]
    edge_dst = torch.repeat_interleave(
        torch.arange(q.shape[0], device=q.device), degree)
    if edge_dst.numel() == 0:
        return q.new_zeros((*q.shape[:2], v.shape[2])) + (
            q.sum() + k.sum() + v.sum()) * 0

    segment_k = k if edge_ids is None else k.index_select(0, edge_ids)
    segment_v = v if edge_ids is None else v.index_select(0, edge_ids)
    score = F.leaky_relu((q[edge_dst] * segment_k).sum(-1), negative_slope=0.2)
    scatter_index = edge_dst[:, None].expand_as(score)
    maximum = q.new_full(q.shape[:2], -torch.inf).scatter_reduce(
        0, scatter_index, score, reduce="amax", include_self=True)
    weight = torch.exp(score - maximum[edge_dst])
    denominator = torch.zeros_like(maximum).scatter_add(0, scatter_index, weight)
    weight = weight / denominator[edge_dst]
    if dropout_mask is not None:
        if edge_ids is not None:
            dropout_mask = dropout_mask.index_select(0, edge_ids)
        weight = weight * dropout_mask / (1.0 - dropout_p)
    out_index = edge_dst[:, None, None].expand(-1, v.shape[1], v.shape[2])
    return q.new_zeros((*q.shape[:2], v.shape[2])).scatter_add(
        0, out_index, weight[:, :, None] * segment_v)


if triton is not None:

    @triton.jit
    def _forward_kernel(
        q, k, v, dst_ptr, dropout_mask, edge_ids, out, lse,
        HEADS: tl.constexpr,
        KEY_DIM: tl.constexpr,
        VALUE_DIM: tl.constexpr,
        BLOCK_EDGES: tl.constexpr,
        BLOCK_KEY: tl.constexpr,
        BLOCK_VALUE: tl.constexpr,
        USE_DROPOUT: tl.constexpr,
        USE_EDGE_IDS: tl.constexpr,
        DROPOUT_SCALE: tl.constexpr,
        FIXED_DEGREE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        dst = pid // HEADS
        head = pid - dst * HEADS
        if FIXED_DEGREE > 0:
            start = dst * FIXED_DEGREE
            end = start + FIXED_DEGREE
        else:
            start = tl.load(dst_ptr + dst)
            end = tl.load(dst_ptr + dst + 1)
        edge_offset = tl.arange(0, BLOCK_EDGES)
        key_col = tl.arange(0, BLOCK_KEY)
        value_col = tl.arange(0, BLOCK_VALUE)

        q_row = tl.load(
            q + (dst * HEADS + head) * KEY_DIM + key_col,
            mask=key_col < KEY_DIM,
            other=0.0,
        )
        maximum = -float("inf")
        denominator = 0.0
        result = tl.zeros((BLOCK_VALUE,), tl.float32)
        for tile_start in tl.range(start, end, BLOCK_EDGES):
            edge = tile_start + edge_offset
            edge_mask = edge < end
            eid = edge
            if USE_EDGE_IDS:
                eid = tl.load(edge_ids + edge, mask=edge_mask, other=0)
            k_rows = tl.load(
                k + (eid[:, None] * HEADS + head) * KEY_DIM + key_col[None, :],
                mask=edge_mask[:, None] & (key_col[None, :] < KEY_DIM),
                other=0.0,
            )
            raw_score = tl.sum(k_rows * q_row[None, :], axis=1)
            score = tl.where(raw_score > 0.0, raw_score, raw_score * 0.2)
            score = tl.where(edge_mask, score, -float("inf"))
            next_maximum = tl.maximum(maximum, tl.max(score, axis=0))
            previous_scale = tl.exp(maximum - next_maximum)
            numerator = tl.where(edge_mask, tl.exp(score - next_maximum), 0.0)
            denominator = denominator * previous_scale + tl.sum(numerator, axis=0)
            if USE_DROPOUT:
                keep = tl.load(dropout_mask + eid * HEADS + head,
                               mask=edge_mask, other=0)
                numerator = tl.where(keep, numerator * DROPOUT_SCALE, 0.0)
            v_rows = tl.load(
                v + (eid[:, None] * HEADS + head) * VALUE_DIM
                + value_col[None, :],
                mask=edge_mask[:, None] & (value_col[None, :] < VALUE_DIM),
                other=0.0,
            )
            result = result * previous_scale + tl.sum(
                numerator[:, None] * v_rows, axis=0)
            maximum = next_maximum
        result = result / tl.maximum(denominator, 1.0e-20)
        tl.store(
            out + (dst * HEADS + head) * VALUE_DIM + value_col,
            result,
            mask=value_col < VALUE_DIM,
        )
        segment_lse = maximum + tl.log(tl.maximum(denominator, 1.0e-20))
        tl.store(lse + dst * HEADS + head,
                 tl.where(end > start, segment_lse, -float("inf")))

    @triton.jit
    def _backward_kernel(
        q, k, v, dst_ptr, dropout_mask, edge_ids, lse, grad_out,
        grad_q, grad_k, grad_v,
        HEADS: tl.constexpr,
        KEY_DIM: tl.constexpr,
        VALUE_DIM: tl.constexpr,
        BLOCK_EDGES: tl.constexpr,
        BLOCK_KEY: tl.constexpr,
        BLOCK_VALUE: tl.constexpr,
        USE_DROPOUT: tl.constexpr,
        USE_EDGE_IDS: tl.constexpr,
        DROPOUT_SCALE: tl.constexpr,
        FIXED_DEGREE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        dst = pid // HEADS
        head = pid - dst * HEADS
        if FIXED_DEGREE > 0:
            start = dst * FIXED_DEGREE
            end = start + FIXED_DEGREE
        else:
            start = tl.load(dst_ptr + dst)
            end = tl.load(dst_ptr + dst + 1)
        edge_offset = tl.arange(0, BLOCK_EDGES)
        key_col = tl.arange(0, BLOCK_KEY)
        value_col = tl.arange(0, BLOCK_VALUE)

        q_row = tl.load(
            q + (dst * HEADS + head) * KEY_DIM + key_col,
            mask=key_col < KEY_DIM,
            other=0.0,
        )
        segment_lse = tl.load(lse + dst * HEADS + head)
        output_grad = tl.load(
            grad_out + (dst * HEADS + head) * VALUE_DIM + value_col,
            mask=value_col < VALUE_DIM,
            other=0.0,
        )
        center = 0.0
        for tile_start in tl.range(start, end, BLOCK_EDGES):
            edge = tile_start + edge_offset
            edge_mask = edge < end
            eid = edge
            if USE_EDGE_IDS:
                eid = tl.load(edge_ids + edge, mask=edge_mask, other=0)
            k_rows = tl.load(
                k + (eid[:, None] * HEADS + head) * KEY_DIM + key_col[None, :],
                mask=edge_mask[:, None] & (key_col[None, :] < KEY_DIM),
                other=0.0,
            )
            raw_score = tl.sum(k_rows * q_row[None, :], axis=1)
            score = tl.where(raw_score > 0.0, raw_score, raw_score * 0.2)
            weight = tl.where(edge_mask, tl.exp(score - segment_lse), 0.0)
            dropout_scale = 1.0
            if USE_DROPOUT:
                keep = tl.load(dropout_mask + eid * HEADS + head,
                               mask=edge_mask, other=0)
                dropout_scale = tl.where(keep, DROPOUT_SCALE, 0.0)
            v_rows = tl.load(
                v + (eid[:, None] * HEADS + head) * VALUE_DIM
                + value_col[None, :],
                mask=edge_mask[:, None] & (value_col[None, :] < VALUE_DIM),
                other=0.0,
            )
            grad_weight = tl.sum(
                v_rows * output_grad[None, :], axis=1) * dropout_scale
            center += tl.sum(weight * grad_weight, axis=0)

        q_grad = tl.zeros((BLOCK_KEY,), tl.float32)
        for tile_start in tl.range(start, end, BLOCK_EDGES):
            edge = tile_start + edge_offset
            edge_mask = edge < end
            eid = edge
            if USE_EDGE_IDS:
                eid = tl.load(edge_ids + edge, mask=edge_mask, other=0)
            k_rows = tl.load(
                k + (eid[:, None] * HEADS + head) * KEY_DIM + key_col[None, :],
                mask=edge_mask[:, None] & (key_col[None, :] < KEY_DIM),
                other=0.0,
            )
            raw_score = tl.sum(k_rows * q_row[None, :], axis=1)
            score = tl.where(raw_score > 0.0, raw_score, raw_score * 0.2)
            weight = tl.where(edge_mask, tl.exp(score - segment_lse), 0.0)
            dropout_scale = 1.0
            if USE_DROPOUT:
                keep = tl.load(dropout_mask + eid * HEADS + head,
                               mask=edge_mask, other=0)
                dropout_scale = tl.where(keep, DROPOUT_SCALE, 0.0)
            v_rows = tl.load(
                v + (eid[:, None] * HEADS + head) * VALUE_DIM
                + value_col[None, :],
                mask=edge_mask[:, None] & (value_col[None, :] < VALUE_DIM),
                other=0.0,
            )
            grad_weight = tl.sum(
                v_rows * output_grad[None, :], axis=1) * dropout_scale
            grad_raw = weight * (grad_weight - center) * tl.where(
                raw_score > 0.0, 1.0, 0.2)
            q_grad += tl.sum(grad_raw[:, None] * k_rows, axis=0)
            tl.store(
                grad_k + (eid[:, None] * HEADS + head) * KEY_DIM
                + key_col[None, :],
                grad_raw[:, None] * q_row[None, :],
                mask=edge_mask[:, None] & (key_col[None, :] < KEY_DIM),
            )
            tl.store(
                grad_v + (eid[:, None] * HEADS + head) * VALUE_DIM
                + value_col[None, :],
                (weight * dropout_scale)[:, None] * output_grad[None, :],
                mask=edge_mask[:, None] & (value_col[None, :] < VALUE_DIM),
            )
        tl.store(
            grad_q + (dst * HEADS + head) * KEY_DIM + key_col,
            q_grad,
            mask=key_col < KEY_DIM,
        )


class _TritonSegmentAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, dst_ptr, dropout_mask, edge_ids,
                dropout_p, block_edges, fixed_degree):
        q, k, v, dst_ptr = (x.contiguous() for x in (q, k, v, dst_ptr))
        dropout_mask = dropout_mask.contiguous()
        edge_ids = edge_ids.contiguous()
        out = torch.empty((*q.shape[:2], v.shape[2]), device=q.device, dtype=q.dtype)
        lse = torch.empty(q.shape[:2], device=q.device, dtype=q.dtype)
        block_key = triton.next_power_of_2(q.shape[2])
        block_value = triton.next_power_of_2(v.shape[2])
        grid = (q.shape[0] * q.shape[1],)
        _forward_kernel[grid](
            q, k, v, dst_ptr, dropout_mask, edge_ids, out, lse,
            HEADS=q.shape[1], KEY_DIM=q.shape[2], VALUE_DIM=v.shape[2],
            BLOCK_EDGES=block_edges, BLOCK_KEY=block_key,
            BLOCK_VALUE=block_value, USE_DROPOUT=dropout_p > 0.0,
            USE_EDGE_IDS=edge_ids.numel() > 0,
            DROPOUT_SCALE=1.0 / (1.0 - dropout_p),
            FIXED_DEGREE=fixed_degree, num_warps=4,
        )
        ctx.save_for_backward(q, k, v, dst_ptr, dropout_mask, edge_ids, lse)
        ctx.block_edges = block_edges
        ctx.dropout_p = dropout_p
        ctx.fixed_degree = fixed_degree
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, dst_ptr, dropout_mask, edge_ids, lse = ctx.saved_tensors
        grad_q = torch.empty_like(q)
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)
        grid = (q.shape[0] * q.shape[1],)
        _backward_kernel[grid](
            q, k, v, dst_ptr, dropout_mask, edge_ids, lse,
            grad_out.contiguous(),
            grad_q, grad_k, grad_v,
            HEADS=q.shape[1], KEY_DIM=q.shape[2], VALUE_DIM=v.shape[2],
            BLOCK_EDGES=ctx.block_edges,
            BLOCK_KEY=triton.next_power_of_2(q.shape[2]),
            BLOCK_VALUE=triton.next_power_of_2(v.shape[2]),
            USE_DROPOUT=ctx.dropout_p > 0.0,
            USE_EDGE_IDS=edge_ids.numel() > 0,
            DROPOUT_SCALE=1.0 / (1.0 - ctx.dropout_p),
            FIXED_DEGREE=ctx.fixed_degree, num_warps=4,
        )
        return grad_q, grad_k, grad_v, None, None, None, None, None, None


def _validate(q, k, v, dst_ptr, edge_ids, validate_ptr):
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must be rank-3 [rows, heads, features]")
    if q.shape[1:] != k.shape[1:] or k.shape[:2] != v.shape[:2]:
        raise ValueError("q/k heads and key dimensions, and k/v edge counts must match")
    if q.device != k.device or q.device != v.device or q.device != dst_ptr.device:
        raise ValueError("q, k, v, and dst_ptr must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if dst_ptr.ndim != 1 or dst_ptr.numel() != q.shape[0] + 1:
        raise ValueError("dst_ptr must have one entry per destination plus one")
    if dst_ptr.dtype not in (torch.int32, torch.int64):
        raise ValueError("dst_ptr must use int32 or int64")
    _prepare_edge_ids(k, edge_ids)
    if validate_ptr:
        degree = dst_ptr[1:] - dst_ptr[:-1]
        if (dst_ptr[0].item() != 0 or dst_ptr[-1].item() != k.shape[0]
                or bool((degree < 0).any().item())):
            raise ValueError("dst_ptr must be monotonic and cover every edge exactly once")


def attention(q, k, v, dst_ptr, dropout_p=0.0, training=False,
              dropout_mask=None, edge_ids=None, validate=True):
    """Fused attention; validate=False trusts builder-owned CSC pointer contents."""
    edge_ids = _prepare_edge_ids(k, edge_ids)
    _validate(q, k, v, dst_ptr, edge_ids, validate)
    dropout_mask = _prepare_dropout(k, dropout_p, training, dropout_mask)
    if not supported(q, k, v):
        return reference(q, k, v, dst_ptr, dropout_p, training, dropout_mask,
                         edge_ids)
    kernel_mask = dropout_mask
    if kernel_mask is None:
        kernel_mask = torch.empty(0, device=q.device, dtype=torch.bool)
    kernel_edge_ids = edge_ids
    if kernel_edge_ids is None:
        kernel_edge_ids = torch.empty(0, device=q.device, dtype=torch.int64)
    # ponytail: builder owns the permutation; use atomics if repeated EIDs become valid.
    return _TritonSegmentAttention.apply(
        q, k, v, dst_ptr, kernel_mask, kernel_edge_ids,
        dropout_p if dropout_mask is not None else 0.0,
        EDGE_TILE, 0)


def uniform_attention(q, k, v, degree, dropout_p=0.0, training=False,
                      dropout_mask=None):
    """Fused attention for equal-size contiguous segments."""
    if not isinstance(degree, int) or degree <= 0:
        raise ValueError("degree must be a positive integer")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must be rank-3 [rows, heads, features]")
    if q.shape[1:] != k.shape[1:] or k.shape[:2] != v.shape[:2]:
        raise ValueError("q/k heads and key dimensions, and k/v rows must match")
    if k.shape[0] != q.shape[0] * degree:
        raise ValueError("k/v rows must equal q rows times degree")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    dropout_mask = _prepare_dropout(
        k, dropout_p, training, dropout_mask)
    if not supported(q, k, v):
        dst_ptr = torch.arange(
            q.shape[0] + 1, device=q.device, dtype=torch.int64) * degree
        return reference(
            q, k, v, dst_ptr, dropout_p, training, dropout_mask)
    kernel_mask = dropout_mask
    if kernel_mask is None:
        kernel_mask = torch.empty(0, device=q.device, dtype=torch.bool)
    empty = torch.empty(0, device=q.device, dtype=torch.int64)
    return _TritonSegmentAttention.apply(
        q, k, v, empty, kernel_mask, empty,
        dropout_p if dropout_mask is not None else 0.0,
        EDGE_TILE, degree)
