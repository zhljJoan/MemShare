"""Segment attention that reads compact K/V projection components directly."""

import torch

from starrygl.module import triton_segment_attention as _base

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised by CPU fallback
    triton = None
    tl = None


def supported(q, state_kv, edge_kv, time_kv):
    return (
        triton is not None and q.is_cuda and q.dtype == torch.float32
        and state_kv.dtype == q.dtype and edge_kv.dtype == q.dtype
        and time_kv.dtype == q.dtype and q.shape[2] <= _base.MAX_HEAD_DIM
        and state_kv.shape[1] == 2 * q.shape[1] * q.shape[2]
        and edge_kv.shape[1] == state_kv.shape[1]
        and time_kv.shape[1] == state_kv.shape[1]
    )


if triton is not None:

    @triton.jit
    def _forward_projected_kernel(
        q, state_kv, edge_kv, time_kv, edge_src, edge_obj, edge_ids, dst_ptr,
        dropout_mask, out, lse,
        HEADS: tl.constexpr, KEY_DIM: tl.constexpr,
        BLOCK_EDGES: tl.constexpr, BLOCK_KEY: tl.constexpr,
        USE_DROPOUT: tl.constexpr, USE_EDGE_OBJ: tl.constexpr,
        USE_EDGE_IDS: tl.constexpr,
        DROPOUT_SCALE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        dst = pid // HEADS
        head = pid - dst * HEADS
        start = tl.load(dst_ptr + dst)
        end = tl.load(dst_ptr + dst + 1)
        edge_offset = tl.arange(0, BLOCK_EDGES)
        key_col = tl.arange(0, BLOCK_KEY)
        total_dim = 2 * HEADS * KEY_DIM

        q_row = tl.load(
            q + (dst * HEADS + head) * KEY_DIM + key_col,
            mask=key_col < KEY_DIM, other=0.0,
        )
        maximum = -float("inf")
        denominator = 0.0
        result = tl.zeros((BLOCK_KEY,), tl.float32)
        for tile_start in tl.range(start, end, BLOCK_EDGES):
            edge = tile_start + edge_offset
            edge_mask = edge < end
            occurrence = edge
            if USE_EDGE_IDS:
                occurrence = tl.load(edge_ids + edge, mask=edge_mask, other=0)
            src = tl.load(edge_src + occurrence, mask=edge_mask, other=0)
            obj = occurrence
            if USE_EDGE_OBJ:
                obj = tl.load(edge_obj + occurrence, mask=edge_mask, other=0)
            component_offset = head * KEY_DIM + key_col[None, :]
            state_ptr = state_kv + src[:, None] * total_dim + component_offset
            edge_ptr = edge_kv + obj[:, None] * total_dim + component_offset
            time_ptr = time_kv + occurrence[:, None] * total_dim + component_offset
            k_rows = (
                tl.load(state_ptr, mask=edge_mask[:, None] & (key_col[None, :] < KEY_DIM), other=0.0)
                + tl.load(edge_ptr, mask=edge_mask[:, None] & (key_col[None, :] < KEY_DIM), other=0.0)
                + tl.load(time_ptr, mask=edge_mask[:, None] & (key_col[None, :] < KEY_DIM), other=0.0)
            )
            raw_score = tl.sum(k_rows * q_row[None, :], axis=1)
            score = tl.where(raw_score > 0.0, raw_score, raw_score * 0.2)
            score = tl.where(edge_mask, score, -float("inf"))
            next_maximum = tl.maximum(maximum, tl.max(score, axis=0))
            previous_scale = tl.exp(maximum - next_maximum)
            numerator = tl.where(edge_mask, tl.exp(score - next_maximum), 0.0)
            denominator = denominator * previous_scale + tl.sum(numerator, axis=0)
            if USE_DROPOUT:
                keep = tl.load(dropout_mask + occurrence * HEADS + head, mask=edge_mask, other=0)
                numerator = tl.where(keep, numerator * DROPOUT_SCALE, 0.0)
            value_offset = HEADS * KEY_DIM + head * KEY_DIM + key_col[None, :]
            v_rows = (
                tl.load(state_kv + src[:, None] * total_dim + value_offset,
                        mask=edge_mask[:, None] & (key_col[None, :] < KEY_DIM), other=0.0)
                + tl.load(edge_kv + obj[:, None] * total_dim + value_offset,
                          mask=edge_mask[:, None] & (key_col[None, :] < KEY_DIM), other=0.0)
                + tl.load(time_kv + occurrence[:, None] * total_dim + value_offset,
                          mask=edge_mask[:, None] & (key_col[None, :] < KEY_DIM), other=0.0)
            )
            result = result * previous_scale + tl.sum(numerator[:, None] * v_rows, axis=0)
            maximum = next_maximum
        result = result / tl.maximum(denominator, 1.0e-20)
        tl.store(out + (dst * HEADS + head) * KEY_DIM + key_col, result, mask=key_col < KEY_DIM)
        tl.store(lse + dst * HEADS + head,
                 tl.where(end > start, maximum + tl.log(tl.maximum(denominator, 1.0e-20)), -float("inf")))

    @triton.jit
    def _backward_projected_kernel(
        q, state_kv, edge_kv, time_kv, edge_src, edge_obj, edge_ids, dst_ptr,
        dropout_mask, lse, grad_out, grad_q, grad_state, grad_edge, grad_time,
        HEADS: tl.constexpr, KEY_DIM: tl.constexpr,
        BLOCK_EDGES: tl.constexpr, BLOCK_KEY: tl.constexpr,
        USE_DROPOUT: tl.constexpr, USE_EDGE_OBJ: tl.constexpr,
        USE_EDGE_IDS: tl.constexpr,
        DROPOUT_SCALE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        dst = pid // HEADS
        head = pid - dst * HEADS
        start = tl.load(dst_ptr + dst)
        end = tl.load(dst_ptr + dst + 1)
        edge_offset = tl.arange(0, BLOCK_EDGES)
        key_col = tl.arange(0, BLOCK_KEY)
        total_dim = 2 * HEADS * KEY_DIM
        value_base = HEADS * KEY_DIM + head * KEY_DIM
        q_row = tl.load(q + (dst * HEADS + head) * KEY_DIM + key_col,
                        mask=key_col < KEY_DIM, other=0.0)
        segment_lse = tl.load(lse + dst * HEADS + head)
        output_grad = tl.load(grad_out + (dst * HEADS + head) * KEY_DIM + key_col,
                              mask=key_col < KEY_DIM, other=0.0)
        center = 0.0
        for tile_start in tl.range(start, end, BLOCK_EDGES):
            edge = tile_start + edge_offset
            edge_mask = edge < end
            occurrence = edge
            if USE_EDGE_IDS:
                occurrence = tl.load(edge_ids + edge, mask=edge_mask, other=0)
            src = tl.load(edge_src + occurrence, mask=edge_mask, other=0)
            obj = occurrence
            if USE_EDGE_OBJ:
                obj = tl.load(edge_obj + occurrence, mask=edge_mask, other=0)
            offset = head * KEY_DIM + key_col[None, :]
            k_rows = (
                tl.load(state_kv + src[:, None] * total_dim + offset,
                        mask=edge_mask[:, None] & (key_col[None, :] < KEY_DIM), other=0.0)
                + tl.load(edge_kv + obj[:, None] * total_dim + offset,
                          mask=edge_mask[:, None] & (key_col[None, :] < KEY_DIM), other=0.0)
                + tl.load(time_kv + occurrence[:, None] * total_dim + offset,
                          mask=edge_mask[:, None] & (key_col[None, :] < KEY_DIM), other=0.0)
            )
            value_offset = value_base + key_col[None, :]
            v_rows = (
                tl.load(state_kv + src[:, None] * total_dim + value_offset,
                        mask=edge_mask[:, None] & (key_col[None, :] < KEY_DIM), other=0.0)
                + tl.load(edge_kv + obj[:, None] * total_dim + value_offset,
                          mask=edge_mask[:, None] & (key_col[None, :] < KEY_DIM), other=0.0)
                + tl.load(time_kv + occurrence[:, None] * total_dim + value_offset,
                          mask=edge_mask[:, None] & (key_col[None, :] < KEY_DIM), other=0.0)
            )
            raw_score = tl.sum(k_rows * q_row[None, :], axis=1)
            score = tl.where(raw_score > 0.0, raw_score, raw_score * 0.2)
            weight = tl.where(edge_mask, tl.exp(score - segment_lse), 0.0)
            scale = 1.0
            if USE_DROPOUT:
                keep = tl.load(dropout_mask + occurrence * HEADS + head,
                               mask=edge_mask, other=0)
                scale = tl.where(keep, DROPOUT_SCALE, 0.0)
            grad_weight = tl.sum(v_rows * output_grad[None, :], axis=1) * scale
            center += tl.sum(weight * grad_weight, axis=0)

        q_grad = tl.zeros((BLOCK_KEY,), tl.float32)
        for tile_start in tl.range(start, end, BLOCK_EDGES):
            edge = tile_start + edge_offset
            edge_mask = edge < end
            occurrence = edge
            if USE_EDGE_IDS:
                occurrence = tl.load(edge_ids + edge, mask=edge_mask, other=0)
            src = tl.load(edge_src + occurrence, mask=edge_mask, other=0)
            obj = occurrence
            if USE_EDGE_OBJ:
                obj = tl.load(edge_obj + occurrence, mask=edge_mask, other=0)
            offset = head * KEY_DIM + key_col[None, :]
            k_rows = (
                tl.load(state_kv + src[:, None] * total_dim + offset,
                        mask=edge_mask[:, None] & (key_col[None, :] < KEY_DIM), other=0.0)
                + tl.load(edge_kv + obj[:, None] * total_dim + offset,
                          mask=edge_mask[:, None] & (key_col[None, :] < KEY_DIM), other=0.0)
                + tl.load(time_kv + occurrence[:, None] * total_dim + offset,
                          mask=edge_mask[:, None] & (key_col[None, :] < KEY_DIM), other=0.0)
            )
            value_offset = value_base + key_col[None, :]
            v_rows = (
                tl.load(state_kv + src[:, None] * total_dim + value_offset,
                        mask=edge_mask[:, None] & (key_col[None, :] < KEY_DIM), other=0.0)
                + tl.load(edge_kv + obj[:, None] * total_dim + value_offset,
                          mask=edge_mask[:, None] & (key_col[None, :] < KEY_DIM), other=0.0)
                + tl.load(time_kv + occurrence[:, None] * total_dim + value_offset,
                          mask=edge_mask[:, None] & (key_col[None, :] < KEY_DIM), other=0.0)
            )
            raw_score = tl.sum(k_rows * q_row[None, :], axis=1)
            score = tl.where(raw_score > 0.0, raw_score, raw_score * 0.2)
            weight = tl.where(edge_mask, tl.exp(score - segment_lse), 0.0)
            scale = 1.0
            if USE_DROPOUT:
                keep = tl.load(dropout_mask + occurrence * HEADS + head,
                               mask=edge_mask, other=0)
                scale = tl.where(keep, DROPOUT_SCALE, 0.0)
            grad_weight = tl.sum(v_rows * output_grad[None, :], axis=1) * scale
            grad_raw = weight * (grad_weight - center)
            grad_raw *= tl.where(raw_score > 0.0, 1.0, 0.2)
            q_grad += tl.sum(grad_raw[:, None] * k_rows, axis=0)
            grad_k = grad_raw[:, None] * q_row[None, :]
            grad_v = (weight * scale)[:, None] * output_grad[None, :]
            tl.atomic_add(grad_state + src[:, None] * total_dim + offset,
                          grad_k, mask=edge_mask[:, None] & (key_col[None, :] < KEY_DIM))
            tl.atomic_add(grad_edge + obj[:, None] * total_dim + offset,
                          grad_k, mask=edge_mask[:, None] & (key_col[None, :] < KEY_DIM))
            tl.atomic_add(grad_time + occurrence[:, None] * total_dim + offset,
                          grad_k, mask=edge_mask[:, None] & (key_col[None, :] < KEY_DIM))
            tl.atomic_add(grad_state + src[:, None] * total_dim + value_offset,
                          grad_v, mask=edge_mask[:, None] & (key_col[None, :] < KEY_DIM))
            tl.atomic_add(grad_edge + obj[:, None] * total_dim + value_offset,
                          grad_v, mask=edge_mask[:, None] & (key_col[None, :] < KEY_DIM))
            tl.atomic_add(grad_time + occurrence[:, None] * total_dim + value_offset,
                          grad_v, mask=edge_mask[:, None] & (key_col[None, :] < KEY_DIM))
        tl.store(grad_q + (dst * HEADS + head) * KEY_DIM + key_col,
                 q_grad, mask=key_col < KEY_DIM)


class _ProjectedSegmentAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, state_kv, edge_kv, time_kv, edge_src, dst_ptr,
                dropout_mask, edge_obj, edge_ids, dropout_p):
        q, state_kv, edge_kv, time_kv, edge_src, dst_ptr = (
            x.contiguous() for x in (q, state_kv, edge_kv, time_kv, edge_src, dst_ptr)
        )
        dropout_mask = dropout_mask.contiguous()
        edge_obj = edge_obj.contiguous()
        edge_ids = edge_ids.contiguous()
        out = torch.empty_like(q)
        lse = torch.empty(q.shape[:2], device=q.device, dtype=q.dtype)
        block_key = triton.next_power_of_2(q.shape[2])
        _forward_projected_kernel[(q.shape[0] * q.shape[1],)](
            q, state_kv, edge_kv, time_kv, edge_src, edge_obj, edge_ids, dst_ptr,
            dropout_mask, out, lse,
            HEADS=q.shape[1], KEY_DIM=q.shape[2], BLOCK_EDGES=_base.EDGE_TILE,
            BLOCK_KEY=block_key, USE_DROPOUT=dropout_p > 0.0,
            USE_EDGE_OBJ=edge_obj.numel() > 0, USE_EDGE_IDS=edge_ids.numel() > 0,
            DROPOUT_SCALE=1.0 / (1.0 - dropout_p), num_warps=4,
        )
        ctx.save_for_backward(q, state_kv, edge_kv, time_kv, edge_src, dst_ptr,
                              dropout_mask, edge_obj, edge_ids, lse)
        ctx.dropout_p = dropout_p
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (q, state_kv, edge_kv, time_kv, edge_src, dst_ptr, dropout_mask,
         edge_obj, edge_ids, lse) = ctx.saved_tensors
        grad_q = torch.empty_like(q)
        grad_state = torch.zeros_like(state_kv)
        grad_edge = torch.zeros_like(edge_kv)
        grad_time = torch.zeros_like(time_kv)
        _backward_projected_kernel[(q.shape[0] * q.shape[1],)](
            q, state_kv, edge_kv, time_kv, edge_src, edge_obj, edge_ids, dst_ptr,
            dropout_mask, lse, grad_out.contiguous(),
            grad_q, grad_state, grad_edge, grad_time,
            HEADS=q.shape[1], KEY_DIM=q.shape[2], BLOCK_EDGES=_base.EDGE_TILE,
            BLOCK_KEY=triton.next_power_of_2(q.shape[2]),
            USE_DROPOUT=ctx.dropout_p > 0.0,
            USE_EDGE_OBJ=edge_obj.numel() > 0, USE_EDGE_IDS=edge_ids.numel() > 0,
            DROPOUT_SCALE=1.0 / (1.0 - ctx.dropout_p), num_warps=4,
        )
        return (grad_q, grad_state, grad_edge, grad_time, None, None, None,
                None, None, None)


def attention(q, state_kv, edge_kv, time_kv, edge_src, dst_ptr,
              dropout_p=0.0, training=False, dropout_mask=None,
              edge_obj=None, edge_ids=None, validate=True):
    """Read compact projected K/V components without occurrence K/V materialisation."""
    if edge_obj is None:
        edge_obj = torch.empty(0, device=q.device, dtype=torch.int64)
    edge_ids = _base._prepare_edge_ids(time_kv, edge_ids)
    if not supported(q, state_kv, edge_kv, time_kv):
        state_occ = state_kv.index_select(0, edge_src)
        obj = edge_obj if edge_obj.numel() else torch.arange(time_kv.shape[0], device=q.device)
        edge_occ = edge_kv.index_select(0, obj)
        projected = state_occ + edge_occ + time_kv
        total = q.shape[1] * q.shape[2]
        k = projected[:, :total].view(time_kv.shape[0], q.shape[1], q.shape[2])
        v = projected[:, total:].view_as(k)
        return _base.attention(q, k, v, dst_ptr, dropout_p, training,
                               dropout_mask, edge_ids, validate)
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError("dropout_p must be in [0, 1)")
    if not training or dropout_p == 0.0:
        mask = None
    elif dropout_mask is None:
        mask = torch.rand((time_kv.shape[0], q.shape[1]), device=q.device) >= dropout_p
    else:
        if (dropout_mask.shape != (time_kv.shape[0], q.shape[1])
                or dropout_mask.dtype != torch.bool or dropout_mask.device != q.device):
            raise ValueError("dropout_mask must be bool [edges, heads] on the input device")
        mask = dropout_mask
    if mask is None:
        mask = torch.empty(0, device=q.device, dtype=torch.bool)
        effective_p = 0.0
    else:
        effective_p = dropout_p
    if edge_ids is None:
        edge_ids = torch.empty(0, device=q.device, dtype=torch.int64)
    return _ProjectedSegmentAttention.apply(
        q, state_kv, edge_kv, time_kv, edge_src, dst_ptr,
        mask, edge_obj, edge_ids, effective_p,
    )
