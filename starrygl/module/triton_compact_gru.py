import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


MAX_HIDDEN_DIM = 128


def supported(mail_gates, time_gates, hidden_gates, state_to_uid, state_memory):
    tensors = (mail_gates, time_gates, hidden_gates, state_memory)
    return (
        triton is not None
        and all(x.is_cuda and x.dtype == torch.float32 and x.is_contiguous()
                for x in tensors)
        and state_to_uid.is_cuda
        and state_to_uid.dtype in (torch.int32, torch.int64)
        and state_to_uid.is_contiguous()
        and time_gates.shape[0] > 0
        and state_memory.shape[1] <= MAX_HIDDEN_DIM
        and not state_memory.requires_grad
        and not torch.are_deterministic_algorithms_enabled()
    )


if triton is not None:

    @triton.jit
    def _forward_kernel(mail_gates, time_gates, hidden_gates, state_to_uid,
                        state_memory, output,
                        HIDDEN_DIM: tl.constexpr, BLOCK_H: tl.constexpr):
        state = tl.program_id(0)
        col = tl.arange(0, BLOCK_H)
        mask = col < HIDDEN_DIM
        uid = tl.load(state_to_uid + state)
        gate_width = 3 * HIDDEN_DIM

        input_r = tl.load(mail_gates + uid * gate_width + col,
                          mask=mask, other=0.0)
        input_z = tl.load(
            mail_gates + uid * gate_width + HIDDEN_DIM + col,
            mask=mask, other=0.0)
        input_n = tl.load(
            mail_gates + uid * gate_width + 2 * HIDDEN_DIM + col,
            mask=mask, other=0.0)
        input_r += tl.load(time_gates + state * gate_width + col,
                           mask=mask, other=0.0)
        input_z += tl.load(
            time_gates + state * gate_width + HIDDEN_DIM + col,
            mask=mask, other=0.0)
        input_n += tl.load(
            time_gates + state * gate_width + 2 * HIDDEN_DIM + col,
            mask=mask, other=0.0)

        hidden_r = tl.load(hidden_gates + uid * gate_width + col,
                           mask=mask, other=0.0)
        hidden_z = tl.load(
            hidden_gates + uid * gate_width + HIDDEN_DIM + col,
            mask=mask, other=0.0)
        hidden_n = tl.load(
            hidden_gates + uid * gate_width + 2 * HIDDEN_DIM + col,
            mask=mask, other=0.0)
        previous = tl.load(state_memory + state * HIDDEN_DIM + col,
                           mask=mask, other=0.0)

        reset = tl.sigmoid(input_r + hidden_r)
        update = tl.sigmoid(input_z + hidden_z)
        candidate = 2.0 * tl.sigmoid(
            2.0 * (input_n + reset * hidden_n)) - 1.0
        result = candidate + update * (previous - candidate)
        tl.store(output + state * HIDDEN_DIM + col, result, mask=mask)

    @triton.jit
    def _backward_kernel(mail_gates, time_gates, hidden_gates, state_to_uid,
                         state_memory, grad_output, grad_mail, grad_time,
                         grad_hidden,
                         HIDDEN_DIM: tl.constexpr, BLOCK_H: tl.constexpr):
        state = tl.program_id(0)
        col = tl.arange(0, BLOCK_H)
        mask = col < HIDDEN_DIM
        uid = tl.load(state_to_uid + state)
        gate_width = 3 * HIDDEN_DIM

        input_r = tl.load(mail_gates + uid * gate_width + col,
                          mask=mask, other=0.0)
        input_z = tl.load(
            mail_gates + uid * gate_width + HIDDEN_DIM + col,
            mask=mask, other=0.0)
        input_n = tl.load(
            mail_gates + uid * gate_width + 2 * HIDDEN_DIM + col,
            mask=mask, other=0.0)
        input_r += tl.load(time_gates + state * gate_width + col,
                           mask=mask, other=0.0)
        input_z += tl.load(
            time_gates + state * gate_width + HIDDEN_DIM + col,
            mask=mask, other=0.0)
        input_n += tl.load(
            time_gates + state * gate_width + 2 * HIDDEN_DIM + col,
            mask=mask, other=0.0)
        hidden_r = tl.load(hidden_gates + uid * gate_width + col,
                           mask=mask, other=0.0)
        hidden_z = tl.load(
            hidden_gates + uid * gate_width + HIDDEN_DIM + col,
            mask=mask, other=0.0)
        hidden_n = tl.load(
            hidden_gates + uid * gate_width + 2 * HIDDEN_DIM + col,
            mask=mask, other=0.0)
        previous = tl.load(state_memory + state * HIDDEN_DIM + col,
                           mask=mask, other=0.0)
        grad = tl.load(grad_output + state * HIDDEN_DIM + col,
                       mask=mask, other=0.0)

        reset = tl.sigmoid(input_r + hidden_r)
        update = tl.sigmoid(input_z + hidden_z)
        candidate = 2.0 * tl.sigmoid(
            2.0 * (input_n + reset * hidden_n)) - 1.0
        grad_candidate = grad * (1.0 - update)
        grad_update = grad * (previous - candidate)
        grad_candidate_pre = grad_candidate * (1.0 - candidate * candidate)
        grad_reset = grad_candidate_pre * hidden_n
        grad_reset_pre = grad_reset * reset * (1.0 - reset)
        grad_update_pre = grad_update * update * (1.0 - update)
        grad_hidden_n = grad_candidate_pre * reset

        tl.atomic_add(grad_mail + uid * gate_width + col,
                      grad_reset_pre, mask=mask)
        tl.atomic_add(grad_mail + uid * gate_width + HIDDEN_DIM + col,
                      grad_update_pre, mask=mask)
        tl.atomic_add(grad_mail + uid * gate_width + 2 * HIDDEN_DIM + col,
                      grad_candidate_pre, mask=mask)
        tl.store(grad_time + state * gate_width + col,
                 grad_reset_pre, mask=mask)
        tl.store(grad_time + state * gate_width + HIDDEN_DIM + col,
                 grad_update_pre, mask=mask)
        tl.store(grad_time + state * gate_width + 2 * HIDDEN_DIM + col,
                 grad_candidate_pre, mask=mask)
        tl.atomic_add(grad_hidden + uid * gate_width + col,
                      grad_reset_pre, mask=mask)
        tl.atomic_add(grad_hidden + uid * gate_width + HIDDEN_DIM + col,
                      grad_update_pre, mask=mask)
        tl.atomic_add(grad_hidden + uid * gate_width + 2 * HIDDEN_DIM + col,
                      grad_hidden_n, mask=mask)


class _CompactGRUPointwise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mail_gates, time_gates, hidden_gates, state_to_uid,
                state_memory):
        output = torch.empty_like(state_memory)
        hidden_dim = state_memory.shape[1]
        _forward_kernel[(state_memory.shape[0],)](
            mail_gates, time_gates, hidden_gates, state_to_uid, state_memory,
            output, HIDDEN_DIM=hidden_dim,
            BLOCK_H=triton.next_power_of_2(hidden_dim), num_warps=4,
        )
        ctx.save_for_backward(
            mail_gates, time_gates, hidden_gates, state_to_uid, state_memory)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mail_gates, time_gates, hidden_gates, state_to_uid, state_memory = (
            ctx.saved_tensors)
        grad_mail = torch.zeros_like(mail_gates)
        grad_time = torch.empty_like(time_gates)
        grad_hidden = torch.zeros_like(hidden_gates)
        hidden_dim = state_memory.shape[1]
        _backward_kernel[(state_memory.shape[0],)](
            mail_gates, time_gates, hidden_gates, state_to_uid, state_memory,
            grad_output.contiguous(), grad_mail, grad_time, grad_hidden,
            HIDDEN_DIM=hidden_dim,
            BLOCK_H=triton.next_power_of_2(hidden_dim), num_warps=4,
        )
        return grad_mail, grad_time, grad_hidden, None, None


def pointwise(mail_gates, time_gates, hidden_gates, state_to_uid,
              state_memory):
    if not supported(mail_gates, time_gates, hidden_gates, state_to_uid,
                     state_memory):
        raise ValueError('fused compact GRU requires contiguous CUDA FP32 inputs')
    return _CompactGRUPointwise.apply(
        mail_gates, time_gates, hidden_gates, state_to_uid, state_memory)
