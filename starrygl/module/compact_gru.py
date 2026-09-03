import torch
from torch.nn import functional as F

from starrygl.module import triton_compact_gru


def compact_rnn_cell(cell, mailbox, memory, time_feat, state_to_uid,
                     state_memory):
    if any(value.requires_grad for value in (mailbox, memory, state_memory)):
        state_mailbox = mailbox.index_select(0, state_to_uid)
        return cell(torch.cat((state_mailbox, time_feat), dim=1), state_memory)

    mail_dim = mailbox.shape[1]
    if cell.input_size != mail_dim + time_feat.shape[1]:
        raise ValueError('compact RNN inputs do not match RNNCell.input_size')

    object_gates = F.linear(
        mailbox, cell.weight_ih[:, :mail_dim], cell.bias_ih)
    object_gates = object_gates + F.linear(
        memory, cell.weight_hh, cell.bias_hh)
    state_gates = torch.addmm(
        object_gates.index_select(0, state_to_uid),
        time_feat,
        cell.weight_ih[:, mail_dim:].t(),
    )
    if cell.nonlinearity == 'tanh':
        return torch.tanh(state_gates)
    if cell.nonlinearity == 'relu':
        return torch.relu(state_gates)
    raise ValueError('compact RNN supports tanh or relu RNNCell')


def compact_gru_cell(cell, mailbox, memory, time_feat, state_to_uid, state_memory):
    if (
        not mailbox.is_cuda
        or mailbox.dtype not in (torch.float16, torch.float32, torch.float64)
        or not hasattr(torch.ops.aten, '_thnn_fused_gru_cell')
        or state_memory.requires_grad
    ):
        state_mailbox = mailbox.index_select(0, state_to_uid)
        return cell(torch.cat((state_mailbox, time_feat), dim=1), state_memory)

    mail_dim = mailbox.shape[1]
    if cell.input_size != mail_dim + time_feat.shape[1]:
        raise ValueError('compact GRU inputs do not match GRUCell.input_size')

    if (
        triton_compact_gru.triton is not None
        and all(x.is_cuda and x.dtype == torch.float32
                for x in (mailbox, memory, time_feat, state_memory))
        and state_to_uid.is_cuda
    ):
        mail_gates = F.linear(mailbox, cell.weight_ih[:, :mail_dim], None)
        time_gates = F.linear(
            time_feat, cell.weight_ih[:, mail_dim:], cell.bias_ih)
        hidden_gates = F.linear(memory, cell.weight_hh, cell.bias_hh)
        if triton_compact_gru.supported(
                mail_gates, time_gates, hidden_gates, state_to_uid,
                state_memory):
            return triton_compact_gru.pointwise(
                mail_gates, time_gates, hidden_gates, state_to_uid,
                state_memory)

    input_gates = F.linear(mailbox, cell.weight_ih[:, :mail_dim], None)
    input_gates = input_gates.index_select(0, state_to_uid)
    input_gates = torch.addmm(
        input_gates,
        time_feat,
        cell.weight_ih[:, mail_dim:].t(),
    )
    hidden_gates = F.linear(memory, cell.weight_hh, None)
    hidden_gates = hidden_gates.index_select(0, state_to_uid)
    return torch.ops.aten._thnn_fused_gru_cell(
        input_gates,
        hidden_gates,
        state_memory,
        cell.bias_ih,
        cell.bias_hh,
    )[0]
