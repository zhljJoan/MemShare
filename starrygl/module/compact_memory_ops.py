import torch
from torch.nn import functional as F
from torch_scatter import scatter_max

from starrygl.module import triton_segment_attention


def latest_object_selection(
        state_to_object, state_rows, timestamps, object_ids):
    """Select each used object's latest candidate in object-table order.

    ``object_ids`` must be owner-grouped when the result is routed directly.
    """
    tensors = (state_to_object, state_rows, timestamps, object_ids)
    if any(value.ndim != 1 for value in tensors):
        raise ValueError('compact latest selection expects one-dimensional inputs')
    if state_rows.shape[0] != timestamps.shape[0]:
        raise ValueError('candidate state rows and timestamps must have equal length')
    if any(value.device != object_ids.device for value in tensors[:-1]):
        raise ValueError('compact latest selection inputs must share one device')
    object_slots = state_to_object.index_select(0, state_rows)
    if object_slots.numel() == 0:
        return object_ids[:0], object_slots

    _, positions = scatter_max(
        timestamps, object_slots, dim=0, dim_size=object_ids.shape[0])
    used = torch.zeros(
        object_ids.shape[0], dtype=torch.bool, device=object_ids.device)
    used[object_slots] = True
    used_slots = used.nonzero(as_tuple=False).flatten()
    return (
        object_ids.index_select(0, used_slots),
        positions.index_select(0, used_slots),
    )


def shared_object_plan(state_to_object, shared_object_mask):
    """Return used shared objects and their state-expansion rows."""
    used_shared = torch.zeros_like(shared_object_mask)
    used_shared[state_to_object] = shared_object_mask[state_to_object]
    state_mask = used_shared.index_select(0, state_to_object)
    shared_rank = used_shared.long().cumsum(0) - 1
    rows = shared_rank.index_select(0, state_to_object[state_mask])
    return used_shared, state_mask, rows


def normalize_transition(values):
    if values.numel() == 0:
        return values
    original_max = values.max()
    shifted = values - values.min()
    normalized = 2 * shifted / shifted.max() - 1
    return torch.where(original_max == 0, values, normalized)


def attach_node_features(
        updated_memory, node_features, state_to_object, projection=None):
    if projection is not None:
        node_features = projection(node_features)
    return updated_memory + node_features.index_select(0, state_to_object)


def _object_state_linear(linear, object_input, state_input, state_to_object):
    object_width = object_input.shape[-1]
    object_output = F.linear(
        object_input, linear.weight[:, :object_width], linear.bias)
    state_output = F.linear(
        state_input, linear.weight[:, object_width:], None)
    return object_output.index_select(0, state_to_object) + state_output


def compact_transformer_memory(
        updater, mailbox, memory, mailbox_ts, state_to_uid,
        state_ts, state_memory):
    mailbox_size = updater.memory_param['mailbox_size']
    mailbox = mailbox.reshape(mailbox.shape[0], mailbox_size, -1)
    mailbox_ts = mailbox_ts.reshape(mailbox_ts.shape[0], mailbox_size)
    if updater.w_k.in_features != mailbox.shape[-1] + updater.dim_time:
        raise ValueError('compact Transformer inputs do not match key/value Linear')

    query = updater.w_q(memory).index_select(0, state_to_uid)
    query = query.reshape(state_memory.shape[0], updater.att_h, -1)
    if updater.dim_time > 0:
        selected_mail_ts = mailbox_ts.index_select(0, state_to_uid)
        time_feat = updater.time_enc(
            state_ts[:, None] - selected_mail_ts)
        time_feat = time_feat.reshape(
            state_memory.shape[0], mailbox_size, -1)
        key = _object_state_linear(
            updater.w_k, mailbox, time_feat, state_to_uid)
        value = _object_state_linear(
            updater.w_v, mailbox, time_feat, state_to_uid)
    else:
        key = updater.w_k(mailbox).index_select(0, state_to_uid)
        value = updater.w_v(mailbox).index_select(0, state_to_uid)

    key = key.reshape(state_memory.shape[0], mailbox_size, updater.att_h, -1)
    value = value.reshape(
        state_memory.shape[0], mailbox_size, updater.att_h, -1)
    flat_key = key.flatten(0, 1)
    flat_value = value.flatten(0, 1)
    if (
        updater.att_dropout.p < 1.0
        and triton_segment_attention.supported(query, flat_key, flat_value)
    ):
        result = triton_segment_attention.uniform_attention(
            query, flat_key, flat_value, mailbox_size,
            dropout_p=updater.att_dropout.p,
            training=updater.att_dropout.training,
        )
    else:
        attention = updater.att_act((query[:, None, :, :] * key).sum(dim=3))
        attention = F.softmax(attention, dim=1)
        attention = updater.att_dropout(attention)
        result = (attention[:, :, :, None] * value).sum(dim=1)
    result = result.reshape(state_memory.shape[0], -1)
    result = result + state_memory
    result = updater.layer_norm(result)
    result = updater.mlp(result)
    result = updater.dropout(result)
    return F.relu(result)
