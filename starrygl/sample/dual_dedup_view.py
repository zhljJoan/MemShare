import os
import threading


_KEYS = (
    "calls",
    "state_rows",
    "object_node_rows",
    "edge_occurrence_rows",
    "object_edge_rows",
    "memory_object_rows",
    "mailbox_object_rows",
)
_LOCK = threading.Lock()
_STATS = {key: 0 for key in _KEYS}


def enabled():
    return os.environ.get("MEMSHARE_DUAL_DEDUP_VIEW", "off").lower() in {
        "1",
        "true",
        "on",
        "yes",
    }


def reset():
    with _LOCK:
        for key in _KEYS:
            _STATS[key] = 0


def stats():
    with _LOCK:
        result = dict(_STATS)
    result["enabled"] = enabled()
    result["state_per_object_node"] = _ratio(
        result["state_rows"], result["object_node_rows"]
    )
    result["edge_occurrence_per_object_edge"] = _ratio(
        result["edge_occurrence_rows"], result["object_edge_rows"]
    )
    return result


def attach(block, node_feat, edge_feat, mem_embedding):
    state_to_uid = block.srcdata["__ID"]
    view = {
        "object_view": {"node_feat": node_feat},
        "state_view": {"state_to_uid": state_to_uid},
        "edge_view": {},
        "reuse_stats": {
            "state_rows": int(state_to_uid.numel()),
            "object_node_rows": int(node_feat.shape[0])
            if node_feat is not None
            else 0,
            "edge_occurrence_rows": 0,
            "object_edge_rows": 0,
            "memory_object_rows": 0,
            "mailbox_object_rows": 0,
        },
    }
    if "__ID" in block.edata:
        edge_to_eid = block.edata["__ID"]
        view["edge_view"] = {
            "edge_to_eid": edge_to_eid,
            "edge_feat": edge_feat,
        }
        view["reuse_stats"]["edge_occurrence_rows"] = int(edge_to_eid.numel())
        view["reuse_stats"]["object_edge_rows"] = (
            int(edge_feat.shape[0]) if edge_feat is not None else 0
        )
    if mem_embedding is not None:
        memory, memory_ts, mailbox, mailbox_ts = mem_embedding
        object_view = view["object_view"]
        object_view["memory"] = memory
        object_view["memory_ts"] = memory_ts
        object_view["mailbox"] = mailbox.reshape(mailbox.shape[0], -1)
        object_view["mailbox_ts"] = mailbox_ts.reshape(mailbox_ts.shape[0], -1)
        view["reuse_stats"]["memory_object_rows"] = int(memory.shape[0])
        view["reuse_stats"]["mailbox_object_rows"] = int(mailbox.shape[0])
    block._memshare_dual_dedup_view = view
    _record(view["reuse_stats"])
    return view


def _ratio(num, den):
    return float(num) / float(den) if den else 0.0


def _record(values):
    with _LOCK:
        _STATS["calls"] += 1
        for key, value in values.items():
            _STATS[key] += int(value)
