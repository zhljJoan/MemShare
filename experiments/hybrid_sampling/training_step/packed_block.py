"""Assemble one-layer training blocks from one ordered CPU index packet."""

import torch
import numpy as np
import dgl
from memshare_packed_block import compact
from starrygl.sample import batch_data


def install():
    ordinary, reversed_block = batch_data.to_block, batch_data.to_reversed_block

    def build(
        graph,
        data,
        sample_out,
        device,
        unique=True,
        state_unique=None,
        identity=False,
        reverse=False,
    ):
        fallback = reversed_block if reverse else ordinary
        ret, meta = sample_out if len(sample_out) == 2 else (None, None)
        eligible = (
            unique
            and state_unique is not False
            and isinstance(meta, dict)
            and "_hierarchical_compaction" not in meta
            and "seed" in meta
            and "seed_ts" in meta
            and all(
                meta[k].device.type == "cpu" and meta[k].dtype == torch.int64
                for k in ("seed", "seed_ts")
            )
            and graph.nids_mapper.device.type == graph.eids_mapper.device.type == "cpu"
            and (identity or (isinstance(ret, (tuple, list)) and len(ret) == 1))
        )

        def original():
            kw = dict(identity=identity) if reverse else {}
            return fallback(
                graph,
                data,
                sample_out,
                device,
                unique=unique,
                state_unique=state_unique,
                **kw,
            )

        if not eligible:
            return original()
        seed = meta["seed"]
        ts = meta["seed_ts"]
        if identity:
            fields = [np.empty(0, dtype=np.int64)] * 4
        else:
            entry = ret[0]
            fields = [
                entry.sample_nodes().numpy(),
                entry.sample_nodes_ts().numpy(),
                entry.eid().numpy(),
                entry.src_index().numpy(),
            ]
        packet = compact(
            seed.numpy(),
            ts.numpy(),
            *fields,
            graph.nids_mapper.numpy(),
            graph.eids_mapper.numpy(),
            reverse,
            torch.distributed.get_rank(),
        )
        if packet is None:
            return original()
        del meta["seed"], meta["seed_ts"]
        packed, sizes, num_dst, local = packet
        ids, eids, uid, stamp, roots, row, col, edge_ref, delta = (
            torch.from_numpy(packed).to(device).split(sizes)
        )
        for key, value in meta.items():
            if isinstance(value, torch.Tensor):
                meta[key] = roots[value.to(device)]
        if reverse:
            batch_data.CountComm.origin_local = local
            batch_data.CountComm.origin_remote = len(seed) - local
            block = dgl.create_block(
                (row, torch.arange(len(col), device=device)),
                num_src_nodes=len(uid),
                num_dst_nodes=len(col),
            )
            block.dstdata["ID"] = col
        else:
            block = dgl.create_block(
                (row, col), num_src_nodes=len(uid), num_dst_nodes=num_dst
            )
            block._memshare_raw_node_rows = len(seed) + len(row)
            block.edata["__ID"] = edge_ref
            if len(delta):
                block.edata["dt"] = delta
        block.srcdata["__ID"] = uid
        block.srcdata["ts"] = stamp.float()
        return (data, [[block]], meta), ids, eids

    batch_data.to_block = build

    def reverse(*args, **kwargs):
        return build(*args, **kwargs, reverse=True)

    batch_data.to_reversed_block = reverse
