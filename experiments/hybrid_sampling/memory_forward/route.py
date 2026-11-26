"""Owner-side payload assembly for the pipelined GPU feature route."""
import json
from pathlib import Path

import torch
import torch.distributed as dist
import triton
import triton.language as tl


@triton.jit
def _assemble(ids, feat, mem, mts, mail, ats, keys, order, recent, shared, out,
              nkeys, F: tl.constexpr, M: tl.constexpr, A: tl.constexpr,
              S: tl.constexpr, B: tl.constexpr, FORWARD: tl.constexpr):
    i = tl.program_id(0)
    row = tl.load(ids + i).to(tl.int64)
    j = tl.arange(0, B)
    output = out + i.to(tl.int64) * (F + M + 1 + A + S)
    tl.store(output + j, tl.load(feat + row * F + j, j < F, 0), j < F)
    if FORWARD:
        lo = 0
        hi = nkeys
        while lo < hi:
            mid = (lo + hi) // 2
            if tl.load(keys + mid) < row:
                lo = mid + 1
            else:
                hi = mid
        # Duplicate writes have no deterministic winner in index_put_.
        hit = ((tl.load(keys + lo, lo < nkeys, -1) == row)
               & (tl.load(keys + lo + 1, lo + 1 < nkeys, -1) != row)
               & (tl.load(shared + row) < 0))
        slot = tl.load(order + lo, hit, 0)
        value = tl.load(recent + slot * M + j, hit & (j < M), 0)
        value = tl.where(hit, value, tl.load(mem + row * M + j, ~hit & (j < M), 0))
    else:
        value = tl.load(mem + row * M + j, j < M, 0)
    tl.store(output + F + j, value, j < M)
    tl.store(output + F + M, tl.load(mts + row))
    tl.store(output + F + M + 1 + j, tl.load(mail + row * A + j, j < A, 0), j < A)
    tl.store(output + F + M + 1 + A + j, tl.load(ats + row * S + j, j < S, 0), j < S)


def install(mode, verify_dir=None):
    from starrygl.sample import data_loader
    from starrygl.sample.memory.shared_mailbox import SharedMailBox
    from starrygl.distributed.context import DistributedContext

    if mode not in ("pack", "forward"):
        raise ValueError(mode)
    ordinary = data_loader.get_node_all_to_all_route
    if mode == "forward":
        write = SharedMailBox.set_memory_local
        sync = SharedMailBox.sychronize_shared

        def remember(self, index, source, source_ts, Reduce_Op=None):
            write(self, index, source, source_ts, Reduce_Op)
            self._recent_rows = None
            if (Reduce_Op is None and index.is_cuda and index.numel()
                    and source.dtype == torch.float32 and source.is_contiguous()
                    and self.historical_cache is not None):
                keys, order = index.sort()
                self._recent_rows = (self.node_memory.accessor.data._version,
                                     torch.cuda.current_stream(index.device),
                                     keys, order, source.detach(), source._version)

        def sync_shared(self):
            # This writer only touches shared rows, which never qualify for forwarding.
            buf = getattr(self, "_recent_rows", None)
            table = self.node_memory.accessor.data
            valid = buf is not None and buf[0] == table._version
            sync(self)
            if valid:
                self._recent_rows = (table._version, *buf[1:])

        SharedMailBox.set_memory_local = remember
        SharedMailBox.sychronize_shared = sync_shared

    def route(graph, box, query, out_device=torch.device("cuda")):
        if box is None or graph.nfeat is None or graph.feature_device.type != "cuda":
            return ordinary(graph, box, query, out_device)
        tables = [t.accessor.data for t in (graph.nfeat, box.node_memory,
                  box.node_memory_ts, box.mailbox, box.mailbox_ts)]
        feat, mem, mts, mail, ats = tables
        if (not all(t.device == query.device and t.is_cuda and t.is_contiguous()
                    and t.dtype == torch.float32 for t in tables)
                or torch.device(out_device) not in (query.device, torch.device("cuda"))
                or (feat.ndim, mem.ndim, mts.ndim, mail.ndim, ats.ndim) != (2, 2, 1, 3, 2)):
            return ordinary(graph, box, query, out_device)
        group = DistributedContext.get_default_context().memory_nccl_group
        ind = graph.nfeat.all_to_all_ind2ptr(query, group=group)
        ids = ind["recv_ind"]
        f, m, a, s = feat.size(1), mem.size(1), mail.size(1)*mail.size(2), ats.size(1)
        out = mem.new_empty((ids.numel(), f + m + 1 + a + s))
        buf = getattr(box, "_recent_rows", None)
        valid = (buf is not None and buf[0] == mem._version
                 and buf[1] == torch.cuda.current_stream(mem.device) and buf[4]._version == buf[5])
        keys, order, source = buf[2:5] if valid else (ids, ids, mem)
        if ids.numel():
            _assemble[(ids.numel(),)](ids, *tables, keys, order, source,
                box.is_shared_mask if valid else ids, out, keys.numel(), f, m, a, s,
                triton.next_power_of_2(max(f, m, a, s)), valid)
        if verify_dir:
            expected = torch.cat([t.index_select(0, ids).flatten(1) if t.ndim > 1
                                  else t[ids, None] for t in tables], dim=1)
            assert torch.equal(out, expected), "memory payload mismatch"
            hits = 0
            if valid:
                pos = torch.searchsorted(keys, ids)
                first = pos.clamp_max(keys.numel()-1)
                same = (pos < keys.numel()) & (keys[first] == ids)
                duplicate = ((pos+1 < keys.numel()) &
                             (keys[(pos+1).clamp_max(keys.numel()-1)] == ids))
                hits = (same & ~duplicate & (box.is_shared_mask[ids] < 0)).sum().item()
            with (Path(verify_dir) / f"rank-{dist.get_rank()}.jsonl").open("a") as file:
                print(json.dumps(dict(rows=ids.numel(), hits=hits, memory_width=m)), file=file)
        box._recent_rows = None
        return ind, out

    data_loader.get_node_all_to_all_route = route
