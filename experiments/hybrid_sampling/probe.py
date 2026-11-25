"""Isolated one-layer CPU/GPU query split; never imported by production training."""

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice
import memshare_hybrid_cpu


@triton.jit
def _mix(x):
    x = x.to(tl.uint64) + 0x9E3779B97F4A7C15
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB
    return x ^ (x >> 31)


@triton.jit
def _sample(
    starts,
    ends,
    bias,
    nbr,
    ts,
    remote,
    roots,
    times,
    ids,
    out,
    F: tl.constexpr,
    COORD: tl.constexpr,
    BETA: tl.constexpr,
    GROUP: tl.constexpr,
    SEED: tl.constexpr,
    BATCH,
    B: tl.constexpr,
):
    i = tl.program_id(0)
    node = tl.load(roots + i)
    query_time = tl.load(times + i)
    begin = tl.load(starts + node)
    end = tl.load(ends + node)
    offset = tl.load(bias + node)
    low = begin
    high = end
    while low < high:
        mid = (low + high) // 2
        v = tl.load(ts + mid)
        low = tl.where(v < query_time, mid + 1, low)
        high = tl.where(v < query_time, high, mid)
    end = low
    lane = tl.arange(0, B)
    tl.store(out + i * (F + 1) + lane, -1, mask=lane < F + 1)
    if not COORD:
        first = tl.maximum(begin, end - F)
        pos = first + lane
        tl.store(out + i * (F + 1) + lane, pos + offset, mask=pos < end)
    else:
        last_time = tl.load(ts + end - 1, mask=end > begin, other=0)
        delta = ((query_time - last_time) * F).to(tl.float64)
        pos = end - 1 - lane
        valid = (lane < F) & (pos >= begin)
        r = tl.load(remote + pos, mask=valid, other=0) != 0
        t = tl.load(ts + pos, mask=valid, other=0)
        weight = tl.where(
            r, libdevice.exp((t - query_time).to(tl.float64) / delta), 0.0
        )
        total = tl.sum(weight, 0)
        total = tl.where(total < 1.0e-6, 1.0, total)
        rem = tl.sum(r.to(tl.int32), 0)
        obj = tl.load(nbr + pos, mask=valid, other=0).to(tl.uint64)
        group = (tl.load(ids + i) // GROUP).to(tl.uint64)
        key = (
            SEED
            ^ _mix(obj)
            ^ _mix(group + 0xD1B54A32D192ED03)
            ^ _mix(tl.cast(BATCH, tl.uint64) + 0x94D049BB133111EB)
            ^ _mix(tl.full((), 0xBF58476D1CE4E5B9, tl.uint64))
        )
        uniform = (_mix(key) >> 11).to(tl.float64) / 9007199254740992.0
        accepted = valid & ((~r) | (uniform <= BETA * weight / total * rem))
        order = tl.cumsum(accepted.to(tl.int32), 0) - 1
        tl.store(out + i * (F + 1) + order, pos + offset, mask=accepted & (order <= F))
        count = tl.sum(accepted.to(tl.int32), 0)
        cursor = end - F - 1
        while (count <= F) & (cursor >= begin):
            pos = cursor - lane
            valid = pos >= begin
            r = tl.load(remote + pos, mask=valid, other=1) != 0
            accepted = valid & (r == 0)
            order = count + tl.cumsum(accepted.to(tl.int32), 0) - 1
            tl.store(
                out + i * (F + 1) + order, pos + offset, mask=accepted & (order <= F)
            )
            count += tl.sum(accepted.to(tl.int32), 0)
            cursor -= B


class SplitSampler:
    def __init__(
        self, ptr, nbr, ts, remote, gpu_nodes, fanout=20, threads=10, coordinated=True
    ):
        self.cpu = memshare_hybrid_cpu
        self.graph = tuple(
            np.ascontiguousarray(x, dtype=np.int64) for x in (ptr, nbr, ts, remote)
        )
        ptr, nbr, ts, remote = self.graph
        if (
            ptr.ndim != 1
            or not ptr.size
            or ptr[0] != 0
            or ptr[-1] != len(ts)
            or np.any(np.diff(ptr) < 0)
        ):
            raise ValueError("invalid CSR")
        if ts.ndim != 1 or any(x.shape != ts.shape for x in (nbr, remote)):
            raise ValueError("invalid edge arrays")
        if np.any((nbr < 0) | (nbr >= len(ptr) - 1)) or np.any(
            (remote != 0) & (remote != 1)
        ):
            raise ValueError("invalid neighbor or remote flag")
        descents = np.flatnonzero(ts[1:] < ts[:-1]) + 1
        if np.setdiff1d(descents, ptr, assume_unique=False).size:
            raise ValueError("timestamps must be sorted within each row")
        if fanout < 1 or threads < 1:
            raise ValueError("invalid fanout/threads")
        gpu_nodes = np.unique(np.asarray(gpu_nodes, dtype=np.int64))
        if np.any(gpu_nodes < 0) or np.any(gpu_nodes >= len(ptr) - 1):
            raise ValueError("invalid cached nodes")
        self.slot = np.full(len(ptr) - 1, -1, dtype=np.int64)
        self.slot[gpu_nodes] = np.arange(len(gpu_nodes))
        # Only cached rows need device metadata. A constant offset maps each
        # sampled position back to its original CSR position on the device.
        lengths = np.diff(ptr)[gpu_nodes]
        ends = np.cumsum(lengths)
        starts = ends - lengths
        bias = ptr[gpu_nodes] - starts
        index = np.repeat(bias, lengths) + np.arange(lengths.sum())
        self.gpu = (
            tuple(
                torch.from_numpy(x).cuda()
                for x in (
                    starts,
                    ends,
                    bias,
                    nbr[index],
                    ts[index],
                    remote[index].astype(np.uint8),
                )
            )
            if len(gpu_nodes)
            else None
        )
        if self.gpu:
            torch.cuda.current_stream().synchronize()
        self.fanout = fanout
        self.threads = threads
        self.coordinated = coordinated
        self.stream = torch.cuda.Stream()
        self.cache_bytes = (
            sum(x.numel() * x.element_size() for x in self.gpu) if self.gpu else 0
        )

    def sample(self, roots, times, batch=0, beta=0.1, group=8, seed=6773):
        roots = np.ascontiguousarray(roots, dtype=np.int64)
        times = np.ascontiguousarray(times, dtype=np.int64)
        if roots.ndim != 1 or times.shape != roots.shape:
            raise ValueError("query shape mismatch")
        if np.any(roots < 0) or np.any(roots >= len(self.slot)):
            raise ValueError("invalid query root")
        if (
            not np.isfinite(beta)
            or not 0 <= beta <= 1
            or group < 1
            or batch < 0
            or seed < 0
        ):
            raise ValueError("invalid sampling controls")
        slots = self.slot[roots]
        on_gpu = slots >= 0
        gid = np.flatnonzero(on_gpu)
        cid = np.flatnonzero(~on_gpu)
        result = np.full((len(roots), self.fanout + 1), -1, dtype=np.int64)
        if not len(gid):
            self.cpu.sample(
                *self.graph,
                roots,
                times,
                cid,
                result,
                self.fanout,
                self.threads,
                self.coordinated,
                beta,
                group,
                seed,
                batch,
            )
            return result, dict(gpu_queries=0, cpu_queries=len(cid), gpu_kernel_ms=0.0)
        ready = None
        event_start = None
        event_end = None
        # GPU launch first; native CPU sampling releases the GIL and overlaps it.
        if len(gid):
            with torch.cuda.stream(self.stream):
                gq = torch.from_numpy(np.ascontiguousarray(slots[gid])).to(
                    "cuda", non_blocking=True
                )
                gt = torch.from_numpy(np.ascontiguousarray(times[gid])).to(
                    "cuda", non_blocking=True
                )
                gi = torch.from_numpy(gid).to("cuda", non_blocking=True)
                out = torch.empty(
                    (len(gid), self.fanout + 1), dtype=torch.int64, device="cuda"
                )
                host = torch.empty_like(out, device="cpu", pin_memory=True)
                event_start = torch.cuda.Event(enable_timing=True)
                event_end = torch.cuda.Event(enable_timing=True)
                event_start.record()
                _sample[(len(gid),)](
                    *self.gpu,
                    gq,
                    gt,
                    gi,
                    out,
                    self.fanout,
                    self.coordinated,
                    beta,
                    group,
                    seed,
                    batch,
                    triton.next_power_of_2(max(32, self.fanout + 1)),
                    num_warps=1,
                )
                event_end.record()
                host.copy_(out, non_blocking=True)
                ready = torch.cuda.Event()
                ready.record()
        cpu_out = np.empty((len(cid), self.fanout + 1), dtype=np.int64)
        if len(cid):
            self.cpu.sample(
                *self.graph,
                np.ascontiguousarray(roots[cid]),
                np.ascontiguousarray(times[cid]),
                cid,
                cpu_out,
                self.fanout,
                self.threads,
                self.coordinated,
                beta,
                group,
                seed,
                batch,
            )
        result[cid] = cpu_out
        gpu_ms = 0.0
        if ready:
            ready.synchronize()
            if len(cid):
                result[gid] = host.numpy()
            else:
                result = host.numpy()
            gpu_ms = event_start.elapsed_time(event_end)
        return result, dict(
            gpu_queries=len(gid), cpu_queries=len(cid), gpu_kernel_ms=gpu_ms
        )
