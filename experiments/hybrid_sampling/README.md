# CPU/GPU split temporal sampling

A single-layer sampler for immutable, timestamp-sorted CSR graphs. The caller supplies a partition-relative remote flag for each adjacency entry and the GPU-resident node IDs. Each original root occurrence runs on exactly one processor. Both paths return original CSR positions in per-root order, padded with -1. The coordinated policy keys its random draw on neighbor ID, original query group, seed, batch and layer (zero here). It preserves strict timestamp exclusion and the existing boundary sampler's maximum of fanout + 1 accepted edges.

GNNFlow describes GPU traversal of a temporal block index and supports host-resident graph data. This experiment adds CPU execution of cache misses. Its GPU cache holds whole adjacency rows, including older local edges that may be needed after boundary rejection. CPU routing translates node IDs into compact GPU row numbers. A per-row offset lets the kernel emit original CSR positions directly. [GNNFlow sampling source](https://github.com/jasperzhong/GNNFlow/blob/ea525b3d9db13d85c0bde1727a7ab306d7125c71/gnnflow/csrc/sampling_kernels.cu)

## Reproduce

Use Python with NumPy, pandas, PyTorch (including its bundled pybind11 headers), Triton, an OpenMP C++ compiler, and a CUDA device. These runs used Triton 3.1.0. Run the following from this directory.

```bash
bash build.sh
# Requires an installed MemShare native extension with configure_coordinated_sampling.
python check_native.py
OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 taskset -c 0-9 \
    python benchmark.py /path/to/LASTFM/edges.csv /path/to/LASTFM_4parts_top0.1
```

The benchmark starts halfway through rank 0's training edges, using seed-123457 four-way partitions. Both endpoints of 3,000 training edges supply 6,000 queries per batch. Five calibration batches rank rows by observed query count divided by adjacency length. The benchmark fills the selected fraction of total adjacency entries in that order. The next 21 batches supply the correctness and timing workload. The fully resident case includes every row. Fanout is 20, boundary probability 0.1, group size 8, and seed 6773. Shared nodes are treated as local on rank 0. Each mode gets 25 warmups and 105 timed calls. The three numbered files per dataset are separate process runs.

Wall time includes routing, launch, host/device copies and ordered host output. CPU core-ms is process CPU time across threads. GPU event time covers the sampling kernel launch on its stream. Cache MiB counts persistent GPU graph tensors only. Setup, cold compilation, negatives, block construction, distributed compaction, and model execution are outside this measurement. The timed CPU baseline is the OpenMP CSR implementation in this directory. The native MemShare sampler is a separate correctness reference.

## Three-run medians

| Dataset | Cached adjacency budget | Wall ms | CPU core-ms | GPU event ms | GPU graph MiB | GPU queries |
|---|---:|---:|---:|---:|---:|---:|
| LASTFM | 0% | 1.114 | 4.818 | 0.000 | 0.00 | 0 |
| LASTFM | 10% | 1.082 | 4.197 | 0.080 | 4.17 | 1986 |
| LASTFM | 30% | 0.837 | 2.727 | 0.092 | 12.01 | 3839 |
| LASTFM | 100% | 0.407 | 0.409 | 0.115 | 41.97 | 6000 |
| WikiTalk | 0% | 1.007 | 4.231 | 0.000 | 0.00 | 0 |
| WikiTalk | 10% | 1.132 | 2.991 | 0.083 | 25.53 | 2371 |
| WikiTalk | 30% | 0.777 | 1.873 | 0.099 | 76.32 | 3537 |
| WikiTalk | 100% | 0.425 | 0.427 | 0.122 | 280.08 | 6000 |

The [four-GPU training experiment](training_results/README.md) measures the sampler in the existing pipeline and evaluates a one-batch static edge-feature cache.

The [host submission experiment](dispatch_results/README.md) compares the original launch path to a precompiled fixed-tile sampler and to CUDA Graph replay.
