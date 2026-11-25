# CPU cost of submitting temporal sampling to the GPU

The same coordinated GPU sampling policy is exercised through three host implementations. The existing hybrid path sends three query vectors, dispatches through Triton's JIT wrapper, and returns sampled CSR indices to the CPU. The fixed path packs query slots, timestamps, original occurrence IDs and batch controls in one pinned buffer. It copies the active input and output prefixes and calls a precompiled Triton launcher. Its query grid rounds up to 256 programs. A guard excludes padded queries.

The graph path records the input copy, precompiled sampler and output copy at stable addresses. A replay submits all three operations together. It copies its full power-of-two workspace capacity. All modes return independent results to the caller. The graph retains scratch space and execution commands across batches. Each batch still uses its own timestamps, probabilities, group IDs, random key and sampled edges.

Compilation and CUDA module loading take place during training sampler initialization on each rank. The signature depends on fanout, sampling policy, device and tensor types. Batch number, effective count, seed, group size and probability are runtime data. Calibration and CSR construction still happen during epoch zero. A larger query batch grows the workspace. PyTorch 2.1 graph capture interacts with its CUDA random generator. In training, captures occur on the main thread at a synchronous sampling call. Background sampling uses the direct launcher until a graph is available. The sampler explicitly selects the training rank's device on its worker thread.

## Results

| Path | Replay CPU submit µs | Replay dispatch call µs | GPU sampler µs | Training CPU submit µs | Training dispatch µs | Steady s/epoch |
|---|---:|---:|---:|---:|---:|---:|
| hybrid | 239.8 | 57.2 | 43.0 | 418.3 | 106.0 | 2.1596 |
| fixed | 181.3 | 29.1 | 43.0 | 341.5 | 78.7 | 2.1055 |
| graph | 155.8 | 51.4 | 43.0 | 290.1 | 70.5 | 2.9540 |

Two additional native CPU sampling launches measured 2.1035 and 2.1116 s/epoch (2.1075 mean). These baselines ran after the three-mode comparison.

Replay values are medians of the three independent process medians. Training submission values are medians of the four rank medians from separate diagnostic runs. With the direct path, profiler traces show one driver kernel launch and two explicit asynchronous transfers per batch, down from four transfers. The cold direct compilation and module load took 312–314 ms in the isolated replay. Precompilation in the training processes is reported separately from epoch times in each result file.

The graph path gave the shortest isolated submission time and the longest training epochs. Across both timing runs, ranks 0 and 3 used replay in 152 steady batches each. Ranks 1 and 2 used replay in 79 and 78 batches respectively, with direct launches while waiting for a safe recapture after growth. The graph path transfers padded capacity. Profiling also recorded two extra runtime kernel launches per replay. PyTorch 2.1.1 refills the graph RNG seed and offset before launching the graph. These observations alone do not attribute the epoch slowdown. [PyTorch CUDA Graph implementation](https://github.com/pytorch/pytorch/blob/v2.1.1/aten/src/ATen/cuda/CUDAGraph.cpp#L219-L236)

The real-training correctness run compared original root indices, edge IDs, neighbor IDs, timestamps, and deltas to the native sampler in all 223 batches after calibration per rank. Logical remote node and edge demand matched across corresponding timed epochs. This experiment uses one dataset, one seed and two timed launches per mode. It does not establish time-to-accuracy.

## Measurement

- **Host submission** uses a CPU wall clock around staging the GPU queries, queuing transfers, issuing the kernel or replay, and recording completion. It excludes initial routing, the CPU sampler, waiting for the result, and block construction.
- **Dispatch call** times the Triton JIT call, the precompiled launch call or CUDA Graph replay, respectively. Replay includes submission of both transfers. These are host wall times and can include driver stalls.
- **GPU kernel execution** uses CUDA activity timestamps from a separate profiler pass. The timing loops run with the profiler disabled.
- **Training** uses LASTFM / TGN_large, four A40s, ordinary blocks, frozen boundary probability and a 30% hybrid adjacency budget. Each launch runs three complete training epochs. Each steady number averages the slowest rank's times for epochs 1 and 2. Edge-feature caching is off. First epoch and precompile times are reported separately. Model initialization, evaluation and checkpointing are outside the epoch timer.
- **Training-thread diagnostics** use a separate run per mode, with a CPU clock around the actual loader's GPU submission. Each rank reports a median across its last two epochs. Trace collection is excluded from the training time comparison.

From the experiment directory, build the native sampler using the parent README, then run the following in the coordinated training environment.

```bash
python check_native.py
OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 taskset -c 0-9 \
  python dispatch_benchmark.py /path/to/LASTFM/edges.csv /path/to/LASTFM_4parts_top0.1
bash train_compare.sh fixed /tmp/fixed-run
bash train_compare.sh graph /tmp/graph-run
MEMSHARE_PROFILE_SUBMISSION=/tmp/submission-run bash train_compare.sh graph /tmp/submission-run
```

Each isolated replay process tests hybrid, direct and graph in that order. A new Triton disk cache per process gives a cold signature for direct mode. Graph mode shares that signature within the process. Each mode checks the same 21 batches, warms up for a further 21 iterations and times 105 calls. GPU durations and host CUDA API counts are from 12 additional calls. Training uses the order hybrid, fixed, graph, graph, fixed, hybrid.
