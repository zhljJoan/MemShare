# Packing sampled indices before block construction

The training sampler already returns CPU indices. The packed builder reads them together with the CPU distributed-ID maps, assigns temporal states in first-occurrence order, sorts unique distributed node/edge IDs, and returns one contiguous int64 array. The GPU receives the array once and the DGL builder takes views for each field. Training still uses the same root groups, timestamps, edge set, optimizer and communication policy. Positive and negative prediction indices use the original root-to-state mapping.

TGN consumes ordinary blocks. APAN builds reverse blocks from the sampled neighbors. JODIE uses the reverse identity path and samples no neighbors. The extension supports the current one-layer int64 inputs with object and state deduplication enabled. The launcher delegates other configurations to the existing builder, including hierarchical compaction. Timestamps outside the exact float64 integer range also use the existing builder, whose state keys compare timestamps in float64.

## Reproduction

Use the coordinated, metrics-enabled MemShare checkout and native extension documented in the parent experiment. Build the CPU sampling extension in the parent directory, then from this directory run

```bash
bash build.sh
PYTHONPATH="$MEMSHARE_TRAIN_ROOT" python check_blocks.py
bash run.sh LASTFM TGN_large /tmp/block-baseline 3
MEMSHARE_PACKED_BLOCK=1 bash run.sh LASTFM TGN_large /tmp/block-packed 3
```

The correctness check compares sorted object lists, temporal state IDs/timestamps, DGL edges in order, edge indices and time deltas, and prediction metadata. It covers all three block layouts with 0, 1, 37 and 11,000 occurrences. Both arms use fixed-tile split sampling for TGN and APAN. JODIE uses its native identity sampler. Each comparison keeps the original float32 objective and Adam optimizer. The previous-batch feature cache and CUDA Graph replay are off.

The study uses four A40s per run, batch size 3,000 per rank before filtering, one negative per positive, data partition seed 123457, training seed 6773, four-way partitions with topk 0.1, historical memory, and frozen coefficients α=0.3 and β=0.1. Epoch time includes negative sampling, block construction, distributed input fetch, forward/backward, the optimizer and mailbox updates, ending after CUDA synchronization and a barrier. File loading and process initialization precede the clock. Epoch 0 includes any remaining cache calibration, workspace initialization and first-use work. Reporting the slowest rank avoids hiding distributed stalls.

The two large datasets use the journal's complete epoch-0 timing protocol. The small datasets also have two subsequent steady epochs. Only LASTFM/TGN has two independent launches per arm. Comparing the first epoch across all 12 cells uses the same boundary. The independent loss/Adam pilot did not pass the timing gate and is excluded from the matrix. No validation, inference or time-to-accuracy measurements were taken.

The optional large_input.py operates on an isolated training snapshot. It reads the existing per-column NumPy caches of the event CSV, omits a diagnostic distinct-edge count, and reads GDELT's immutable boolean edge feature file by memory mapping before converting the selected rank-local rows to float32. With evaluation disabled it avoids constructing a duplicate evaluation sampler. Both timed arms use the same input loader per dataset. Reruns require the actual data and partitions.
