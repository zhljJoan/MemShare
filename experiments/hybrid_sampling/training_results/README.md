# Four-GPU training and inter-batch reuse

The two factors are sampling (native or 30% hybrid) and caching (off or previous-batch static edge features). The training entry point is the current coordinated-sampling MemShare checkout. The experiment uses ordinary blocks with batch-local block/state deduplication. Hybrid mode calibrates on the first five actual training batches and builds a separate CSR and device cache within epoch zero. The adapter converts its output back to the five fields consumed by block construction.

Each run trains LASTFM / TGN_large for three complete epochs on four A40s, including negative sampling, distributed feature fetch, mailbox updates, and forward/backward/optimizer work. The loader uses its usual generated 172-dimensional features for this dataset. Epoch timing includes the final CUDA synchronization and rank barrier. Startup before epoch zero, evaluation and checkpoints are outside the training timer. Each rank executes 76 batches per epoch. Times use the slowest rank. The steady estimate averages epochs 1 and 2 within each independent launch, then averages the two launch means. The timing order without caching was native, hybrid, hybrid, native. With caching it was native, hybrid, hybrid, native. The two groups ran successively, so small between-group differences can include machine drift.

The cache issues collectives for the ordered misses and reconstructs the full feature tensor before downstream consumers. It retains one completed batch per rank, using immutable distributed edge IDs as keys. The ordinary sampler emits sorted feature IDs. Hierarchical compaction can use the cache's separately sorted index, but the timing study only covers ordinary blocks. Logical demand counts in the training metrics are recorded before cache lookup.

The independent sampler trace (batches 5 through 29 per rank, including negatives) checks root indices, edge IDs, sampled nodes, timestamps and time deltas against the native sampler. Reuse is the fraction of distinct keys in the current batch that also appeared in the given number of preceding batches, pooled by request count and excluding windows with insufficient history. Edge IDs and root IDs identify immutable features and adjacency rows respectively. An exact root and timestamp pair still requires matching sampling context to reuse sampled output.

The separate cache verification run fetches full reference features and checks exact equality for all four ranks at each asynchronous handoff. Its timing is excluded. The four TSV columns record requested rows, hit rows, requested remote rows and hit remote rows. All-rank totals exclude the first synchronous fetch of each epoch. These counters measure skipped feature requests. They do not measure link traffic or protocol overhead.

## Results

| Sampler | Edge-feature cache | Steady training s/epoch | First epoch s, two launches |
|---|---|---:|---:|
| Native CPU | Off | 2.065 | 3.104 / 2.768 |
| 30% hybrid | Off | 2.106 | 4.071 / 3.882 |
| Native CPU | Previous batch | 2.107 | 3.066 / 2.759 |
| 30% hybrid | Previous batch | 2.082 | 3.836 / 3.949 |

Native CPU with caching off had the shortest mean. Hybrid with caching consumed 0.8% more steady training time than that baseline. All modes had identical logical remote node and edge demand for each corresponding epoch. These two-launch measurements do not isolate the relative cost of lookup, materialization, communication, and GPU contention.

In the sampler trace, 73.1% of distinct root IDs and 47.1% of distinct sampled edge IDs appeared in the previous batch. Over the preceding four batches the corresponding fractions were 87.8% and 53.4%. Exact root/time pairs had zero overlap in both windows. In the independent three-epoch cache verification, 6,066,191 of 12,789,326 edge rows were served from the previous batch. Only 278,561 of 2,776,317 remote requests hit the cache.

## Reproduce

Use the coordinated training checkout containing the built sampler, the model and the metrics-enabled training script. Set MEMSHARE_TRAIN_ROOT to its absolute path. Activate the same PyTorch/Triton environment as the sampling replay and build the experimental CPU kernel with bash build.sh. From the experiment directory, use a fresh output directory for each call.

```bash
bash train_compare.sh native /tmp/native-run
bash train_compare.sh hybrid /tmp/hybrid-run
MEMSHARE_PREVIOUS_EDGE_FEATURES=1 bash train_compare.sh native /tmp/native-cache-run
MEMSHARE_PREVIOUS_EDGE_FEATURES=1 bash train_compare.sh hybrid /tmp/hybrid-cache-run
MEMSHARE_REUSE_TRACE=/tmp/hybrid-trace bash train_compare.sh hybrid /tmp/hybrid-trace
MEMSHARE_PREVIOUS_EDGE_FEATURES=1 MEMSHARE_VERIFY_EDGE_CACHE=1 \
    MEMSHARE_CACHE_TRACE_DIR=/tmp/cache-trace bash train_compare.sh hybrid /tmp/cache-trace
python summarize_training.py training_results
```

The runner overlays the loader in an output-local copy when feature caching is enabled. The two trace runs intentionally add reference computation and logging. Files nc1/nc2 contain native sampling with caching, hc1/hc2 contain hybrid sampling with caching.
