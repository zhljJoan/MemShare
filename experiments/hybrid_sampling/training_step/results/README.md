# Paired four-GPU training times

Each entry measures a complete training epoch in seconds on four A40 GPUs. The time is the maximum across the four ranks. The baseline and packed arms of each cell ran on the same host. Epoch 0 includes first-use work inside the training loop. The steady columns average epochs 1 and 2. The two LASTFM/TGN launches per arm are averaged. Each other cell has one launch per arm.

| Dataset | Model | Host | Batches/rank | Epoch 0 baseline → packed (s) | Time saved | Steady baseline → packed (s) | Time saved |
|---|---|---|---:|---:|---:|---:|---:|
| LASTFM | TGN | gpu07 | 76 | 3.387 → 3.158 | +6.76% | 2.133 → 2.047 | +4.04% |
| LASTFM | JODIE | gpu07 | 76 | 2.909 → 2.799 | +3.79% | 0.965 → 0.891 | +7.59% |
| LASTFM | APAN | gpu07 | 76 | 3.325 → 2.549 | +23.33% | 1.440 → 1.370 | +4.90% |
| WikiTalk | TGN | gpu07 | 457 | 16.205 → 15.615 | +3.64% | 12.581 → 12.082 | +3.97% |
| WikiTalk | JODIE | gpu07 | 457 | 7.778 → 6.864 | +11.75% | 6.015 → 5.416 | +9.97% |
| WikiTalk | APAN | gpu07 | 457 | 15.310 → 14.842 | +3.05% | 11.403 → 11.105 | +2.61% |
| StackOverflow | TGN | gpu07 | 3704 | 157.400 → 153.386 | +2.55% | — | — |
| StackOverflow | JODIE | gpu04 | 3704 | 51.995 → 47.809 | +8.05% | — | — |
| StackOverflow | APAN | gpu04 | 3704 | 134.468 → 128.394 | +4.52% | — | — |
| GDELT | TGN | gpu06 | 10714 | 607.948 → 605.175 | +0.46% | — | — |
| GDELT | JODIE | gpu01 | 10714 | 191.227 → 183.872 | +3.85% | — | — |
| GDELT | APAN | gpu04 | 10714 | 458.557 → 349.018 | +23.89% | — | — |

Equal-weight geometric mean of the twelve epoch-0 speedups is 1.091×. Positive percentages denote shorter training time. Use the paired rows when comparing individual workloads.

## Interpretation

The paired change packs index and timestamp fields on the CPU before DGL block construction. Sampler selection, deduplication settings, optimizer, coefficients and batch ordering are fixed within each pair. The summarizer verifies complete epochs on four ranks, equal per-rank batch counts and equal logical remote node and edge demand across arms, plus finite losses. The comparison measures this increment over the experimental sampling baseline. The absolute numbers use a different configuration from the paper’s main comparison. [Per-rank sampler setup times](sampler_setup.csv) show first-use variability. For StackOverflow/TGN the slowest sampler setup fell by 3.54 s while the full epoch fell by 4.01 s, so attributing the entire first-epoch difference to the block builder would overstate the evidence. On GDELT/APAN, peak sampler setup alone changed from 206.78 s to 97.19 s. The cold-epoch difference there is especially sensitive to initialization.

An independent CUDA runtime trace on gpu07 counted launches for 20 warmed constructions of each synthetic block, reporting counts per call. The forward and reverse fixtures contain 120 roots with 11,000 sampled occurrences. The identity case uses the same roots without neighbors. See [CUDA API counts](block_cuda_api.json). Mean host time inside cudaLaunchKernel per build was 476.3 → 43.8 µs (forward), 364.4 → 48.3 µs (reverse), and 284.6 → 32.5 µs (identity). These instrumented values isolate runtime API submission inside the builder. Full-epoch timings above were collected without tracing.

GDELT uses the actual 413-dimensional node and 182-dimensional edge feature files. The manuscript dataset summary lists 186 edge features. For the other datasets these runs use the training loader’s generated 172-dimensional features. The large graph measurements cover one full epoch per arm. Validation and test evaluation were disabled. There are no measured effects on time to target accuracy or a separate gain attributable to cross-batch caching. The [earlier LASTFM/TGN trace](../../training_results/README.md) reports reuse and remote feature-cache hits separately.

[Machine-readable table](matrix.csv) · [Run provenance](provenance.json) · [Paired raw records](matrix/) · [LASTFM/TGN pilot records](pilot/)

## StackOverflow/TGN steady confirmation

A second independent pair on gpu07 ran two complete epochs in reverse launch order (packed first, then baseline). After both samplers had initialized, epoch 1 took 123.563 s with the baseline and 123.354 s with packing (+0.17% time saved). Each rank processed 3,704 batches and both arms had equal logical remote demand in each epoch. This confirmatory run is reported separately from the twelve epoch-0 comparisons. [Raw steady confirmation](steady_check/StackOverflow.TGN/)

## GDELT/APAN steady confirmation

A second independent pair on gpu06 ran two complete epochs in reverse launch order (packed first, then baseline). After both samplers had initialized, epoch 1 took 241.823 s with the baseline and 240.015 s with packing (+0.75% time saved). Each rank processed 10,714 batches and both arms had equal logical remote demand in each epoch. This confirmatory run is reported separately from the twelve epoch-0 comparisons. [Raw steady confirmation](steady_check/GDELT.APAN/)
