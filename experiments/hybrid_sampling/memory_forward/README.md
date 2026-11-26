# Reusing recently written node memory

The experimental owner-side route assembles the outgoing node-feature, memory, memory timestamp, mail and mail timestamp fields directly in the collective payload using one Triton kernel. The optional forwarding path reads eligible private memory rows from the most recent local write buffer. Both paths use the original write-through state table, mailbox and distributed communication policy. Every consumer, including other ranks, still receives the complete original payload.

The forwarding buffer lasts for at most one eligible fetch. The read checks the canonical table and source tensor version counters and the producing CUDA stream. Historical synchronization may change shared rows, which always read from the canonical table. Duplicate write keys also read from the table. Mail and timestamps always come from their own canonical tables. This experiment targets the existing historical, pipelined, contiguous float32 GPU configuration.

Run from the parent experiment's coordinated training environment with its native CPU sampling and block-packing extensions built. Use a fresh output path for each command. For example, from the parent directory:

```bash
MEMSHARE_PACKED_BLOCK=1 bash training_step/run.sh LASTFM TGN_large /tmp/base 5
MEMSHARE_PACKED_BLOCK=1 MEMSHARE_MEMORY_ROUTE=pack bash training_step/run.sh LASTFM TGN_large /tmp/pack 5
MEMSHARE_PACKED_BLOCK=1 MEMSHARE_MEMORY_ROUTE=forward bash training_step/run.sh LASTFM TGN_large /tmp/forward 5
```

For correctness, set `MEMSHARE_VERIFY_ROUTE` to an existing directory. Every eligible routed payload is compared element for element against the original field gathers. The first batch is fetched by the original synchronous path. Standalone `check_route.py` exercises duplicate writes, historical synchronization, changed source tensors, other writes, consumption and an empty route. Timed runs disable verification.

Four A40s on gpu07 ran two fresh launches per arm with the same partitions, seed, frozen coefficients, fixed-tile sampling and packed DGL blocks. LASTFM used five complete epochs, WikiTalk three. Each launch mean excludes its first epoch. LASTFM launch order was baseline, pack, forward, forward, pack, baseline; WikiTalk swapped pack and forward in that order. Verification launches ran separately. Use `python summarize.py results` to check paired logical demand and batch counts and print the measurements. Hit rates refer to all request rows served on the owner, including requests originating on the same rank. They measure local reads bypassed, without any reduction in network payload.

## Results

Mean steady training seconds per epoch across two launches, with the slowest rank determining each epoch:

| TGN_large dataset | Original fetch | Fused payload | Fused payload + forwarding | Eligible memory rows forwarded in separate verification |
|---|---:|---:|---:|---:|
| LASTFM | 2.0635 | 2.0483 | 2.0636 | 273,988 / 368,866 (74.28%) |
| WikiTalk | 12.1012 | 12.0308 | 12.1211 | 573,546 / 18,953,192 (3.03%) |

The forwarding column took about 0.75% longer than fused packing alone on both datasets. Fusing the payload alone reduced the measured mean by 0.74% and 0.58%, respectively. The small differences and two-launch sample do not establish a robust speedup. The optional experiment remains disabled by default. Both full verification epochs passed exact payload comparisons at all asynchronous fetches on all ranks. The study did not defer owner writes or evaluate time to accuracy.
