# Direction 06 — On-Prem Deployment Profile (M4 GPU & CPU)

**Question:** Which model is the better *deployable* choice on-prem, and at what cost?
**Maps to:** RQ5 · Experiment **E7** · Implementation: appendix Phase 5 (Task 5.1).
**Why it matters for the choice:** The assignment requires the chosen model to run **fully on-prem in an offline Docker container**. Accuracy is necessary but not sufficient: image/RAM/latency/energy decide customer feasibility. Critically, the shipped image is **CPU-only**, so CPU is the number customers actually live with.

## Device strategy (explicit)
- **M4 GPU = MPS** (`--device mps`, your 10-core GPU). **M4 CPU = `--device cpu`.**
- **Step 0:** verify `torch.backends.mps.is_available()` on the host (your earlier run logged it as false) and enable MPS; if it cannot be enabled, document why and proceed CPU-only.
- **Measure BOTH `cpu` and `mps`.**
- **Deployment relevance:** the offline Docker image installs **CPU-only torch** (to keep the image small) → **CPU latency/throughput/RSS are the customer-facing numbers**; MPS is the local-dev/accelerated ceiling. Every figure is labeled with its device, and the recommendation is made on **CPU** numbers with MPS shown as upside.

## Setup
- Models (frozen): both.
- **Synthetic workload** (not the labeled set): inputs at fixed token lengths {16,32,64,128,256} × batch {1,8,32}, sampled from DNRTI tokens, with warmup runs discarded. This isolates operational cost from accuracy and needs no gold labels.
- Metrics: latency p50/p95/p99, throughput (tok/s, sent/s), peak RSS, CPU%, MPS utilization, **real energy via `powermetrics`** (sudo; falls back to `elapsed×assumed_watts` with a clear label), model **load/cold-start** time, on-disk + cache size.

## What to do
1. Implement workload generator + latency bench + power sampler (Task 5.1).
2. Run on **cpu** and **mps**; capture all metrics with warmup.
3. Produce latency-vs-length and throughput-vs-batch plots, per device.
4. Derive a **minimum-hardware sizing** note for the CPU-only Docker image.

## Outputs
- `reports/fork1/operational_envelope.md` + per-device plots.

## Findings (write from evidence)
- CPU envelope (deployment-relevant): both models were measured on CPU and MPS in `reports/fork1/operational_envelope.jsonl`; `reports/fork1/operational_envelope.md` contains all p50/p95/p99 rows and plots. On CPU batch=1, SecureBERT p50 latency by token length was 25.64 ms (16), 26.54 ms (32), 29.55 ms (64), 57.63 ms (128), 75.24 ms (256). CyNER was 23.73 ms, 28.33 ms, 42.39 ms, 66.12 ms, 124.69 ms. SecureBERT is slightly slower only at 16 tokens, then faster from 32 tokens onward and much faster at 256 tokens.
- CPU throughput/RSS/load: at batch=1 and 256 tokens, SecureBERT processed 3,402 tok/s vs CyNER 2,053 tok/s. CPU RSS after load was 291.0 MB for SecureBERT vs 780.7 MB for CyNER. Load time was 0.11 s vs 0.42 s. Energy is labeled `estimated` because `powermetrics` did not provide a real sample; estimates use elapsed x assumed watts.
- MPS envelope (local-dev ceiling): at batch=1, SecureBERT p50 latency was 14.73 ms (16), 12.62 ms (32), 16.93 ms (64), 21.69 ms (128), 40.91 ms (256). CyNER was 22.83 ms, 14.95 ms, 17.31 ms, 27.57 ms, 50.06 ms. CPU to MPS p50 speedup for SecureBERT was 1.74x at 16 tokens, 1.75x at 64, 2.66x at 128, and 1.84x at 256. CyNER speedup was 1.04x, 2.45x, 2.40x, and 2.49x for the same lengths.
- Cache/model footprint: SecureBERT cache size was 950.1 MB and 124,085,800 parameters; CyNER cache size was 1,072.0 MB and 277,461,515 parameters. MPS RSS after load was 301.1 MB for SecureBERT vs 407.9 MB for CyNER.

## Conclusion → contribution to model choice
- On CPU, SecureBERT is cheaper to operate for the deployment-relevant long inputs: at 256 tokens batch=1 it is 1.66x faster by p50 latency (75.24 ms vs 124.69 ms), uses about 37% of CyNER's RSS (291.0 MB vs 780.7 MB), loads about 4x faster, and has a smaller local cache.
- The operational profile reinforces the accuracy verdict rather than complicating it. CyNER only has a small CPU latency edge at very short 16-token inputs; SecureBERT is faster, smaller in memory, and more accurate for the workloads that matter more for CTI text.
- Vote toward final decision (08): SecureBERT on deployment cost as well as accuracy. Caveat: energy values are estimates, not real `powermetrics` readings.
