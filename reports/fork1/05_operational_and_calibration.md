# 05 — Operational & Calibration

For an offline Docker product, accuracy is necessary but not sufficient: image size, RAM,
startup time, latency, and confidence usability all affect deployment. SecureBERT is not just
more accurate — it is also the cheaper model to ship.

---

## 1. Deployment footprint

![Accuracy and deployment tradeoffs](figures/operational_tradeoffs.svg)

| Model | Parameters | Cache bytes | Full-test elapsed | RSS peak MB | Estimated energy J |
|---|---:|---:|---:|---:|---:|
| SecureBERT-NER | 124,085,800 | 996,220,887 | 24.140 s | 812.77 | 434.52 |
| CyNER | 277,461,515 | 1,124,083,139 | 27.400 s | 948.53 | 493.19 |

SecureBERT has **~2.2× fewer parameters**, a smaller Hugging Face cache (smaller image/mounted
artifact), lower peak RSS (lower minimum customer hardware), and lower full-test elapsed time.
Energy is an **estimate** (elapsed × an assumed 18 W profile); for a production-grade number,
rerun while sampling `powermetrics` for package power.

---

## 2. Latency & throughput envelope

CPU is the deployment-relevant device because the offline Docker image ships CPU-only torch; MPS
is the local-dev ceiling. Measured on synthetic workloads at several token lengths and batch
sizes (batch=1 is the Streamlit single-document path).

![p50 latency by token length](figures/operational_latency.svg)
![Throughput by token length](figures/operational_throughput.svg)

| Device | Model | Tokens | Batch | p50 ms | p95 ms | sent/s | tok/s | RSS MB | Load s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cpu | securebert | 16 | 1 | 27.15 | 32.49 | 36.05 | 576.81 | 408.0 | 0.12 |
| cpu | securebert | 16 | 8 | 48.17 | 51.11 | 165.11 | 2641.81 | 408.0 | 0.12 |
| cpu | securebert | 16 | 32 | 228.72 | 254.53 | 138.77 | 2220.31 | 408.0 | 0.12 |
| cpu | securebert | 32 | 1 | 32.25 | 36.81 | 30.48 | 975.29 | 408.0 | 0.12 |
| cpu | securebert | 32 | 8 | 92.05 | 95.65 | 86.19 | 2758.16 | 408.0 | 0.12 |
| cpu | securebert | 32 | 32 | 447.09 | 467.45 | 71.31 | 2281.82 | 408.0 | 0.12 |
| cpu | securebert | 64 | 1 | 34.28 | 36.71 | 29.14 | 1864.68 | 408.0 | 0.12 |
| cpu | securebert | 64 | 8 | 166.93 | 168.69 | 47.95 | 3068.52 | 408.0 | 0.12 |
| cpu | securebert | 64 | 32 | 712.62 | 723.26 | 44.92 | 2874.77 | 408.0 | 0.12 |
| cpu | securebert | 128 | 1 | 58.21 | 59.71 | 17.13 | 2193.23 | 408.0 | 0.12 |
| cpu | securebert | 128 | 8 | 351.98 | 354.33 | 22.73 | 2909.63 | 408.0 | 0.12 |
| cpu | securebert | 128 | 32 | 1320.83 | 1447.20 | 23.86 | 3055.33 | 408.0 | 0.12 |
| cpu | securebert | 256 | 1 | 115.64 | 238.64 | 7.39 | 1892.99 | 408.0 | 0.12 |
| cpu | securebert | 256 | 8 | 703.44 | 896.40 | 10.72 | 2745.52 | 408.0 | 0.12 |
| cpu | securebert | 256 | 32 | 2827.28 | 3015.58 | 11.16 | 2857.96 | 408.0 | 0.12 |
| cpu | cyner | 16 | 1 | 36.00 | 45.52 | 26.51 | 424.18 | 1564.1 | 0.41 |
| cpu | cyner | 16 | 8 | 77.10 | 77.80 | 104.95 | 1679.20 | 1564.1 | 0.41 |
| cpu | cyner | 16 | 32 | 291.42 | 299.63 | 109.33 | 1749.32 | 1564.1 | 0.41 |
| cpu | cyner | 32 | 1 | 29.60 | 32.69 | 34.14 | 1092.36 | 1564.1 | 0.41 |
| cpu | cyner | 32 | 8 | 135.87 | 145.08 | 58.27 | 1864.69 | 1564.1 | 0.41 |
| cpu | cyner | 32 | 32 | 620.85 | 717.91 | 49.87 | 1595.99 | 1564.1 | 0.41 |
| cpu | cyner | 64 | 1 | 57.17 | 63.17 | 17.35 | 1110.59 | 1564.1 | 0.41 |
| cpu | cyner | 64 | 8 | 237.97 | 239.91 | 33.55 | 2147.01 | 1564.1 | 0.41 |
| cpu | cyner | 64 | 32 | 922.57 | 943.80 | 34.60 | 2214.13 | 1564.1 | 0.41 |
| cpu | cyner | 128 | 1 | 79.05 | 80.12 | 12.84 | 1643.63 | 1564.1 | 0.41 |
| cpu | cyner | 128 | 8 | 481.35 | 494.50 | 16.56 | 2119.47 | 1564.1 | 0.41 |
| cpu | cyner | 128 | 32 | 1684.04 | 1709.21 | 18.98 | 2429.87 | 1564.1 | 0.41 |
| cpu | cyner | 256 | 1 | 129.23 | 130.03 | 7.81 | 1998.73 | 1564.1 | 0.41 |
| cpu | cyner | 256 | 8 | 898.23 | 932.13 | 8.84 | 2263.66 | 1564.1 | 0.41 |
| cpu | cyner | 256 | 32 | 3808.92 | 3965.78 | 8.34 | 2134.26 | 1564.1 | 0.41 |

*MPS rows are unavailable on this host (`mps_available=false`); rerun on an Apple M-series host to
populate them. Full grid incl. p99 + energy in `operational_envelope.jsonl`.*

**Headline deployment point (256 tokens, batch=1):** SecureBERT p50 **115.64 ms** vs CyNER
**129.23 ms** (1.12× faster), with RSS **408 MB** vs **1564 MB** — a ~3.8× memory difference and
faster startup (0.12 s vs 0.41 s). Model selection should still be accuracy-first; MPS mainly
shifts the operational envelope, not the ranking.

---

## 3. Confidence calibration

Can the raw model confidence be used as an analyst triage threshold? **Not as-is** — both models
are badly miscalibrated.

![Reliability diagram](figures/calibration_reliability.svg)

| Model | ECE | Entity predictions | Precision ≥ 0.90 threshold |
|---|---:|---:|---|
| securebert | 0.6520 | 4007 | unreachable |
| cyner | 0.7044 | 1815 | unreachable |

### Threshold sweep (condensed)

| Model | Threshold | Predictions | Precision | Recall |
|---|---:|---:|---:|---:|
| securebert | 0.00 | 4007 | 0.2239 | 0.3820 |
| securebert | 0.50 | 3912 | 0.2270 | 0.3782 |
| securebert | 0.80 | 3544 | 0.2322 | 0.3505 |
| securebert | 0.85 | 3387 | 0.2273 | 0.3279 |
| securebert | 0.90 | 2873 | 0.2102 | 0.2572 |
| securebert | 0.95 | 0 | 0.0000 | 0.0000 |
| cyner | 0.00 | 1815 | 0.1201 | 0.0928 |
| cyner | 0.50 | 1692 | 0.1288 | 0.0928 |
| cyner | 0.75 | 1297 | 0.1557 | 0.0860 |
| cyner | 0.90 | 868 | 0.1440 | 0.0532 |
| cyner | 0.95 | 644 | 0.1460 | 0.0400 |
| cyner | 0.99 | 276 | 0.0870 | 0.0102 |

Raising the confidence threshold barely moves precision for either model: SecureBERT precision
never exceeds ~0.23 and collapses to 0 above 0.95 (no predictions survive); CyNER peaks around
0.155. **There is no high-precision operating point from raw scores.** Before confidence is used
for analyst triage, the product needs post-hoc calibration (temperature scaling / isotonic) and
an abstention policy. Full sweep in `calibration_thresholds.jsonl`.

---

## Operational verdict

SecureBERT-NER wins the deployment axis as well as accuracy: smaller, lower-memory, faster to
load and to infer at realistic document lengths. The one shared weakness — unusable raw
confidence — is a calibration/abstention work item, not a model-selection differentiator.
