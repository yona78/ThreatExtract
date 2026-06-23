# Operational Metrics For On-Prem Deployment

## Why Operations Matter

The selected model must ship inside an offline Docker environment. Accuracy is
necessary, but image size, RAM headroom, startup time, and energy cost also
affect customer deployment feasibility.

![Operational tradeoffs](figures/operational_tradeoffs.svg)

| Model | Parameters | Cache bytes | Full-test elapsed | RSS peak MB | Estimated energy J |
|---|---:|---:|---:|---:|---:|
| SecureBERT-NER | 124085800 | 996220887 | 24.140s | 812.765625 | 434.524 |
| CyNER | 277461515 | 1124083139 | 27.400s | 948.53125 | 493.191 |

## Discussion

SecureBERT is not just more accurate in this benchmark; it is also the easier
model to ship. It has fewer parameters, a smaller Hugging Face cache, lower
observed RSS peak, and lower full-test elapsed time. This matters for an
offline Docker image because model cache size directly affects image size or
mounted artifact size, and memory peak affects minimum customer hardware.

Energy is reported as an estimate: elapsed seconds multiplied by an assumed
18 W device profile. For a production-grade measurement on macOS, rerun the
benchmark while sampling `powermetrics` and record package power directly.

## MPS Note

The script supports `--device auto`, `--device mps`, and `--device cpu`. This
run used CPU because the process reported `mps_available=false`. On an Apple
M4 workstation, rerun with MPS exposed to get deployment-relevant latency.
The model-selection logic should still be based on accuracy first; MPS mainly
changes the operational envelope.
