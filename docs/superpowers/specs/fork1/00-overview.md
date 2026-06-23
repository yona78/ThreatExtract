# Fork 1 — Research Overview & Evidence Framework

**How to compare two frozen, taxonomy-mismatched black-box NER models (SecureBERT-NER vs CyNER) for a cyber-threat-intelligence product on DNRTI, deployable on-prem.**

This folder holds one brief per **research direction**. Each brief fully describes that direction: the question, why it matters for the model-choice evidence, the device/setup, the experiments, and a mandatory **Findings** + **Conclusion** section to be written from evidence. Line-by-line implementation (TDD tasks + code) lives in the implementation appendix: [`../2026-06-22-fork1-data-centric-ner-eval-design.md`](../2026-06-22-fork1-data-centric-ner-eval-design.md).

## Thesis

A single F1 number cannot fairly pick between these models, because the result is confounded by (a) preprocessing, (b) the evaluation subset, (c) the label projection, and (d) training-data/ontology bias favoring SecureBERT. We therefore measure the **sensitivity** of the decision to each factor, **separate capability from bias**, profile **on-prem cost**, and only then **conclude**.

## Direction map

| # | Brief | Question | Experiments | RQ |
|---|---|---|---|---|
| 01 | [Evaluation methodology](01-evaluation-methodology.md) | How do we measure "better" rigorously? | E1, E5 | all |
| 02 | [Preprocessing sensitivity](02-preprocessing-sensitivity.md) | Does the winner depend on how we preprocess input? | E2 | RQ1 |
| 03 | [Subset design](03-subset-design.md) | Does the winner depend on which/how much data we score? | E4 | RQ2 |
| 04 | [Protocol faithfulness](04-protocol-faithfulness.md) | Does the winner change under each model's native protocol? | E3 | RQ3 |
| 05 | [Capability vs bias](05-capability-vs-bias.md) | How much of the gap is real vs structural advantage? | E6, E8 | RQ4 |
| 06 | [On-prem deployment](06-onprem-deployment.md) | Which model is deployable on-prem (M4 GPU & CPU)? | E7 | RQ5 |
| 07 | [Reliability](07-reliability.md) | Are confidences trustworthy; does noise break it? | E9, E10 | RQ6 |
| 08 | [Synthesis & evidence package](08-synthesis-decision.md) | Final, bias-adjusted evidence-leader package. | E11 | all |

## Device strategy (M4, 10-core GPU)

- **M4 GPU = MPS** (`--device mps`); **M4 CPU = `--device cpu`**. First verify `torch.backends.mps.is_available()` on the host and enable it.
- **Accuracy directions (01–05, 07, 08):** run on **MPS** for speed; results are device-invariant — confirm once with a **CPU↔MPS parity check** (strict-F1 delta < 0.005).
- **On-prem direction (06):** measure on **both MPS and CPU**.
- **Deployment relevance:** the offline Docker image ships **CPU-only torch** → **CPU is the customer-facing number**; MPS is the local-dev/accelerated ceiling. Every operational claim states which device it refers to.

## Decision rule (used in every brief)

Model A beats B on a config iff the paired-bootstrap **95% CI of the strict-F1 gap excludes 0** (corroborated by McNemar p<0.05). Otherwise "tie". The synthesis package (08) reports the count of directions/configs where one model wins, ties, or loses — plus the bias adjustment from 05.

## How findings flow

Each brief writes its own **Findings** (evidence) and **Conclusion → vote**. Brief 08 aggregates all votes into an evidence-leader summary and a sensitivity statement ("SecureBERT wins in K/N configs; ranking flips only under X"). No brief asserts a winner before its experiments run.
