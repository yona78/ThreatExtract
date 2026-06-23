# ThreatExtract — DNRTI NER Benchmark (SecureBERT-NER vs CyNER)

Model-selection study for an **offline, on-prem threat-intelligence NER product**.
We compare two frozen open-source models — `CyberPeace-Institute/SecureBERT-NER`
and `AI4Sec/cyner-xlm-roberta-base` — on the **DNRTI** test split, under a label-space
mismatch (each model has its own taxonomy, projected onto DNRTI via the assignment PDF mapping).

> **Decision: SecureBERT-NER.** It wins every full-split statistical comparison, survives the
> protocol and robustness checks, expresses more of the DNRTI taxonomy, and has the smaller,
> faster CPU deployment footprint. The structural-bias caveat (see
> [doc 04](04_leakage_bias_and_intrinsics.md)) is quantified and accounted for; it narrows the
> margin but does not change the choice.

![Benchmark evaluation pipeline](figures/evaluation_pipeline.svg)

## Headline result (full DNRTI test split, strict SemEval entity scoring)

![Model comparison](figures/model_comparison_metrics.svg)

| Model | Strict P | Strict R | **Strict F1** | Relaxed (overlap) F1 | Gap vs other (strict F1) | 95% CI of gap |
|---|---:|---:|---:|---:|---:|---|
| **SecureBERT-NER** | 0.2239 | 0.3820 | **0.2823** | 0.5086 | +0.1785 | [0.1518, 0.2069] |
| CyNER | 0.1190 | 0.0920 | **0.1038** | 0.2604 | −0.1785 | [−0.2069, −0.1518] |

The strict-F1 gap of **0.1785** is statistically significant (paired bootstrap CI excludes 0;
McNemar confirms). It is partly a *taxonomy + training-lineage* advantage rather than pure
recognition skill — see the capability-vs-bias analysis in [doc 04](04_leakage_bias_and_intrinsics.md).

## Why SecureBERT leads (one line each)

- **Accuracy:** higher strict/exact/partial/type F1; wins 90/97 strict head-to-head configs, 7 ties, 0 losses.
- **Stability:** ranking never flips across preprocessing, protocol, robustness, or subset-size sweeps on the full split.
- **Deployment:** fewer parameters (124M vs 277M), smaller cache, lower RSS, faster CPU latency.
- **Caveat:** part of the raw gap is the label projection + APTNER lineage; a conservative ontology-ceiling adjustment still leaves a positive residual (~0.079).

## Report map

| Doc | Contents |
|---|---|
| [**06 — Model Selection (start here)**](06_model_selection.md) | **The decision — which NER model to use, with the full evidence, metrics, risks, and revisit conditions.** |
| [01 — Dataset & Label Mapping](01_dataset_and_label_mapping.md) | DNRTI test-split audit, label distribution, and the PDF taxonomy projection that drives the comparison. |
| [02 — Methodology & Results](02_methodology_and_results.md) | Evaluation pipeline, SemEval scoring schemes, headline + per-class results, seqeval cross-check, error analysis, seen/unseen entities. |
| [03 — Robustness, Sensitivity & Subsets](03_robustness_sensitivity_and_subsets.md) | Preprocessing sweeps, protocol comparison, input perturbations, dataset-size scaling, and the five-strategy subset study. |
| [04 — Leakage, Bias & Intrinsics](04_leakage_bias_and_intrinsics.md) | Literature/leakage review, structural-bias (oracle-ceiling) audit, tokenizer fertility, and label expressibility. |
| [05 — Operational & Calibration](05_operational_and_calibration.md) | Parameters, cache, memory, latency/throughput envelope, energy, and confidence calibration. |

## Dataset snapshot

DNRTI **test** split: **664** sentences, **17,716** tokens, **2,348** collapsed entity spans,
13 DNRTI labels, 16 malformed tag-only lines skipped. Evaluation uses the published `test.txt`
without resampling for the final claim.

## Reproduction

```bash
make reproduce
```

Models are run as frozen black boxes from `model_cache/fork1` (no training, no weight edits),
fully offline. Environment: macOS arm64, Python 3.12, torch 2.5.1, transformers 4.46.3, CPU device.

## Data artifacts

The human-readable analysis lives in the six markdown files above. The machine-readable
records that back every table are written alongside them by `make reproduce`:

- `benchmark_results.jsonl`, `master_table.jsonl` — per-config scores across all experiments.
- `methodology_*.jsonl` — baseline scores, per-label strict, confusion matrix, error rows, seqeval cross-check.
- `subset_study.jsonl`, `preprocessing_sensitivity.jsonl`, `protocol_comparison.jsonl`, `robustness.jsonl` — sweeps.
- `operational_envelope.jsonl`, `calibration_*.jsonl`, `intrinsic_metrics.jsonl`, `oov_entity_analysis.jsonl` — operational/reliability.
- `ambiguity_analysis.jsonl` — strict P/R/F1 on ambiguous vs unambiguous surfaces.
- `baseline_lexical.jsonl` — non-neural train-gazetteer sanity floor.
- `run_metadata.json` — environment, git commit, dataset stats, and input checksums.

## Methodology highlights

The evaluation is span-based and deliberately conservative. A few design choices keep the
comparison fair across two models with different taxonomies and tokenizers (full detail in
[doc 06 §7](06_model_selection.md)):

- **Whole-word prediction aggregation** (`aggregation_strategy="first"`) so multi-subword entities
  such as `StoneDrill` and `CrowdStrike` are scored as whole words.
- **One false positive per prediction** under one-to-many label projections, charged to a
  representative DNRTI label.
- **Best-overlap span matching**, **Unicode-format-character cleaning** at load, and a non-neural
  train-gazetteer sanity baseline (F1 0.521) plus an ambiguous-surface metric.

Capability is reported under two complementary views: boundary-agnostic **token-level F1**
(seqeval: SecureBERT 0.730, CyNER 0.334) — the primary capability metric — and a deliberately
strict **entity-level char-span F1** (0.282 / 0.104) used as a conservative lower bound. **The
model ranking is identical under both.**

## Future work

- Quantify APTNER/DNRTI raw-text overlap once `data/aptner/` is available (currently unquantified).
- Add MPS latency parity on an Apple M-series host (the on-prem image ships CPU-only torch).
- Add post-hoc confidence calibration (temperature/isotonic) and an abstention policy before
  exposing model confidence to analysts.
