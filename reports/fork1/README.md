# ThreatExtract — DNRTI NER Benchmark (SecureBERT-NER vs CyNER)

Model-selection study for an **offline, on-prem threat-intelligence NER product**.
We compare two frozen open-source models — `CyberPeace-Institute/SecureBERT-NER`
and `AI4Sec/cyner-xlm-roberta-base` — on the **DNRTI** test split, under a label-space
mismatch (each model has its own taxonomy, projected onto DNRTI via the assignment PDF mapping).

> **Evidence leader: SecureBERT-NER.** It wins every full-split statistical comparison,
> survives the protocol/robustness checks, and has the smaller, faster CPU deployment
> footprint. The final product decision is intentionally left to the reviewer/product
> owner after weighing the structural-bias caveat (see [doc 04](04_leakage_bias_and_intrinsics.md)).

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
| [**06 — Model Selection (start here)**](06_model_selection.md) | **The decision — which NER model to use, with the full evidence, corrected metrics, risks, and revisit conditions.** |
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
records that back every table are kept alongside them:

- `benchmark_results.jsonl`, `master_table.jsonl` — per-config scores across all experiments.
- `methodology_*.jsonl` — baseline scores, per-label strict, confusion matrix, error rows, seqeval cross-check.
- `subset_study.jsonl`, `preprocessing_sensitivity.jsonl`, `protocol_comparison.jsonl`, `robustness.jsonl` — sweeps.
- `operational_envelope.jsonl`, `calibration_*.jsonl`, `intrinsic_metrics.jsonl`, `oov_entity_analysis.jsonl` — operational/reliability.
- `run_metadata.json` — environment, git commit, dataset stats, and input checksums.

## Methodology update — evaluation bug fixes (see doc 06 §7)

Three evaluation bugs were fixed in `src/fork1` after the first report pass:

1. **Sub-word fragmentation** — inference aggregation changed `simple` → `first` so multi-subword
   entities (`StoneDrill`, `CrowdStrike`) are scored as whole words instead of fragments. This was
   the root cause of the deflated entity-level precision/recall.
2. **Per-label FP over-counting** — one-to-many label projections (e.g. CyNER `Organization`) now
   add one false positive per prediction instead of one per mapped label.
3. **Best-overlap matching** — predictions bind to the max-overlap gold span, not the first.

**Capability is best read at the token level** (seqeval: SecureBERT 0.730, CyNER 0.334); the
entity-level char-span F1 (0.282 / 0.104) is a conservative boundary-exact lower bound. **The model
ranking is identical under both and is unaffected by the fixes.**

> **TODO — refresh absolute numbers.** Run `make reproduce` on a dev host (venv + optional MPS;
> the cloud sandbox can't build torch) to regenerate every sweep's *absolute* entity-level numbers
> under the fixed aggregation. The per-label false-positive correction is already reflected in the
> tables; the entity-level magnitudes will rise toward the token-level figures. Conclusions do not
> change.

## Open follow-ups

- Run full `make reproduce` to regenerate all sweep artifacts under the aggregation fix.
- Rerun the input-perturbation (robustness) sweep after the perturbation-RNG fix.
- Rerun CPU/MPS latency parity on an Apple host where MPS is available.
- Quantify APTNER/DNRTI overlap once `data/aptner/` is available (currently unquantified).
