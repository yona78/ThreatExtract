# Fork 1 Project Report: DNRTI NER Model Selection

## Abstract

This report benchmarks two cybersecurity named entity recognition models,
`CyberPeace-Institute/SecureBERT-NER` and `AI4Sec/cyner-xlm-roberta-base`, on the DNRTI threat
intelligence dataset. The central difficulty is not just inference accuracy:
the two models expose different taxonomies, DNRTI uses a third taxonomy,
and the selected model must be deployable in a fully offline on-premise
environment. We therefore evaluate strict exact-span F1, relaxed boundary
overlap F1, per-label behavior, subset-size stability, latency, memory,
model footprint, and leakage risk. On the full DNRTI test split,
SecureBERT-NER is the clear winner.

![Evaluation pipeline](figures/evaluation_pipeline.svg)

## 1. Experimental Setting

- DNRTI source: https://github.com/SCreaMxp/DNRTI-A-Large-scale-Dataset-for-Named-Entity-Recognition-in-Threat-Intelligence
- Test split: `664` sentences, `17716` tokens,
  `2348` collapsed gold spans.
- SecureBERT-NER: `CyberPeace-Institute/SecureBERT-NER`.
- CyNER: `AI4Sec/cyner-xlm-roberta-base`.
- Hardware in this run: CPU execution, because PyTorch reported
  `mps_available=false` in the current process.
- Offline mode: enabled, using cached Hugging Face snapshots under
  `model_cache/fork1`.

## 2. Headline Results

![Model comparison](figures/model_comparison_metrics.svg)

| Model | Exact F1 | Exact Precision | Exact Recall | Relaxed F1 | Full-test elapsed |
|---|---:|---:|---:|---:|---:|
| SecureBERT-NER | 0.2823 | 0.2239 | 0.3820 | 0.5086 | 24.468s |
| CyNER | 0.1047 | 0.1201 | 0.0928 | 0.2604 | 26.489s |

SecureBERT wins under the primary metric, exact micro-F1, and the ranking
is unchanged under relaxed boundary matching. That matters because relaxed
matching is designed to absorb small tokenization or boundary differences;
the conclusion is therefore not merely a boundary artifact.

## 3. Discussion

The absolute exact F1 values are modest because this is a cross-dataset,
cross-taxonomy benchmark. We are not evaluating models fine-tuned on DNRTI;
we are measuring transfer plus assignment-specified label projection. The
model taxonomy matters. CyNER collapses the world into five categories,
which makes it compact conceptually but unable to express DNRTI `Time` and
`Area`. SecureBERT has finer APTNER-derived labels and covers `TIME` and
`LOC`, which improves both recall and operational usefulness.

The result also changes how the product should be framed. SecureBERT is the
better deployment default, but DNRTI `Purp` and `Features` remain uncovered
by both models under the PDF mapping. If those classes are product-critical,
they require a second-stage classifier, weak rules, or fine-tuning.

## 4. Threats To Validity

- The benchmark reconstructs DNRTI sentences using normalized single spaces.
  This is appropriate for the provided token/tag files, but it may differ
  from original report whitespace.
- Reported energy is estimated from elapsed seconds and a fixed wattage
  assumption. It is not a hardware-counter measurement.
- The current run is CPU-only. The script supports `--device mps`; rerunning
  on an M4 process with MPS exposed should improve latency, but not the
  model-selection conclusion unless a backend-specific inference bug appears.
- CyberNER-trained checkpoints should not be substituted into this comparison
  because CyberNER includes DNRTI and would contaminate the benchmark.

## 5. Conclusion

Deploy SecureBERT-NER as the offline on-premise default. It dominates CyNER
on exact F1, relaxed F1, recall, model footprint, memory peak, and elapsed
time in this benchmark.
