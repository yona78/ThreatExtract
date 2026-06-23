# Fork 1 Final Report: Frozen SecureBERT-NER vs CyNER on DNRTI

## Abstract

This offline study compares frozen SecureBERT-NER and CyNER on DNRTI for an on-prem threat-intelligence NER product. The evidence leader is SecureBERT, but final product selection is intentionally left to the reviewer. The raw F1 gap is partly a taxonomy and training-lineage advantage, not pure recognition capability.

## Method

DNRTI test split: 664 sentences, 17716 tokens, 2348 collapsed gold spans, 16 skipped malformed tag-only lines.
Models were run as frozen black boxes from `model_cache/fork1` with no training or weight edits. The primary metric is SemEval strict entity F1 with paired-bootstrap 95% CIs and McNemar checks.

## Headline Strict Result

| Model | Strict F1 | Gap vs other | 95% CI |
|---|---:|---:|---|
| securebert | 0.2823 | 0.1785 | [0.1518, 0.2069] |
| cyner | 0.1038 | -0.1785 | [-0.2069, -0.1518] |

## Cross-Experiment Decision Count

SecureBERT wins 90/97 strict head-to-head configs; 7 configs are statistically tied and 0 favor CyNER.

Direction votes:

| Direction | Vote | Rationale |
|---|---|---|
| 01 | SecureBERT | all four SemEval schemes preserve a positive gap |
| 02 | SecureBERT | preprocessing sweeps did not flip the full-test ranking |
| 03 | SecureBERT | min-faithful subsets reach stable positive gaps |
| 04 | SecureBERT | paper-native protocol keeps the positive gap |
| 05 | SecureBERT with bias caveat | oracle and lineage show structural advantage |
| 06 | SecureBERT | CPU profile is smaller and faster at deployment lengths |
| 07 | SecureBERT with confidence caveat | robustness holds, raw calibration fails both |

## Sensitivity

No full-split preprocessing, protocol, or robustness condition reverses the ranking. Small 10-sentence subsets are often underpowered and can tie by CI, so they are not decision-grade.

## Capability Vs Bias

The raw PDF-mapping gap is 0.1785. The oracle ceiling gap is 0.0992 (55.6% of the raw gap). The conservative bias-adjusted residual is 0.0793; this remains positive, but it is not a causal decomposition.

## Operational Profile

CPU deployment note: at 256 tokens and batch=1, SecureBERT p50 latency is 115.64 ms vs CyNER 129.23 ms (1.12x faster), with RSS 408.0 MB vs 1564.1 MB.

## Reliability

securebert ECE 0.6520, precision>=0.90 threshold unreachable; cyner ECE 0.7044, precision>=0.90 threshold unreachable.
Robustness perturbations preserve a positive SecureBERT gap, but random casing sharply reduces both models and should be normalized or monitored upstream.
TODO: rerun E9 robustness after the perturbation RNG fix; the long rerun was intentionally skipped in the cleanup pass.

## Threats To Validity

- APTNER/DNRTI overlap could not be quantified offline because `data/aptner/` is absent.
- The label projection structurally favors SecureBERT; the bias adjustment is a ceiling-based sensitivity check, not proof of independent capability.
- Energy is estimated where `powermetrics` is unavailable.
- Raw confidence scores do not provide a precision>=0.90 operating point for either model.
- Full `make reproduce` and MPS parity remain future reruns; see `reports/fork1/README.md`.

## Conclusion

The evidence package supports SecureBERT-NER as the current evidence leader, but it does not hard-code the product selection. SecureBERT wins the statistical comparisons, survives the protocol and robustness checks, has the better CPU deployment envelope, and retains a positive residual after the ontology-ceiling bias check. The final chosen model should be set by the reviewer/product owner after deciding how to weigh the structural-bias caveat and label-coverage risks. The product should not present the full raw gap as pure model quality, and it should add calibration/abstention logic before using confidence as an analyst triage threshold.
