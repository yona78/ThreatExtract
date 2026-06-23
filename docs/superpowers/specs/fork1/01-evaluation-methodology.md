# Direction 01 — Evaluation Methodology (the measuring instrument)

**Question:** How do we measure "better" rigorously enough to defend a model choice?
**Maps to:** RQ(all) · Experiments **E1, E5** · Implementation: appendix Phase 0 (Tasks 0.2–0.4).
**Why it matters for the choice:** Every other direction reports numbers *through this instrument*. If the metric is wrong (e.g., token-level accuracy, or naive seqeval that can't handle the one-to-many label mapping), the whole decision is wrong. This direction makes the verdict trustworthy.

## Setup
- Models (frozen): SecureBERT-NER, CyNER. Device: MPS (accuracy is device-invariant; parity-checked once vs CPU).
- Gold: DNRTI `test.txt` (664 sentences, 2,348 collapsed spans, 13 labels).
- Metric: **SemEval 4-scheme** entity-level scoring (strict / exact / partial / type) built on MUC counts (correct/incorrect/partial/missing/spurious), implemented directly on spans with set-membership labels so the one-to-many projection (e.g., CyNER `System`→{OffAct,Way}) scores correctly. `seqeval` strict is used as a cross-check on the 1:1-mappable labels.
- Significance: paired-bootstrap 95% CI on the strict-F1 gap (10k resamples) + McNemar.

## What to do
1. Implement the 4-scheme scorer + MUC counts; unit-test on hand-built spans (appendix Task 0.2).
2. Implement bootstrap CI + McNemar (Task 0.3).
3. Implement alignment policies for the seqeval cross-check (Task 0.4).
4. Reconcile: confirm strict-F1 on the default config reproduces the legacy exact numbers (SecureBERT ≈0.282, CyNER ≈0.105) within ±0.01.

## Outputs
- `src/fork1/metrics.py`, `tests/fork1/test_metrics.py`, a reconciled baseline table in `reports/fork1/benchmark_summary.md`.

## Findings (write from evidence after running)
- Baseline 4-scheme table (`reports/fork1/methodology_baseline.jsonl`): SecureBERT leads under every SemEval scheme, and every gap CI excludes 0. Strict F1 is 0.2823 vs 0.1038, gap +0.1785 (95% CI [0.1518, 0.2069], McNemar p=9.32e-113, flip `no`). Exact F1 is 0.2999 vs 0.1581, gap +0.1419 ([0.1127, 0.1729], flip `no`). Partial F1 is 0.4230 vs 0.2710, gap +0.1520 ([0.1259, 0.1799], flip `no`). Type F1 is 0.5013 vs 0.2484, gap +0.2530 ([0.2230, 0.2840], flip `no`).
- MUC error mix explains the strict losses. SecureBERT has many spurious entities (2216) and type/boundary incorrect matches (894), but it finds far more exact typed spans than CyNER (897 correct vs 216) and misses far fewer gold spans (557 vs 1436). CyNER emits fewer spurious predictions (903) but misses most DNRTI spans under the mapped ontology.
- Seqeval cross-check delta on the 1:1-label BIO subset is 0.0000 for both models, below the <0.005 tolerance. The unique-label subset is not the headline metric, but it validates the token-aligned strict implementation against `seqeval`: SecureBERT unique-label F1 0.7300 over 1601 unique-label gold spans; CyNER 0.3341 over 695.

## Conclusion → contribution to model choice
- The instrument does not reverse the old exact-only ranking. It strengthens confidence in the choice because SecureBERT wins under strict, exact, partial, and type scoring, and the strict gap reproduces the legacy sanity target within tolerance.
- Strict F1 remains the decision metric because the product needs exact entity boundaries and the DNRTI label; type F1 is diagnostic. The type-score gap is larger (+0.2530), which shows SecureBERT's taxonomy is much closer to DNRTI even when boundary errors are discounted.
- Direction 01 vote toward final decision: SecureBERT, with the caveat that this measuring instrument also exposes a structural ontology advantage that Direction 05 must discount rather than treating the full raw gap as pure recognition capability.
