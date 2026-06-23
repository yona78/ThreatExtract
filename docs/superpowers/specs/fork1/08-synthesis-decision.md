# Direction 08 — Synthesis & Evidence Package

**Question:** Taking all evidence together, which model is the current evidence leader — and how confident are we?
**Maps to:** RQ(all) · Experiment **E11** · Implementation: appendix Phase 7 (Tasks 7.1–7.2).
**Why it matters:** This is where the directions become an evidence package. It must aggregate, not cherry-pick, and it must state the evidence leader's *robustness*, not just a winner.

## Setup
- Inputs: every `reports/fork1/*.jsonl` produced by Directions 01–07.
- Evidence-leader rule: per config, winner = strict-F1 gap CI excludes 0 (+ McNemar). Aggregate across all configs.

## What to do
1. Build the **master table**: config × model × scheme × F1 × CI × flip (Task 7.1).
2. Tally votes from each direction's Conclusion (01–07).
3. Apply the **bias adjustment** from Direction 05 (separate capability from structural advantage).
4. Rewrite `reports/fork1/benchmark_summary.md` as the publication-style report (abstract, RQ1–RQ6, method, results-with-CIs, sensitivity analysis, error/bias analysis, operational profile, reliability, threats, conclusion). Regenerate all figures.
5. Wire `make reproduce` + extended `run_metadata.json` (Task 7.2).

## Outputs
- Upgraded `reports/fork1/benchmark_summary.md` (the "paper"); `make reproduce`.

## Findings (write from evidence — the aggregate)
- Master table summary (`reports/fork1/master_table.jsonl`): 200 model rows total, including 97 strict head-to-head SecureBERT decision rows. Applying the decision rule (95% CI excludes 0 and McNemar p<0.05 when available), SecureBERT wins 90/97 strict configs, 7 are statistical ties, and 0 favor CyNER.
- The tied configs are not evidence for a CyNER win. One preprocessing row (`context=document`) has a small positive CI but McNemar p=0.0545, so it fails the corroboration rule. The other six ties are underpowered 10-sentence subset cells: random seeds 1/2, label-stratified seeds 1/3, density seed 1, and length seed 3. No full-test preprocessing, protocol, or robustness config reverses the ranking.
- Bias-adjusted gap: the raw PDF-mapping strict-F1 gap is +0.1785. The oracle expressibility ceiling gap from Direction 05 is +0.0992, about 55.6% of the raw gap. Treating that ceiling difference as a conservative structural handicap leaves a positive residual of +0.0793. This is a sensitivity check, not a causal decomposition.

## Evidence conclusion (write from evidence)
- **Evidence leader:** SecureBERT-NER. It is the current evidence leader because it wins the statistical comparisons, keeps the advantage under protocol and robustness checks, has the better CPU deployment envelope, and still has a positive residual after the ontology-ceiling bias check. The product's final chosen model is intentionally left to the reviewer/product owner.
- **Confidence & sensitivity statement:** The conclusion holds across all full-test preprocessing, subset sizes >=50 for representative strategies, paper-native protocol, and all robustness perturbations. It weakens only for tiny 10-sentence subsets and the `context=document` preprocessing row, which is not a viable headline protocol because it truncates/collapses the evaluation context and fails McNemar corroboration.
- **Deployment note:** CPU is the shipped Docker-relevant device. At 256 tokens and batch=1, SecureBERT p50 latency is 75.24 ms vs CyNER 124.69 ms (1.66x faster), with RSS 291.0 MB vs 780.7 MB. MPS is useful as a local-dev ceiling, but the product-facing note is CPU.
- **Residual risks & what would change the decision:** proven APTNER/DNRTI overlap could downgrade confidence in SecureBERT's capability claim; a CyNER-fair remapping or a CyNER checkpoint with DNRTI-compatible labels could narrow the gap; product-critical `Purp`/`Features` extraction would require another layer because both mappings leave those labels uncovered; neither raw confidence stream reaches precision>=0.90, so analyst triage needs calibration/abstention logic; random casing should be normalized upstream.
