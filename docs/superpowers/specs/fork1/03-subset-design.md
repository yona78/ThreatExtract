# Direction 03 — Evaluation Subset Design

**Question:** Does the winner depend on *which* sentences and *how many* we evaluate on?
**Maps to:** RQ2 · Experiment **E4** · Implementation: appendix Phase 2 (Tasks 2.1–2.2).
**Why it matters for the choice:** The assignment explicitly allows subsetting "but specify how you created the subset and why." Small or skewed subsets can flip rankings or hide variance. This direction turns subset choice from an afterthought into a controlled variable and finds the smallest subset that still yields a trustworthy decision (useful for fast iteration and for the operational workload in Direction 06).

## Setup
- Models (frozen): both. Device: MPS.
- Strategies: **random**, **label-stratified** (preserve per-label support), **density** (by entity count), **length** (by token count), **hardness** (rare labels `Way/Purp/Features` + length).
- Sizes: {10, 50, 100, 250, all}. Seeds: {1,2,3} for variance.
- Best preprocessing from Direction 02; instrument from Direction 01.

## What to do
1. Implement the subset strategies with invariant tests (proportions/quotas) (Task 2.1).
2. Run both models across strategy × size × seed (Task 2.2).
3. Compute per-cell mean/variance of strict-F1 and the winner.
4. Compute **min-faithful-subset** = smallest size at which the full-split winner wins with CI-excluding-0 across all 3 seeds.

## Outputs
- `reports/fork1/subset_study.md` + "F1 vs size" curves per strategy.

## Findings (write from evidence)
- E4 produced 150 head-to-head rows in `reports/fork1/subset_study.jsonl`: 5 strategies x 5 sizes x 3 seeds x 2 models. The full split reproduced the Direction 02 baseline for every strategy's `all` row: SecureBERT strict-F1 0.2823, CyNER 0.1038, gap +0.1785; the SecureBERT 95% CI ranged from [0.1509, 0.2065] to [0.1514, 0.2055] across bootstrap seeds, with flip flag `no`.
- F1-vs-size curves are written to `reports/fork1/figures/subset_random.svg`, `subset_label_stratified.svg`, `subset_density.svg`, `subset_length.svg`, and `subset_hardness.svg`. The Markdown report includes the mean/variance table. At size 50, mean SecureBERT/CyNER strict-F1 was random 0.2784/0.0941, label-stratified 0.3477/0.1151, density 0.2190/0.0714, length 0.2776/0.1062, and hardness 0.2434/0.0178.
- Smallest subset preserving the full-split verdict with all three seeds CI-excluding-0: random 50, label-stratified 50, density 50, length 50, hardness 10. Hardness reaches the verdict at 10 because it deliberately selects rare-label/long sentences; it is not a representative fast gate.
- Ranking instability was confined to 10-sentence subsets outside hardness. The gap always stayed positive for SecureBERT, but the 95% CI included 0 for random size 10 in 2/3 seeds, label-stratified size 10 in 2/3 seeds, density size 10 in 1/3 seeds, and length size 10 in 1/3 seeds. All size 50, 100, 250, and all-split cells had flip flag `no`.

## Conclusion → evidence contribution
- The ranking is stable once the subset reaches 50 sentences for representative strategies: SecureBERT wins every size >=50 cell with a 95% CI excluding 0. The 10-sentence subsets are too small for trustworthy model-choice evidence except for the deliberately biased hardness strategy.
- Recommended fast regression gate: a fixed 50-sentence label-stratified subset for verdict checks, because it preserves label support and clears the CI rule across all seeds. Use the full test split for publication numbers, and use at least 250 random/label-stratified examples when absolute F1 comparability to the full split matters.
- Direction 03 evidence contribution: SecureBERT leads, with a subset-size caveat that tiny 10-sentence representative samples are underpowered and should be treated as smoke tests, not evidence for model choice.
