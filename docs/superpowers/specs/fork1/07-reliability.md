# Direction 07 — Reliability (Robustness + Calibration)

**Question:** Can we trust the model's confidences, and does realistic input noise break it?
**Maps to:** RQ6 · Experiments **E9 (robustness) + E10 (calibration)** · Implementation: appendix Phase 1 Task 1.5 (perturbations) + Phase 6 (Task 6.1).
**Why it matters for the choice:** A SOC product needs (a) reliable confidence scores so analysts can set a high-precision triage threshold, and (b) stability on messy real-world threat text (defanged IOCs like `hxxp://`/`1.1.1[.]1`, inconsistent casing, typos). A more accurate but uncalibrated or brittle model can be the worse product.

## Setup
- Models (frozen): both. Device: MPS for E9; CPU/MPS parity remains pending.
- **Robustness (E9):** apply perturbations to input text and re-score: `defang`, `refang`, `random_case`, `keyboard_typo(rate)` — each deterministic given a seed.
- **Calibration (E10):** read the pipeline's per-entity confidence (`score`); bin predicted entities by score; precision per bin.
- Instrument: Direction 01 for ΔF1; ECE for calibration.

## What to do
1. Implement perturbations (Task 1.5) and run clean→noisy eval for both models; report ΔF1 with CI.
2. Implement ECE, reliability bins, threshold sweep (Task 6.1).
3. Compute ECE per model; sweep confidence thresholds to find a precision≥0.9 operating point (or report "unreachable").
4. Produce a reliability diagram per model.

## Outputs
- `reports/fork1/robustness.md`, `reports/fork1/calibration.md` + reliability diagram.

## Findings (write from evidence)
- Robustness (E9): no perturbation flipped the head-to-head ranking. SecureBERT stayed ahead for every noisy condition, with strict-F1 gaps that excluded 0: defang +0.1941 (95% CI [0.1685, 0.2214]), refang +0.1785 ([0.1518, 0.2069]), random-case +0.0238 ([0.0145, 0.0336]), and keyboard-typo +0.1152 ([0.0917, 0.1398]).
- Defang was effectively neutral for SecureBERT: clean 0.2823 to noisy 0.2839, ΔF1 +0.0016 (95% CI [-0.0015, 0.0049]). It hurt CyNER: 0.1038 to 0.0898, ΔF1 -0.0139 ([-0.0202, -0.0081]).
- Refang was a no-op on this DNRTI test run for both models: ΔF1 0.0000 with [0.0000, 0.0000] intervals.
- Random casing was the strongest perturbation for both models. SecureBERT dropped from 0.2823 to 0.0296, ΔF1 -0.2527 ([-0.2766, -0.2300]); CyNER dropped from 0.1038 to 0.0058, ΔF1 -0.0980 ([-0.1130, -0.0836]). SecureBERT still led, but the gap narrowed sharply.
- Keyboard typos hurt SecureBERT more in absolute F1: 0.2823 to 0.2177, ΔF1 -0.0645 ([-0.0765, -0.0535]). CyNER was statistically unchanged under this perturbation: 0.1038 to 0.1025, ΔF1 -0.0012 ([-0.0097, 0.0070]).
- Calibration (E10): pending. ECE, reliability bins, and high-precision threshold sweeps have not been run yet.

## Conclusion → contribution to model choice
- Robustness vote: SecureBERT remains the better model by noisy strict F1 because every perturbation preserves a positive gap with a 95% CI excluding 0. The caveat is brittleness: SecureBERT loses far more absolute F1 under random casing and keyboard typos, while CyNER starts much lower and therefore has less absolute score to lose.
- High-precision analyst triage: no vote yet. Calibration and threshold evidence are still pending.
- Direction 07 vote toward final decision: provisional SecureBERT on robustness-only evidence; final Direction 07 vote is pending E10 calibration.
