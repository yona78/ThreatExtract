# Direction 07 — Reliability (Robustness + Calibration)

**Question:** Can we trust the model's confidences, and does realistic input noise break it?
**Maps to:** RQ6 · Experiments **E9 (robustness) + E10 (calibration)** · Implementation: appendix Phase 1 Task 1.5 (perturbations) + Phase 6 (Task 6.1).
**Why it matters for the choice:** A SOC product needs (a) reliable confidence scores so analysts can set a high-precision triage threshold, and (b) stability on messy real-world threat text (defanged IOCs like `hxxp://`/`1.1.1[.]1`, inconsistent casing, typos). A more accurate but uncalibrated or brittle model can be the worse product.

## Setup
- Models (frozen): both. E9/E10 are accuracy/reliability evaluations; E7 separately reports the CPU/MPS operational envelope.
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
- Calibration (E10): both confidence streams are badly miscalibrated. SecureBERT has lower ECE than CyNER (0.6520 vs 0.7044), but neither score scale is trustworthy as an analyst triage probability.
- The high-precision threshold requirement is unreachable for both models. SecureBERT's threshold sweep peaks at only 0.2322 precision with 0.3505 recall at threshold 0.80; thresholds 0.95 and 0.99 produce no predictions. CyNER peaks at 0.1557 precision with 0.0860 recall at threshold 0.75, and remains only 0.1460 precision at threshold 0.95.
- At no threshold does CyNER overtake SecureBERT on a useful precision/recall operating point. SecureBERT produces more entity predictions (4007 vs 1815), more strict true positives at threshold 0.0 (897 vs 218), and higher recall (0.3820 vs 0.0928), but the absolute precision remains too low for a high-confidence alerting lane.

## Conclusion → evidence contribution
- Robustness vote: SecureBERT remains the better model by noisy strict F1 because every perturbation preserves a positive gap with a 95% CI excluding 0. The caveat is brittleness: SecureBERT loses far more absolute F1 under random casing and keyboard typos, while CyNER starts much lower and therefore has less absolute score to lose.
- High-precision analyst triage: no model is acceptable from raw confidence thresholding. If the product needs precision>=0.90, this study supports adding an external calibration/abstention layer or human-review rule rather than trusting either frozen model's `score`.
- Direction 07 evidence contribution: SecureBERT leads because it keeps the positive noisy-F1 gap under every perturbation and is slightly less miscalibrated. The vote is caveated: raw confidence is not deployment-ready for either model, so SecureBERT's advantage is robustness and relative calibration, not a usable high-precision triage threshold.
