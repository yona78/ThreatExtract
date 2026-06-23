# Direction 02 — Preprocessing Sensitivity

**Question:** Does which model "wins" depend on how we turn DNRTI's pre-tokenized CoNLL into text the model reads?
**Maps to:** RQ1 · Experiment **E2** · Implementation: appendix Phase 1 (Tasks 1.1–1.4).
**Why it matters for the choice:** The models are frozen, but the *input* is ours. Detokenization, context size, normalization, and span alignment all change a transformer's predictions. If the ranking flips across reasonable choices, a single-config conclusion is unsafe. Your own prior report flagged single-space reconstruction as a threat to validity — this direction quantifies it.

## Setup
- Models (frozen): both. Device: MPS (parity-checked vs CPU).
- Levers swept one-at-a-time from the `pdf_mapping` preset:
  - **Detokenization:** single-space (legacy) vs punctuation/quote-aware (offset-exact).
  - **Context:** per-sentence vs document concat vs sliding window.
  - **Max length:** 128 / 256 / full.
  - **Normalization:** none / NFKC / IOC-refang / lowercase.
  - **Span alignment:** overlap / majority / contained.
- Instrument: Direction 01 (strict-F1 + CI + flip flag).

## What to do
1. Build detokenizers, normalization, refang/defang, context windows (Tasks 1.1–1.3).
2. Run the one-lever-at-a-time sweep for both models (Task 1.4).
3. For each lever level: record both models' strict-F1, the gap, 95% CI, and **flip?**.
4. Acceptance: legacy single-space strict-F1 matches 0.282/0.105 within ±0.01; every lever level has a CI'd comparison.

## Outputs
- `reports/fork1/preprocessing_sensitivity.md` + a **tornado plot** (F1 swing per lever).

## Findings (write from evidence)
- Tornado of |ΔF1| per lever, per model: context was the largest input-side sensitivity lever in the completed E2 run (`reports/fork1/preprocessing_sensitivity.jsonl`), with SecureBERT strict-F1 ranging from 0.0125 under document context to 0.2823 under sentence context (swing 0.2698), and CyNER from 0.0051 to 0.1038 (swing 0.0987). The alignment scoring lever also moved absolute F1 substantially: SecureBERT 0.2823 overlap vs 0.5266 majority / 0.4617 contained; CyNER 0.1038 overlap vs 0.2216 majority / 0.1448 contained. Lowercasing hurt both models (SecureBERT 0.1570, CyNER 0.0676). Punctuation-aware detokenization was close to legacy single-space (SecureBERT 0.2767 vs 0.2823; CyNER 0.1030 vs 0.1038). Max length 128/256/full did not move F1 in this sentence-context run.
- Ranking flips: none. Every head-to-head row had a 95% CI excluding 0 in SecureBERT's favor. The baseline single-space gap was +0.1785 with 95% CI [0.1518, 0.2069]; punctuation-aware gap was +0.1736 [0.1481, 0.2011]; lowercased gap was +0.0894 [0.0666, 0.1130]. Document context degraded both models but still favored SecureBERT with gap +0.0074 [0.0024, 0.0136].
- Best-performing preprocessing per model: under raw overlap strict scoring, sentence context with single-space / no normalization / max_length 128-256-full tied at SecureBERT 0.2823 and CyNER 0.1038. The majority/contained alignment rows produce higher absolute F1 by snapping predictions to token boundaries, but they are scoring-policy variants rather than the legacy raw-span baseline; downstream strict comparisons should keep overlap/raw-span scoring for baseline parity.

## Conclusion → evidence contribution
- SecureBERT > CyNER is robust across the tested preprocessing sweep. The absolute scores are fragile to context construction and alignment/scoring policy, but the ranking did not flip in any produced E2 configuration and all reported CIs exclude 0 in SecureBERT's favor.
- The product/evaluation standard should use sentence-level context and raw overlap span scoring for the headline strict metric because that reproduces the legacy exact-F1 sanity check (SecureBERT 0.2823, CyNER 0.1038). Punctuation-aware detokenization is offset-exact and defensible, but in this run it slightly reduced both models; document context should not be used as configured because it truncates most of the concatenated test split and collapses F1.
- Direction 02 evidence contribution: SecureBERT leads in E2, with a sensitivity caveat that preprocessing/scoring policy changes absolute F1 materially even though it did not change the full-test ranking.
