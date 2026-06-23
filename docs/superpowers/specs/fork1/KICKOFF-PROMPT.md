# Fork 1 — Implementation Kickoff Prompt (agent handoff)

Paste everything below into the implementing agent.

---

You are a coding agent implementing **Fork 1** of the **ThreatExtract** repo (`/Users/yona/ThreatExtract`, branch `codex/fork1-deep-research`). Fork 1 is a **data-centric research study** that compares two **frozen, black-box** cyber-threat NER models — **SecureBERT-NER** vs **CyNER** — on the **DNRTI** dataset, for an **offline on-prem** product.

## Source of truth — read these first, in this order
1. `docs/superpowers/specs/fork1/00-overview.md` — thesis, direction map, device strategy, decision rule.
2. `docs/superpowers/specs/2026-06-22-fork1-data-centric-ner-eval-design.md` — the **implementation appendix**: file structure, dependencies, and bite-sized **TDD tasks with real code** (Phases 0–7).
3. `docs/superpowers/specs/fork1/01..08-*.md` — one brief per research direction; each has **Findings** and **Conclusion** sections you must fill from evidence.

The appendix is authoritative for *how to build*; the briefs are authoritative for *what each direction means and how it informs the choice*. Do not duplicate code — implement it once in `src/fork1/` per the appendix.

## Non-negotiable constraints
- **Models are frozen black boxes.** No training, fine-tuning, quantization, or weight edits — only read-only inference + data/eval manipulation.
- **Fully offline.** All inference uses `local_files_only=True` against `model_cache/fork1` (both models are already cached there). No network at run time.
- **Reuse, don't rewrite.** Move existing logic from `benchmark.py` into `src/fork1/`; keep `benchmark.py` working as a thin shim. The **6 existing tests in `tests/test_benchmark.py` must stay green** (regression gate).
- **Runtime image stays lean.** New research deps (`seqeval==1.2.2`, `nervaluate==0.2.0`, `sacremoses==0.1.1`, `matplotlib==3.9.2`, `numpy==2.1.3`) go in a new `requirements-research.txt`, never in the Docker runtime `requirements.txt`.
- **Device:** M4 GPU = **MPS** (`--device mps`); M4 CPU = `--device cpu`. First verify `torch.backends.mps.is_available()` (a prior run logged it false). Run accuracy experiments on MPS but confirm a one-time **CPU↔MPS parity check** (strict-F1 delta <0.005). For the on-prem study (E7) measure **both**, and treat **CPU as the deployment-relevant number** (the Docker image ships CPU-only torch); MPS is the local-dev ceiling.
- **TDD + frequent commits.** Every task = write failing test → run (FAIL) → implement → run (PASS) → commit. Work on the current branch (or a fresh one); never commit to `main`.
- **No new comparator models.** Do **not** add CyberNER or any CyberNER-derived checkpoint — it contains DNRTI and would leak.

## Critical knowledge (so you don't re-derive or get it wrong)
- **Models:** `CyberPeace-Institute/SecureBERT-NER` (RoBERTa; **APTNER 21-label ontology**, 40 BIO labels) and `AI4Sec/cyner-xlm-roberta-base` (XLM-R; **5 labels**: System/Organization/Vulnerability/Malware/Indicator).
- **DNRTI test split:** 664 sentences, 17,716 tokens, **2,348 collapsed gold spans, 13 labels**, CoNLL token/tag format. Ensure the data is present (`benchmark.py` supports `--download-dnrti`, or place under `data/dnrti`). 16 malformed tag-only lines are expected and skipped.
- **The structural-bias finding (the study's backbone):** SecureBERT is advantaged three ways — cyber-domain pretraining, fine-tuning on **APTNER whose ontology ≈ DNRTI** (and whose source APT reports plausibly overlap DNRTI), and a PDF label-mapping that is **near-lossless for SecureBERT but compresses CyNER**. The study must **separate capability from this bias** (Direction 05), not just report raw F1.
- **Metric gotcha:** the assignment's label mapping is **one-to-many** (e.g., CyNER `System` → DNRTI `{OffAct, Way}`), so raw `seqeval`/`nervaluate` cannot score it. Implement the **SemEval 4-scheme scorer (strict/exact/partial/type) on spans with set-membership labels** per appendix Task 0.2; use `seqeval` only as a 1:1-label cross-check.
- **Baseline sanity:** with single-space detok + PDF mapping, strict-F1 must reproduce the legacy exact numbers **SecureBERT ≈ 0.282, CyNER ≈ 0.105** within ±0.01. If not, stop and debug the instrument before proceeding.
- **Decision rule (everywhere):** model A beats B on a config iff the **paired-bootstrap 95% CI of the strict-F1 gap excludes 0** (corroborated by McNemar p<0.05); else "tie".

## Workflow
1. Read the three sources above. Confirm the model cache and DNRTI data are present; verify/enable MPS.
2. **Execute Phase 0 first** (appendix Tasks 0.1–0.4): package extraction (6 legacy tests green) → SemEval 4-scheme scorer → bootstrap+McNemar → alignment. This is the foundation for everything.
3. Then work the directions in order (Phases 1–6 ≈ Directions 02–07). For each experiment: implement per the appendix tasks (TDD), run it **offline**, write the artifact to `reports/fork1/`, **then write that direction's Findings + Conclusion sections into the matching brief** (`fork1/0X-*.md`).
4. **Phase 7 / Direction 08 last:** aggregate all JSONL into the master table, apply the Direction-05 bias adjustment, rewrite `reports/fork1/benchmark_summary.md` as the publication-style report, and wire `make reproduce` + extended `run_metadata.json`.
5. Commit after every task with a clear message.

## Rules for Findings & Conclusions (the user requires these written, from evidence)
- **Never invent numbers.** Only write results you actually produced; if a run is skipped (e.g., MPS unavailable, APTNER not obtainable offline), say so explicitly.
- **Do not pre-judge the winner.** Each brief states its own evidence and a "vote"; Direction 08 issues the evidence-leader summary, with a **sensitivity statement** ("SecureBERT wins in K/N configs; ranking flips only under X") and the bias caveat.
- Always report the 95% CI and the flip flag for every head-to-head.

## Done =
All 11 experiments run offline; every brief's Findings/Conclusion filled from evidence; `tests/` green (incl. the 6 legacy tests); `make reproduce` regenerates every table and figure; `benchmark_summary.md` states one bias-adjusted evidence-leader summary with its sensitivity statement and the CPU-deployment note.

**Begin now with appendix Task 0.1** (or report any blocker — missing DNRTI data, missing model cache, or MPS status — before starting).
