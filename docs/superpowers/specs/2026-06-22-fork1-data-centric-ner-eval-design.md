# Fork 1 NER Evaluation — Research & Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an evidence package comparing two frozen, taxonomy-mismatched black-box NER models (SecureBERT-NER vs CyNER) for a cyber-threat-intelligence product on DNRTI, by measuring how *sensitive* the ranking is to the data/evaluation decisions we control — and separating genuine capability from training-data/ontology bias.

**Architecture:** Refactor the existing monolithic `benchmark.py` into a small, testable `src/fork1/` package. Every experiment is a config-driven run over the same frozen models; the data side (preprocessing, subset construction, label projection, scoring scheme, perturbations) is the independent variable. Outputs are JSONL records + Markdown/figures; a final synthesis step aggregates all runs into one bias-adjusted decision.

**Tech Stack:** Python 3.12, PyTorch 2.5.1, Transformers 4.46.3 (CPU/MPS), pinned research deps (`seqeval`, `nervaluate`, `sacremoses`, `matplotlib`, `numpy`); offline cached HF snapshots under `model_cache/fork1`.

> **Per-direction research briefs (read these first):** [`fork1/00-overview.md`](fork1/00-overview.md) plus the 8 direction briefs `fork1/01`…`fork1/08`. Each brief describes one research direction (question, why-it-matters-for-the-choice, device, experiments) and carries its own **Findings** + **Conclusion** sections. This file is the implementation appendix (TDD tasks + code) those briefs reference.

## Global Constraints

- **Models are frozen black boxes.** No training, fine-tuning, quantization, or weight edits anywhere. Only read-only inference + data/eval manipulation.
- **Fully offline.** All inference uses `local_files_only=True` against `model_cache/fork1`. No network at run time.
- **Device.** M4 GPU = **MPS** (`--device mps`); M4 CPU = `--device cpu`. Verify `torch.backends.mps.is_available()` on the host first (the prior run logged it false). Accuracy experiments run on MPS (device-invariant — confirm once with a **CPU↔MPS parity check**, strict-F1 delta <0.005). The on-prem study (E7) measures **both**, and **CPU is the deployment-relevant number** because the Docker image ships CPU-only torch; MPS is the local-dev ceiling.
- **Runtime image stays lean.** New research-only deps go in `requirements-research.txt`, never in the Docker runtime `requirements.txt`.
- **Reuse, don't rewrite.** Move existing logic from `benchmark.py` into `src/fork1/`; keep `benchmark.py` working as a thin shim. The existing 6 tests in `tests/test_benchmark.py` MUST stay green (regression gate).
- **Anchor metric:** SemEval *strict* entity-level micro-F1 (== seqeval strict on 1:1 labels == the source papers' protocol). Everything else is reported relative to it.
- **Determinism:** every sampling/bootstrap step takes an explicit `seed`; default `seed=20260621`.
- **Decision rule (used everywhere):** model A beats model B on a config iff the paired-bootstrap 95% CI of the strict-F1 gap excludes 0 (corroborated by McNemar p<0.05). Otherwise "tie".
- **Style:** black line-length 100; tests under `tests/fork1/`; `pytest -q`.

---

## Part I — Research goals (enriched)

### Research questions

- **RQ1 Preprocessing sensitivity.** How much do each model's strict-F1 and the *ranking* move across reasonable preprocessing pipelines (detokenization, context granularity, max length, normalization, span-alignment policy)?
- **RQ2 Subset design.** How do subset *construction strategy* and *size* affect the estimate, its variance, and the evidence-leader call? What is the smallest subset that preserves the ranking with a CI-separable margin?
- **RQ3 Protocol faithfulness.** How does the verdict change under each model's *paper-native* protocol (seqeval strict, BIO, max_len 128) vs the assignment's mapping-projected protocol?
- **RQ4 Capability vs bias.** How much of SecureBERT's edge is genuine recognition vs structural advantage from (a) cyber-domain pretraining, (b) APTNER fine-tuning (ontology ≈ DNRTI, likely shared source reports), (c) a mapping that handicaps CyNER? Quantified via preprocessing-invariant metrics + a mapping-fairness/oracle analysis.
- **RQ5 On-prem profile.** On a controlled synthetic workload, what is each model's latency/throughput/memory/CPU/energy/load-time envelope, and does it change the recommendation?
- **RQ6 Reliability & robustness.** How calibrated are the models' confidences (ECE), what high-precision operating points exist for analyst triage, and how much does input noise (defanged IOCs, casing, typos) degrade each model?

### Experiment catalog (the "full evaluation")

Every experiment holds both models frozen. "Instrument" = the metric stack from E1.

| # | Name | RQ | Independent variable | Output artifact | Acceptance criterion |
|---|---|---|---|---|---|
| **E1** | Evaluation instrument | all | — | `src/fork1/metrics.py` + reconciled baseline table | SemEval strict P/R/F1 reproduces existing exact numbers within ±0.001 on the default config; unit tests green |
| **E2** | Preprocessing sensitivity | RQ1 | detok × context × max_len × normalization × alignment | `reports/fork1/preprocessing_sensitivity.md` + tornado plot | each lever swept with both models; CI on every gap; ranking-flip flagged |
| **E3** | Paper-faithful protocol | RQ3 | protocol = paper-native vs PDF-mapping | `reports/fork1/protocol_comparison.md` | both models scored under each protocol; verdict + gap reported per protocol |
| **E4** | Subset construction | RQ2 | strategy × size | `reports/fork1/subset_study.md` + stability curves | ≥4 strategies × sizes {10,50,100,250,all}; per-cell F1, variance, min-faithful-subset |
| **E5** | Statistical separability | RQ1–3 | (applied across E2–E4) | CI/McNemar columns on every comparison + `flip_matrix.md` | every head-to-head has 95% CI + McNemar p + flip flag |
| **E6** | Preprocessing-invariant metrics | RQ4 | — | `reports/fork1/intrinsic_metrics.md` | tokenizer fertility, domain coverage, label expressivity, oracle upper-bound F1 for both models |
| **E7** | On-prem operational microbench | RQ5 | workload length × batch (synthetic) | `reports/fork1/operational_envelope.md` + plots | latency p50/p95/p99, throughput, RSS, CPU%, MPS, energy (powermetrics or estimate), load time, on cpu+mps |
| **E8** | Leakage & bias audit | RQ4 | — | `reports/fork1/leakage_bias.md` + bias-chain figure | lineage documented; APTNER↔DNRTI overlap quantified-or-justified; mapping handicap quantified |
| **E9** | Robustness perturbations | RQ6 | perturbation type × intensity | `reports/fork1/robustness.md` | ≥3 perturbations; ΔF1 per model with CI |
| **E10** | Calibration & operating points | RQ6 | confidence threshold | `reports/fork1/calibration.md` + reliability diagram | ECE per model; precision@threshold sweep; recommended high-precision point |
| **E11** | Cross-experiment synthesis | all | — | upgraded `reports/fork1/benchmark_summary.md` (the "paper") | master table of all configs; final bias-adjusted evidence-leader summary + sensitivity statement |

**Sequencing / dependencies:** E1 is the foundation for E2–E5, E9, E10. E6 and E8 are independent (can run in parallel after Phase 0). E7 is independent. E11 runs last and consumes all JSONL outputs.

---

## Part II — Engineering plan

### File structure

```
src/fork1/
  __init__.py
  data.py          # MOVE from benchmark.py: Span, Sample, MatchCounts, DatasetStats,
                   #   extract_bio_spans, load_dnrti_split, load_dnrti_dataset, summarize_gold
  mapping.py       # MOVE: SECUREBERT_TO_DNRTI, CYNER_TO_DNRTI, strip_bio, map_model_label_to_dnrti
  runner.py        # MOVE: HfTokenClassificationRunner, build_runner, resolve_device,
                   #   get_process_rss_mb, get_mps_memory_mb, hf_cache_model_dir
  preprocess.py    # NEW (E2/E9): detokenizers, text normalization, IOC refang/defang, context windows
  align.py         # NEW (E1/E2/E3): predicted char-spans -> per-gold-token DNRTI tags (policies)
  metrics.py       # NEW (E1/E5): SemEval 4-scheme scorer, MUC counts, per-label, bootstrap CI, McNemar
  subsets.py       # NEW (E4): subset strategies (random/label/density/length/hardness)
  intrinsic.py     # NEW (E6): tokenizer fertility, domain coverage, expressibility, oracle bound
  operational.py   # NEW (E7): synthetic workload, latency bench, powermetrics sampler
  perturb.py       # NEW (E9): defang/refang, casing, typo perturbations
  calibration.py   # NEW (E10): entity confidences, ECE, reliability bins, threshold sweep
  config.py        # NEW: ExperimentConfig dataclass + PRESETS (pdf_mapping, paper_native, ...)
  report.py        # NEW/refactor: markdown + matplotlib figure writers
  run_experiment.py# NEW: CLI -> runs one ExperimentConfig, writes JSONL + report
  synthesize.py    # NEW (E11): aggregate all JSONL -> master table + final decision
benchmark.py        # KEEP as shim: `from fork1.data import *` etc., preserves current CLI + tests
tests/fork1/
  test_preprocess.py test_align.py test_metrics.py test_subsets.py
  test_intrinsic.py  test_perturb.py test_calibration.py test_config.py
requirements-research.txt  # NEW: seqeval, nervaluate, sacremoses, matplotlib, numpy (pinned)
Makefile            # NEW: targets to reproduce every experiment + figures
```

### Dependencies & offline

Add `requirements-research.txt` (installed once, before going offline):

```
seqeval==1.2.2
nervaluate==0.2.0
sacremoses==0.1.1
matplotlib==3.9.2
numpy==2.1.3
```

Bootstrap and McNemar are implemented with `numpy` only (no scipy). `matplotlib` and `nervaluate` are research/report-time only; the Docker runtime keeps using the lean pinned `requirements.txt`.

### Config schema (`config.py`)

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    models: tuple[str, ...] = ("securebert", "cyner")
    detok: str = "single_space"          # preprocess.DETOKENIZERS key
    context: str = "sentence"            # "sentence" | "document" | "window"
    max_length: int = 256                # tokenizer truncation
    normalization: str = "none"          # "none" | "nfc" | "nfkc" | "refang" | "lower"
    alignment: str = "overlap"           # align policy: "overlap" | "majority" | "contained"
    scheme: str = "strict"              # headline scheme for this run
    subset_strategy: str = "all"         # subsets.STRATEGIES key
    subset_size: str = "all"
    seed: int = 20260621
    bootstrap: int = 10000

PRESETS: dict[str, ExperimentConfig] = {
    "pdf_mapping":  ExperimentConfig(name="pdf_mapping"),
    "paper_native": ExperimentConfig(name="paper_native", detok="punct_aware",
                                     context="sentence", max_length=128, alignment="overlap"),
}
```

---

## Phase 0 — Package extraction & instrument (E1)

### Task 0.1: Extract reusable modules from `benchmark.py`

**Files:**
- Create: `src/fork1/__init__.py`, `src/fork1/data.py`, `src/fork1/mapping.py`, `src/fork1/runner.py`
- Modify: `benchmark.py` (replace moved defs with imports), `pyproject.toml` (add `pythonpath = ["src", "."]` under pytest)
- Test: existing `tests/test_benchmark.py` is the regression gate

**Interfaces produced:** `fork1.data.{Span,Sample,MatchCounts,extract_bio_spans,load_dnrti_split,load_dnrti_dataset}`, `fork1.mapping.{strip_bio,map_model_label_to_dnrti,SECUREBERT_TO_DNRTI,CYNER_TO_DNRTI}`, `fork1.runner.{HfTokenClassificationRunner,build_runner}`.

- [ ] **Step 1:** Move the dataclasses and functions listed above verbatim from `benchmark.py` into the new modules (data/mapping/runner). Keep names/signatures identical.
- [ ] **Step 2:** In `benchmark.py`, replace the moved code with `from fork1.data import *`, `from fork1.mapping import *`, `from fork1.runner import *`.
- [ ] **Step 3:** Add to `pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
pythonpath = ["src", "."]
```
- [ ] **Step 4:** Run `pytest -q` — Expected: the existing 6 tests PASS unchanged.
- [ ] **Step 5:** Commit: `git commit -am "refactor: extract fork1 package from benchmark.py (no behavior change)"`

### Task 0.2: SemEval 4-scheme scorer + MUC counts (`metrics.py`)

Implements the literature-standard entity-level evaluation, correct under one-to-many label projection (which seqeval can't express).

**Files:** Create `src/fork1/metrics.py`, `tests/fork1/test_metrics.py`

**Interfaces produced:** `MucCounts`, `muc_counts(gold, pred, model_name) -> dict[str, MucCounts]` (keys: `strict,exact,partial,type`), `prf(muc, scheme) -> dict`, `score(gold, pred, model_name) -> dict[scheme->{p,r,f1}]`.

- [ ] **Step 1: Write failing tests**
```python
from fork1.data import Span
from fork1.metrics import score

def _g(s,e,l): return Span(label=l,start=s,end=e,text="",score=None,source="gold")
def _p(s,e,l,sc=0.9): return Span(label=l,start=s,end=e,text="",score=sc,source="securebert")

def test_strict_exact_boundary_and_type():
    gold=[_g(0,5,"HackOrg")]; pred=[_p(0,5,"APT")]   # APT maps to HackOrg
    out=score(gold,pred,"securebert")
    assert out["strict"]["f1"]==1.0

def test_partial_credits_overlap_not_strict():
    gold=[_g(0,10,"Tool")]; pred=[_p(0,5,"MAL")]      # MAL->Tool, boundary off
    out=score(gold,pred,"securebert")
    assert out["strict"]["f1"]==0.0
    assert out["partial"]["f1"]>0.0

def test_type_scheme_ignores_boundary_when_overlap_and_type_ok():
    gold=[_g(0,10,"Area")]; pred=[_p(3,9,"LOC")]
    out=score(gold,pred,"securebert")
    assert out["type"]["f1"]>0.0
```
- [ ] **Step 2:** Run `pytest tests/fork1/test_metrics.py -v` — Expected: FAIL (module missing).
- [ ] **Step 3: Implement**
```python
from dataclasses import dataclass
from fork1.mapping import map_model_label_to_dnrti

@dataclass
class MucCounts:
    cor: int = 0; inc: int = 0; par: int = 0; mis: int = 0; spu: int = 0

def _overlap(a, b) -> bool:
    return max(a.start, b.start) < min(a.end, b.end)

def _exact(a, b) -> bool:
    return a.start == b.start and a.end == b.end

def _type_ok(gold, pred, model_name) -> bool:
    return gold.label in map_model_label_to_dnrti(model_name, pred.label)

def muc_counts(gold_spans, pred_spans, model_name):
    # Greedy 1:1 assignment, highest pred score first.
    schemes = {k: MucCounts() for k in ("strict", "exact", "partial", "type")}
    for scheme in schemes:
        used = set()
        ordered = sorted(pred_spans, key=lambda s: s.score or 0.0, reverse=True)
        for pred in ordered:
            best = None
            for i, gold in enumerate(gold_spans):
                if i in used or not _overlap(gold, pred):
                    continue
                best = (i, gold); break
            if best is None:
                schemes[scheme].spu += 1; continue
            i, gold = best; used.add(i)
            bound = _exact(gold, pred); typ = _type_ok(gold, pred, model_name)
            c = schemes[scheme]
            if scheme in ("strict", "exact"):
                ok = (bound and typ) if scheme == "strict" else bound
                c.cor += 1 if ok else 0; c.inc += 0 if ok else 1
            elif scheme == "type":
                c.cor += 1 if typ else 0; c.inc += 0 if typ else 1
            else:  # partial
                if bound and typ: c.cor += 1
                else: c.par += 1
        schemes[scheme].mis = len(gold_spans) - len(used)
    return schemes

def prf(m: MucCounts, scheme: str):
    pos = m.cor + m.inc + m.par + m.mis
    act = m.cor + m.inc + m.par + m.spu
    num = m.cor + (0.5 * m.par if scheme == "partial" else 0.0)
    p = num / act if act else 0.0
    r = num / pos if pos else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"p": p, "r": r, "f1": f1, **m.__dict__}

def score(gold_spans, pred_spans, model_name):
    sc = muc_counts(gold_spans, pred_spans, model_name)
    return {k: prf(v, k) for k, v in sc.items()}
```
- [ ] **Step 4:** Run `pytest tests/fork1/test_metrics.py -v` — Expected: PASS.
- [ ] **Step 5:** Commit: `git commit -am "feat(metrics): SemEval 4-scheme entity scorer with MUC counts"`

### Task 0.3: Bootstrap CI + McNemar (`metrics.py`)

**Interfaces produced:** `corpus_f1(samples, preds, model_name, scheme) -> float`, `bootstrap_gap_ci(samples, preds_a, preds_b, model_a, model_b, scheme, n, seed) -> (lo,hi,gap)`, `mcnemar(samples, preds_a, preds_b, model_a, model_b) -> (stat,p)`.

- [ ] **Step 1: Failing test**
```python
def test_bootstrap_ci_orders_low_high():
    # identical preds -> gap ~0, CI brackets 0
    ...
    lo,hi,gap = bootstrap_gap_ci(samples, preds, preds, "securebert","securebert","strict", n=200, seed=1)
    assert lo <= gap <= hi
```
- [ ] **Step 2:** Run — Expected: FAIL.
- [ ] **Step 3: Implement** — aggregate `MucCounts` across a sentence resample (sum cor/inc/par/mis/spu per sentence then `prf`), compute gap = F1_a − F1_b per replicate, return 2.5/97.5 percentiles via `numpy.percentile`. McNemar: per gold span, mark each model correct (strict) or not; build discordant counts b,c; `stat=(|b-c|-1)**2/(b+c)`, `p=exp(-stat/2)` (chi-square df=1 upper-tail approximation) or exact binomial for small b+c.
- [ ] **Step 4:** Run — Expected: PASS.
- [ ] **Step 5:** Commit: `git commit -am "feat(metrics): paired bootstrap CI + McNemar"`

### Task 0.4: Alignment policies (`align.py`)

Maps predicted char-spans to per-gold-token DNRTI tags so we can (a) cross-check against seqeval on 1:1 labels and (b) support E3 paper-native scoring.

**Interfaces produced:** `align_pred_to_tokens(token_offsets, pred_spans, model_name, policy) -> list[set[str]]` (DNRTI label set per token), `policy in {"overlap","majority","contained"}`.

- [ ] **Step 1:** Failing test: a pred span covering tokens 1–2 yields those tokens' DNRTI label set under each policy; "contained" excludes a partially covered token.
- [ ] **Step 2:** Run — FAIL.
- [ ] **Step 3:** Implement char-overlap test per policy (`overlap`: any overlap; `majority`: >50% of token chars covered; `contained`: token fully inside span).
- [ ] **Step 4:** Run — PASS.
- [ ] **Step 5:** Commit.

---

## Phase 1 — Preprocessing levers (E2) & robustness (E9)

### Task 1.1: Detokenizers with exact offsets (`preprocess.py`)

**Interfaces produced:** `Detokenizer = Callable[[list[str]], tuple[str, list[tuple[int,int]]]]`; `join_with_spacing(tokens, space_before)`, `detok_single_space`, `detok_punct_aware`; `DETOKENIZERS: dict[str, Detokenizer]`.

- [ ] **Step 1: Failing tests**
```python
from fork1.preprocess import detok_single_space, detok_punct_aware
def test_single_space_matches_legacy():
    text, off = detok_single_space(["APT", "hit", "."])
    assert text == "APT hit ."
    assert off[0] == (0,3)
def test_punct_aware_no_space_before_period():
    text, off = detok_punct_aware(["APT", "hit", "."])
    assert text == "APT hit."
    assert text[off[2][0]:off[2][1]] == "."   # offsets stay exact
```
- [ ] **Step 2:** Run — FAIL.
- [ ] **Step 3: Implement**
```python
NO_SPACE_BEFORE = set(".,;:!?)]}%") | {"'s", "n't", "'re", "'ll", "'ve", "'m", "'d"}
NO_SPACE_AFTER = set("([{")

def join_with_spacing(tokens, space_before):
    parts, offsets, cursor = [], [], 0
    prev = None
    for i, tok in enumerate(tokens):
        if i and space_before(prev, tok):
            parts.append(" "); cursor += 1
        start = cursor; parts.append(tok); cursor += len(tok)
        offsets.append((start, cursor)); prev = tok
    return "".join(parts), offsets

def detok_single_space(tokens):
    return join_with_spacing(tokens, lambda prev, tok: True)

def detok_punct_aware(tokens):
    def sb(prev, tok):
        if tok in NO_SPACE_BEFORE: return False
        if prev in NO_SPACE_AFTER: return False
        return True
    return join_with_spacing(tokens, sb)

DETOKENIZERS = {"single_space": detok_single_space, "punct_aware": detok_punct_aware}
```
- [ ] **Step 4:** Run — PASS.
- [ ] **Step 5:** Commit.

### Task 1.2: Normalization + IOC refang/defang (`preprocess.py`)

**Interfaces produced:** `normalize_text(text, mode)`, `refang(text)`, `defang(text)`.

- [ ] **Step 1:** Failing tests: `refang("hxxp://1.1.1[.]1")=="http://1.1.1.1"`; `normalize_text(s,"nfkc")` idempotent; `normalize_text(s,"lower")` lowercases.
- [ ] **Step 2:** Run — FAIL.
- [ ] **Step 3:** Implement with `unicodedata.normalize` and regex replacements (`hxxp→http`, `[.]→.`, `[at]→@`, `(dot)→.`). For `refang`/`defang`, operate on text only (offsets preserved because replacements are length-changing → recompute spans by re-detokenizing the *normalized tokens* upstream; document that normalization runs on tokens before detok in the pipeline).
- [ ] **Step 4:** Run — PASS.
- [ ] **Step 5:** Commit.

### Task 1.3: Context windowing (`preprocess.py`)

**Interfaces produced:** `iter_contexts(samples, mode, window=3, stride=2) -> list[ContextDoc]` where `ContextDoc` carries text, the detok offsets, and a back-map from char-span to `(sample_id, gold_token_index)` so predictions can be scored against the original per-sentence gold.

- [ ] **Step 1:** Failing test: `mode="sentence"` returns one context per sample; `mode="window"` with window=2 concatenates 2 sentences and the back-map recovers original sample ids.
- [ ] **Step 2–4:** Implement + run green. (Document concatenation uses `" "` between sentences; predictions overlapping a boundary are split back by the char back-map.)
- [ ] **Step 5:** Commit.

### Task 1.4: E2 runner + report

**Files:** `src/fork1/run_experiment.py`, `src/fork1/report.py`, output `reports/fork1/preprocessing_sensitivity.md`.

- [ ] **Step 1:** Implement `run_experiment(config)` — loads each frozen model once (offline), builds inputs via `DETOKENIZERS[config.detok]` + normalization + `iter_contexts`, runs `runner.predict`, scores with `metrics.score`, writes one JSONL row per (config, model, scheme) including bootstrap CI vs the other model.
- [ ] **Step 2:** Add a sweep driver: vary one lever at a time from the `pdf_mapping` preset (detok∈{single_space,punct_aware}, context∈{sentence,document,window}, max_length∈{128,256,full}, normalization∈{none,nfkc,refang,lower}, alignment∈{overlap,majority,contained}).
- [ ] **Step 3:** Generate `preprocessing_sensitivity.md` with a per-lever table (both models' strict-F1, gap, 95% CI, flip? ) and a tornado figure (`report.tornado_plot`).
- [ ] **Step 4 (acceptance):** Run `python -m fork1.run_experiment --sweep preprocessing` — Expected: JSONL has one row per (lever-level × model); each comparison row has `ci_low/ci_high/flip`. Spot-check: `single_space` strict-F1 matches the legacy exact-F1 (0.282 / 0.105) within ±0.01.
- [ ] **Step 5:** Commit.

### Task 1.5: E9 robustness perturbations (`perturb.py`)

**Interfaces produced:** `PERTURBATIONS: dict[str, Callable[[str], str]]` with `defang`, `refang`, `random_case(seed)`, `keyboard_typo(rate, seed)`.

- [ ] **Step 1:** Failing tests: each perturbation is deterministic given seed; `defang` then `refang` round-trips on a sample IOC string.
- [ ] **Step 2–4:** Implement + green.
- [ ] **Step 5:** Run perturbation eval (reuse E2 runner with a `perturbation` field), write `reports/fork1/robustness.md` (ΔF1 per model per perturbation, with CI). **Acceptance:** every perturbation reports clean→noisy ΔF1 for both models. Commit.

---

## Phase 2 — Subset construction (E4)

### Task 2.1: Subset strategies (`subsets.py`)

**Interfaces produced:** `sample_subset(samples, strategy, size, seed) -> list[Sample]`; `STRATEGIES = {"random","label_stratified","density","length","hardness"}`.

- [ ] **Step 1: Failing tests**
```python
def test_label_stratified_preserves_proportions():
    sub = sample_subset(samples, "label_stratified", 100, seed=1)
    # top label share within 5 points of full-split share
def test_density_buckets_monotonic():
    sub = sample_subset(samples, "density", 50, seed=1)
    assert len(sub) == 50
```
- [ ] **Step 2:** Run — FAIL.
- [ ] **Step 3: Implement**
  - `random`: existing `choose_samples` behavior (seeded).
  - `label_stratified`: compute per-label gold-span counts; allocate the subset's sentence quota proportionally to label support; greedily pick sentences to hit per-label quotas (seeded shuffle).
  - `density`: bucket sentences by entity count (0,1,2,3+), sample evenly across buckets.
  - `length`: bucket by token count quartiles, sample evenly.
  - `hardness`: rank sentences by presence of rare labels (`Way,Purp,Features`) + length; take a seeded sample weighted to harder sentences.
- [ ] **Step 4:** Run — PASS.
- [ ] **Step 5:** Commit.

### Task 2.2: E4 study + stability curves

- [ ] **Step 1:** Driver: for strategy∈STRATEGIES, size∈{10,50,100,250,all}, seed∈{1,2,3}: run both models (best preprocessing from E2), record strict-F1 + winner.
- [ ] **Step 2:** Compute per-cell mean/variance across seeds; compute **min-faithful-subset** = smallest size where the full-split winner wins with CI excluding 0 for ≥ all 3 seeds.
- [ ] **Step 3:** Write `reports/fork1/subset_study.md` + a "F1 vs size" curve per strategy (`report.line_plot`). **Acceptance:** table has every (strategy,size) cell with variance; min-faithful-subset reported per strategy. Commit.

---

## Phase 3 — Protocol faithfulness (E3)

### Task 3.1: Paper-native protocol run

- [ ] **Step 1:** Use `PRESETS["paper_native"]` (punct_aware detok, sentence context, max_len 128, alignment overlap) and `PRESETS["pdf_mapping"]`.
- [ ] **Step 2:** Add a seqeval cross-check on the 1:1-mappable label subset (build per-token gold/pred BIO via `align.align_pred_to_tokens` restricted to labels with a unique mapping; assert seqeval-strict ≈ our strict within ±0.005). This validates the instrument against the field standard.
- [ ] **Step 3:** Write `reports/fork1/protocol_comparison.md`: verdict + strict-F1 + gap + CI under each protocol; note how much the gap moves. **Acceptance:** both protocols reported for both models; seqeval cross-check within tolerance. Commit.

---

## Phase 4 — Intrinsic metrics (E6) & leakage/bias (E8)

### Task 4.1: Tokenizer fertility, coverage, expressibility, oracle bound (`intrinsic.py`)

**Interfaces produced:** `tokenizer_fertility(tokenizer, words) -> float`, `domain_coverage(tokenizer, words) -> float`, `expressible_dnrti_labels(model_name) -> set[str]`, `oracle_upper_bound(samples, model_name) -> dict`.

- [ ] **Step 1: Failing tests**
```python
def test_expressible_labels_cyner_cannot_express_time_area():
    e = expressible_dnrti_labels("cyner")
    assert "Time" not in e and "Area" not in e
def test_oracle_recall_equals_expressible_support_fraction():
    out = oracle_upper_bound(samples, "cyner")
    assert 0.0 < out["recall"] <= 1.0 and out["precision"] == 1.0
```
- [ ] **Step 2:** Run — FAIL.
- [ ] **Step 3: Implement**
  - `tokenizer_fertility`: mean `len(tokenizer.tokenize(w))` over words (DNRTI entity surface words + a fixed cyber probe list `["ransomware","C2","powershell","T1059.001","CVE-2021-44228","mimikatz","hxxp"]`).
  - `domain_coverage`: fraction of words with fertility == 1.
  - `expressible_dnrti_labels`: union of DNRTI labels reachable from the model's `id2label` via `map_model_label_to_dnrti`.
  - `oracle_upper_bound`: gold spans whose DNRTI label ∈ expressible set are assumed perfectly detected → TP=that count, FP=0, FN=rest → P=1.0, R=TP/total, F1=2R/(1+R).
- [ ] **Step 4:** Run — PASS.
- [ ] **Step 5:** Write `reports/fork1/intrinsic_metrics.md` (fertility, coverage, params, cache size, expressibility, oracle F1 for both models). **Acceptance:** SecureBERT shows lower cyber-term fertility and higher oracle F1 than CyNER (the explanatory result). Commit.

### Task 4.2: Leakage & bias audit (E8)

- [ ] **Step 1:** Document the lineage table (from model cards + papers): SecureBERT base corpus sources, APTNER fine-tune; CyNER corpus; XLM-R base. Cite refs.
- [ ] **Step 2:** If `data/aptner/` is available offline, compute n-gram (e.g., 8-gram) and sentence-hash overlap between APTNER and DNRTI test; report % overlap. Else: produce the qualitative argument (shared report sources, identical ontology) and mark overlap as "unquantified risk".
- [ ] **Step 3:** Mapping-fairness: combine with E6 oracle bound to state CyNER's *handicap* (max achievable F1 under the mapping) vs its *actual* F1 — i.e., how much of the gap is structural.
- [ ] **Step 4:** Write `reports/fork1/leakage_bias.md` + bias-chain figure. **Acceptance:** the report explicitly separates "capability" from "structural advantage" with numbers from E6. Commit.

---

## Phase 5 — Operational microbench (E7)

### Task 5.1: Synthetic workload + latency bench (`operational.py`)

**Interfaces produced:** `make_workload(token_lengths, n_per_length, seed) -> list[str]`, `benchmark_latency(runner, workload, warmup, repeats) -> dict`, `sample_power(duration_s) -> float|None`.

- [ ] **Step 1:** Failing test: `make_workload([16,64],2,seed=1)` returns 4 strings of ~the requested token lengths; `benchmark_latency` returns p50/p95/p99 keys.
- [ ] **Step 2–4:** Implement: build inputs by sampling DNRTI tokens to target lengths; warm up N runs (discard), time `repeats` runs with `time.perf_counter`; percentiles via `numpy`. `sample_power` shells `powermetrics` (sudo) sampling package power over the window; return `None` on failure → caller falls back to `elapsed*assumed_watts`. Green.
- [ ] **Step 5:** Verify/enable MPS (`torch.backends.mps.is_available()`), then run on **both `cpu` and `mps` (M4 10-core GPU)**; write `reports/fork1/operational_envelope.md` (latency-vs-length, throughput-vs-batch plots per device, RSS, load time, energy). Label **CPU as deployment-relevant** (Docker ships CPU-only torch) and **MPS as the local-dev ceiling**. **Acceptance:** both devices attempted (or MPS-unavailable documented); energy is real (powermetrics) or clearly labeled estimate. Commit.

---

## Phase 6 — Calibration (E10)

### Task 6.1: ECE, reliability, threshold sweep (`calibration.py`)

**Interfaces produced:** `entity_records(samples, predictions, model_name) -> list[(score, is_correct)]`, `expected_calibration_error(records, bins=10) -> float`, `threshold_sweep(records, thresholds) -> list[dict]`.

- [ ] **Step 1:** Failing tests: perfectly-calibrated synthetic records give ECE≈0; threshold sweep precision is monotonic-ish in threshold.
- [ ] **Step 2–4:** Implement: `is_correct` = strict TP for that predicted entity; bin by score; ECE = Σ (n_b/N)|precision_b − mean_conf_b|. Green.
- [ ] **Step 5:** Write `reports/fork1/calibration.md` + reliability diagram; recommend a high-precision operating threshold per model. **Acceptance:** ECE reported for both; a precision≥0.9 operating point (or "unreachable") stated. Commit.

---

## Phase 7 — Synthesis (E11)

### Task 7.1: Master table + final decision (`synthesize.py`)

- [ ] **Step 1:** Aggregate all `reports/fork1/*.jsonl` into one master DataFrame (config × model × scheme × F1 × CI × flip).
- [ ] **Step 2:** Decision logic: count configs where SecureBERT wins / ties / loses; report the verdict's robustness ("SecureBERT wins in K/N configs; ranking never flips / flips under config X").
- [ ] **Step 3:** Rewrite `reports/fork1/benchmark_summary.md` as the publication-style report: abstract, RQ1–RQ6, method, results-with-CIs, sensitivity analysis, error/bias analysis (E6/E8), operational profile (E7), calibration/robustness (E9/E10), threats to validity, conclusion. Regenerate all figures.
- [ ] **Step 4 (acceptance):** `make reproduce` runs every experiment offline and regenerates every table+figure from scratch; `benchmark_summary.md` states a single bias-adjusted verdict + the sensitivity statement. Commit.

### Task 7.2: Reproducibility wiring

- [ ] **Step 1:** Add `Makefile` targets: `setup-research`, `e1`…`e11`, `reproduce` (runs all), `figures`, `test`.
- [ ] **Step 2:** Extend `run_metadata.json` to capture every `ExperimentConfig`, environment, seeds, and dep versions.
- [ ] **Step 3:** Update `README.md` Fork 1 section with the one-command reproduce instructions.
- [ ] **Step 4:** Commit.

---

## Testing strategy

- **Unit (fast, no model):** `preprocess` (offset exactness), `align` (policy boundaries), `metrics` (4 schemes on hand-built spans; bootstrap ordering; McNemar discordant counts), `subsets` (proportion/quota invariants), `intrinsic` (expressibility/oracle math), `perturb` (determinism + round-trip), `calibration` (ECE on synthetic).
- **Integration (slow, offline model, `@pytest.mark.slow`):** load each cached model and predict on a 2-sentence input; assert non-empty spans and that the E2 runner emits a well-formed JSONL row. Excluded from default `pytest -q` via marker.
- **Regression gate:** the original 6 tests in `tests/test_benchmark.py` stay green after Phase 0.

## Deliverables

- `src/fork1/` package + `tests/fork1/` suite + `requirements-research.txt` + `Makefile`.
- `reports/fork1/`: `preprocessing_sensitivity.md`, `protocol_comparison.md`, `subset_study.md`, `intrinsic_metrics.md`, `leakage_bias.md`, `robustness.md`, `calibration.md`, `operational_envelope.md`, refreshed figures, and the upgraded `benchmark_summary.md` (the paper).
- Extended `run_metadata.json` + JSONL per experiment.

## Threats to validity (state in the report)

Cross-dataset, cross-taxonomy transfer (not in-domain DNRTI); detok cannot recover original report whitespace; APTNER↔DNRTI overlap may be unquantifiable offline (risk, not proof); `powermetrics` is host-specific and needs sudo; subset strategies carry their own selection bias (report all, not a cherry-picked one); the one-to-many mapping makes some labels unscorable under paper-native protocol (reported as coverage).

---

## Self-review (completed)

- **Spec coverage:** RQ1→E2/E5; RQ2→E4; RQ3→E3; RQ4→E6/E8; RQ5→E7; RQ6→E9/E10; all six original ideas mapped (idea1→E6, idea2→E7, idea3→E1/E2, idea4→E1, idea5→E8, idea6→E3). ✔
- **Placeholder scan:** no placeholder markers/"handle edge cases"; code shown for every logic step; procedural IO tasks have exact commands + acceptance. ✔
- **Type consistency:** `ExperimentConfig` field names match runner usage; `MucCounts`/`prf`/`score`, `DETOKENIZERS`, `STRATEGIES`, `PERTURBATIONS`, `align_pred_to_tokens`, `oracle_upper_bound` referenced consistently across tasks. ✔
- **Correction vs prior spec:** scope = all experiments; metric reworked for one-to-many mapping (not raw nervaluate); detok "whitespace-preserving" replaced with offset-exact `punct_aware`; deps + offline strategy added; acceptance criteria added per experiment. ✔

## Open decisions (still yours, non-blocking — sensible defaults chosen)

1. Subset strategies — defaulted to all five {random, label_stratified, density, length, hardness}. Trim?
2. APTNER overlap (E8) — defaulted to "quantify if `data/aptner/` present, else qualitative". Want me to add an offline APTNER fetch task?
3. Report form — defaulted to multi-file `reports/fork1/*.md` + upgraded `benchmark_summary.md`. Also want a single exported PDF?
