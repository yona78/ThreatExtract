# Product Requirements Document — ThreatExtract

**Version:** 1.00
**Status:** Approved
**Owner:** yona78 (ML Engineer)
**Last updated:** 2026-06-23

---

## 1. Overview

ThreatExtract is an on-prem web application that extracts named entities from
cyber-threat-intelligence reports. A security analyst uploads a plain-text report
and receives a structured breakdown of the entities a Named Entity Recognition
(NER) model found in it — threat actors, malware, indicators (IPs, domains,
hashes), CVEs, tooling, and more. It exists because analysts triage unstructured
reports by hand, and because the underlying intelligence is sensitive enough that
it cannot leave the organization's network — so the system runs **fully offline**.

## 2. Context and problem statement

### 2.1 The problem
Threat-intel reports arrive as prose. Pulling the actionable entities (which APT,
which malware family, which IOCs) out of a multi-page report is slow, manual, and
error-prone. The cost of leaving it unsolved is analyst hours spent reading and
the risk of missing an indicator. Existing cloud NER services are off-limits
because the reports are confidential and the deployment target is air-gapped.

### 2.2 Market and landscape
Cloud NLP APIs (AWS Comprehend, Google NL) solve generic NER but are SaaS — they
cannot be used on confidential, on-prem data. Open cybersecurity NER models
(SecureBERT-NER, CyNER) exist on Hugging Face but ship as bare model weights with
no serving layer. ThreatExtract fills the gap: a polished, **offline** serving
layer that is **model-agnostic**, so the organization can adopt whichever model
its evaluation (Fork 1) selected and swap it later without re-engineering.

### 2.3 Target users
- **Primary — SOC analyst / threat-intel researcher.** Goals: quickly surface the
  entities in a report and export them. Pain points: manual reading, copy-paste
  into spreadsheets. Technical sophistication: comfortable with a browser and
  `.txt` exports; not expected to write code or run Python.
- **Secondary — ML/platform engineer (operator).** Goals: deploy the container
  on-prem, point it at the chosen model, swap models over time. Comfortable with
  Docker and environment variables.

## 3. Goals and success metrics

### 3.1 Goals
- **G1:** Serve **any** Hugging Face `token-classification` model with no code
  change — measured by swapping the model via one environment variable.
- **G2:** Run **fully offline** — measured by the container completing a full
  upload→extract cycle with no outbound network access.
- **G3:** Accept only valid plain-text uploads — measured by rejecting a binary
  file renamed to `.txt` via content inspection, not extension alone.
- **G4:** Handle reports larger than the model's token limit without dropping
  entities — measured by correct extraction on inputs exceeding 512 tokens.
- **G5:** Be reproducible and verifiable — measured by a green CI pipeline
  (lint + ≥85% logic-layer test coverage) on every change.

### 3.2 Non-goals
- Not benchmarking or selecting the model — that is Fork 1, already complete.
- Not training or fine-tuning models.
- Not multi-file or batch processing — one file at a time, per the assignment.
- Not multi-tenant: no authentication, accounts, or RBAC (single-team on-prem tool).
- Not a public/internet-facing service.

### 3.3 Key Performance Indicators (KPIs)

| KPI | Definition | Target | Measurement method |
|-----|-----------|--------|--------------------|
| Offline integrity | Network calls during a request | 0 | `TRANSFORMERS_OFFLINE=1` + `local_files_only`; verified by in-container model load with no network |
| Validation precision | Disallowed files rejected | 100% of non-text / non-`.txt` | Unit tests in `tests/test_file_utils.py` |
| Logic-layer coverage | Statement coverage of `src/` (excl. GUI) | ≥ 85% | `pytest-cov` gate in CI |
| Single-report latency | End-to-end extract for a typical (~1–2k char) report on CPU | < 5 s | Latency metric shown in the UI |
| Model-swap effort | Files changed to serve a different model | 0 | Change `MODEL_PATH` only |

### 3.4 Acceptance criteria
- All P0 user stories (§4.1) pass their acceptance tests.
- The Docker image builds and runs with no network access and serves the UI.
- CI is green: Ruff (zero violations), Black (clean), pytest with the ≥85% gate.
- Documentation complete: README, this PRD, the NER-engine PRD, and the design spec.

## 4. Requirements

### 4.1 User stories

> **US-1.** As a SOC analyst, I want to upload a `.txt` report and see the entities
> it contains, so that I don't have to read the whole document by hand.
> **Acceptance criteria:**
> - Given a valid UTF-8 `.txt` report, when I upload it, then a table of
>   **Class Name / Identified Entity** rows is shown.
> - Given the report, when extraction finishes, then the original text is shown
>   with each entity highlighted and labeled by class.
> **Priority:** P0

> **US-2.** As a SOC analyst, I want to download the results as CSV, so that I can
> share or further process them.
> **Acceptance criteria:**
> - Given results are displayed, when I click download, then a CSV with the
>   Class Name and Identified Entity columns is produced.
> **Priority:** P0

> **US-3.** As a SOC analyst, I want bad files rejected with a clear message, so
> that I trust the tool and am not misled by garbage output.
> **Acceptance criteria:**
> - Given a non-`.txt` file or a binary renamed to `.txt`, when I upload it, then
>   the system rejects it with an explanatory message and no results.
> **Priority:** P0

> **US-4.** As a SOC analyst, I want progress feedback on long reports, so that I
> know the tool is working and not frozen.
> **Acceptance criteria:**
> - Given a report that is split into multiple chunks, when it is processing, then
>   a progress bar advances per chunk.
> **Priority:** P1

> **US-5.** As an operator, I want to swap the served model without touching code,
> so that I can adopt a new/better model.
> **Acceptance criteria:**
> - Given a downloaded model in the cache, when I set `MODEL_PATH` to it and
>   restart, then the app serves that model and the sidebar reflects its classes.
> **Priority:** P0

### 4.2 Functional requirements
- **FR-1:** The system shall accept exactly one `.txt` file per request.
- **FR-2:** The system shall validate uploads by extension, by content MIME type
  (text family, via libmagic), and by UTF-8 decodability, rejecting any that fail.
- **FR-3:** The system shall read the entire upload into memory before processing.
- **FR-4:** The system shall split text exceeding the model's token limit into
  overlapping chunks and remap entity offsets back to the full document.
- **FR-5:** The system shall display results as a two-column table (Class Name,
  Identified Entity) and offer the same data as a CSV download.
- **FR-6:** The system shall read its model from a local directory given by
  `MODEL_PATH`, loading offline, with the entity classes taken from the model's
  own configuration.
- **FR-7:** The system shall show a progress indicator while processing.

### 4.3 Non-functional requirements
- **Performance:** A typical report (≤ 2,000 chars) completes in < 5 s on CPU; the
  app loads the model once per process (cached) and memoizes results per file.
- **Scalability:** Single-user, one file at a time; no concurrency target. Upload
  capped (default 10 MB; hard cap 25 MB in the server config).
- **Reliability:** Invalid input, missing model cache, and inference errors are
  caught and surfaced as user-facing messages; the app never crashes the session.
- **Security:** Fully offline; content-sniffed file validation; no execution of
  uploaded content; no secrets in the image. No authentication (on-prem, single team).
- **Usability:** Browser UI; clear empty/working/error/result states; results
  exportable as CSV.
- **Observability:** The UI shows the active model, its classes, token limit, and
  per-report latency. The container exposes a health endpoint.
- **Maintainability:** Ruff with zero violations; Black formatting; ≥ 85% coverage
  on the logic layer; the engine is a pure, model-agnostic module with no UI code.

## 5. User experience

The app is a single page. Key flows:
1. **Land → upload.** The analyst sees the engine status (model, classes) in the
   sidebar and an upload zone. They drop a `.txt` report.
2. **Validate → process.** On a valid file the app toasts success, shows file
   metrics, and runs extraction with a per-chunk progress bar. An invalid file is
   rejected inline with a specific message.
3. **Review → export.** Results appear as the Class Name / Identified Entity
   table, a per-class summary, the latency, a CSV download, and a highlighted view
   of the source document with color-coded inline entities.

Errors are recoverable: a rejected or empty file leaves the app ready for another
upload; a missing model cache shows guidance to run the pre-build download step.

## 6. Assumptions, dependencies, and constraints

### 6.1 Assumptions
- The chosen NER model is a Hugging Face `token-classification` model with a fast
  tokenizer (offset mapping available) and a populated `id2label`.
- Reports are UTF-8 plain text in English (the models' training language).
- The deployment host is on-prem/air-gapped; the browser is modern (supports
  `color-mix`, used by the theme).

### 6.2 Dependencies
- A Hugging Face model downloaded **before** image build via `download_model.py`
  into `./model_cache/` (default `CyberPeace-Institute/SecureBERT-NER`).
- `libmagic` system library (Docker: `libmagic1`; macOS dev: `brew install libmagic`).
- Docker / Docker Compose for the on-prem deployment.
- Pinned Python stack: streamlit, transformers 4.46.3, torch 2.5.1 (CPU),
  huggingface_hub, tokenizers, sentencepiece, python-magic, pandas.

### 6.3 Constraints
- CPU-only deployment target (no GPU assumed).
- Models enter the image solely via `COPY model_cache/`; no build- or run-time
  downloads.
- One full day of work, per the assignment brief.

### 6.4 Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Chosen model lacks a fast tokenizer (no offset mapping) | Low | Med | Document the requirement; both reference models (SecureBERT-NER, CyNER) have fast tokenizers |
| Large model cache bloats the image | Med | Low | `download_model.py` skips the duplicate `pytorch_model.bin` when safetensors exist |
| Model tags entity runs piecewise (e.g. `CVE`/`-`/`2021`) | High | Med | Engine stitches adjacent same-class spans into one entity |

## 7. Timeline and milestones

| Milestone | Date | Deliverables | Definition of done |
|-----------|------|-------------|-------------------|
| M0 — Design | 2026-06-21 | Approved design spec | Spec in `docs/superpowers/specs/`, approved |
| M1 — Core | 2026-06-22 | Engine + validation + UI + Docker | App boots, container loads model offline |
| M2 — Hardening | 2026-06-23 | Tests, CI, Ruff, ≥85% coverage, PRDs | CI green; this PRD + NER-engine PRD written |

## 8. Open questions

- **Q1:** Should the app expose a headless REST endpoint for programmatic use, in
  addition to the UI? — Owner: operator. Target: post-submission, if needed.
- **Q2:** Should multiple models be selectable from the UI at runtime (vs. one per
  container)? — Owner: operator. Target: post-submission.

## 9. Appendix

- **Glossary:** *IOC* — indicator of compromise. *NER* — named entity recognition.
  *DNRTI* — the dataset whose label set the default model emits. *BIO* — the
  Begin/Inside/Outside token-tagging scheme.
- **Related documents:** `docs/PRD_ner_engine.md` (the NER mechanism in depth);
  `docs/superpowers/specs/2026-06-21-threatextract-app-design.md` (design).
