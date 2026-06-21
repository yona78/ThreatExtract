# ThreatExtract — Fork 2 Design Spec

- **Status:** Approved
- **Date:** 2026-06-21
- **Author:** yona78 (with Claude)
- **Scope:** Fork 2 of the "MLE Home Assignment" — the production-grade serving application. Fork 1 (benchmarking SecureBERT-NER vs CyNER on DNRTI) is complete and out of scope here.

## 1. Goal

Build a robust, premium, fully offline application that serves a Named Entity
Recognition (NER) model over a Streamlit web UI, packaged in a Docker container,
with CI/CD and a clean atomic git history.

The single most important architectural constraint: **the application is
model-agnostic.** It must serve *any* Hugging Face `token-classification` model
without code changes — the specific model is selected purely via configuration
(an environment variable). Nothing about a particular model's entity classes is
hardcoded.

## 2. Requirements

From the assignment and the Fork 2 brief:

1. **File handling & security**
   - Accept `.txt` files only.
   - Verify the *real* MIME type (`text/plain`) in the Python backend via content
     sniffing — not just the file extension.
   - Read the full file into memory before processing.
   - Chunk text that exceeds the model's token limit.
   - Keep the user informed with `st.progress` / `st.spinner` during processing.
   - Process one file at a time.

2. **Offline Docker architecture**
   - A pre-build script (`download_model.py`) downloads model weights to
     `./model_cache/` *before* the image is built.
   - The `Dockerfile` only `COPY`s `model_cache/` in — it never downloads at build
     or runtime.
   - The model to load is chosen dynamically via an environment variable, so models
     can be swapped without touching code.

3. **Premium UI (`app.py`)**
   - Highly interactive: toasts, columns, expanders, clean typography.
   - Results table with two columns: **Class Name** and **Identified Entity**.
   - Highlighted-text view of the source document (premium polish).
   - CSV download of results.

4. **CI/CD** — GitHub Actions: checkout → setup Python → lint (flake8 + black) →
   unit tests.

5. **Git workflow** — small, logical, atomic commits on a feature branch.

## 3. Approach

**Chosen: a thin Streamlit `app.py` over a small, pure `src/` package.**

Entity classes are read at runtime from the model's own `config.json`
(`id2label`). Inference runs through Hugging Face's
`pipeline("token-classification", aggregation_strategy=...)`, which merges
sub-word tokens into whole-word entities for any model. The same code path serves
SecureBERT-NER, CyNER, or any other token-classification model.

**Rejected alternatives:**

- *Everything inline in `app.py`* — not unit-testable, fails the production bar.
- *A CyNER-toolkit adapter* (spaCy/Flair/regex special-casing) — violates the
  model-agnostic requirement and adds heavy dependencies for no benefit, since the
  CyNER Hugging Face model is itself a standard token-classification model the
  generic path already handles. YAGNI.

## 4. Architecture

```
ThreatExtract/
├── app.py                     # Streamlit UI (thin; delegates to src/)
├── src/
│   ├── __init__.py
│   ├── config.py              # AppConfig: env-driven settings (MODEL_PATH, limits)
│   ├── ner_engine.py          # model-agnostic load + chunk + infer + merge (pure)
│   └── file_utils.py          # .txt + real-MIME validation, in-memory read
├── download_model.py          # PRE-BUILD: snapshot_download -> ./model_cache/<id>/
├── tests/
│   ├── __init__.py
│   ├── test_file_utils.py     # accepts text/plain, rejects renamed binaries
│   └── test_ner_engine.py     # chunk offsets + cross-chunk merge (stub tokenizer)
├── .github/workflows/ci.yml   # checkout -> py3.11 -> flake8 + black --check -> pytest
├── Dockerfile                 # COPY model_cache/, offline env, healthcheck
├── docker-compose.yml         # MODEL_PATH env = the dynamic swap point
├── requirements.txt           # app deps (pinned)
├── requirements-dev.txt       # flake8, black, pytest (no torch -> fast CI)
├── .gitignore                 # model_cache/ and .claude/ are ignored
├── .dockerignore
├── pyproject.toml             # black + flake8 config
└── README.md                  # full pre-build -> build -> run flow
```

## 5. Component specifications

### 5.1 `src/config.py`
- `AppConfig` dataclass built from environment variables with sane defaults.
- Fields: `model_path` (`MODEL_PATH`), `max_file_mb` (`MAX_FILE_MB`, default 10),
  `chunk_overlap` (`CHUNK_OVERLAP`, default 32 tokens), `aggregation_strategy`
  (default `simple`).
- Single source of truth for configuration; imported by `app.py` and the engine.

### 5.2 `src/ner_engine.py` (pure, no Streamlit)
- `load_model(model_path) -> NerEngine`: loads tokenizer + model with
  `local_files_only=True`; builds the HF pipeline; exposes `id2label`, model
  `max_length`, and `device`.
- `Entity` dataclass: `class_name`, `text`, `score`, `start`, `end`.
- `chunk_text(text, tokenizer, max_len, overlap) -> list[Chunk]`: tokenizer-aware
  windowing with overlap; each chunk records its character offset in the full
  document so spans can be remapped.
- `extract_entities(text, progress_cb=None) -> list[Entity]`: runs the pipeline per
  chunk, remaps offsets to the full document, merges/dedupes entities that span
  chunk boundaries, calls `progress_cb(done, total)` for the UI.
- Class names come solely from the model config; no hardcoded label set.

### 5.3 `src/file_utils.py`
- `read_upload(uploaded_file, max_bytes) -> bytes`: reads the full upload into
  memory; raises on oversize.
- `validate_text_file(filename, data) -> None | raises ValidationError`: checks
  `.txt` extension **and** real MIME via `python-magic` (libmagic) is in the
  text/plain family, **and** that the bytes decode as UTF-8. Rejects a binary file
  renamed to `.txt`.
- `ValidationError` carries a user-friendly message for the UI.

### 5.4 `app.py` (Streamlit, thin)
- Loads `AppConfig`; loads the engine once via `@st.cache_resource`.
- Sidebar: model path, number of entity classes, device, max token length, max file
  size — so the active model is always visible.
- Main: single-file uploader (`type=["txt"]`). On upload → validate → `st.toast`.
- Processing: `st.spinner` for model load, `st.progress` advancing per chunk.
- Results: the required **Class Name | Identified Entity** table, a per-class count
  summary (columns/metrics), a highlighted-text expander, and a CSV
  `st.download_button`.

### 5.5 `download_model.py` (online, pre-build only)
- CLI: `python download_model.py [--model <hf_id>] [--out ./model_cache]`.
- Default model id `CyberPeace-Institute/SecureBERT-NER` (overridable; also reads
  `MODEL_NAME`). This default is a convenience only and is never assumed by app code.
- Uses `huggingface_hub.snapshot_download` to fetch all files into
  `model_cache/<sanitized_id>/` (slashes → `__`), so multiple models can coexist.
- Verifies the snapshot (config.json present and contains `id2label`) and prints the
  exact `MODEL_PATH` to use.

## 6. Configuration / environment variables

| Variable | Stage | Meaning | Default |
|---|---|---|---|
| `MODEL_NAME` | pre-build (online) | HF repo id to download | `CyberPeace-Institute/SecureBERT-NER` |
| `MODEL_PATH` | runtime (offline) | local cached model dir to load | `/app/model_cache/CyberPeace-Institute__SecureBERT-NER` |
| `MAX_FILE_MB` | runtime | upload size guard | `10` |
| `CHUNK_OVERLAP` | runtime | token overlap between chunks | `32` |
| `TRANSFORMERS_OFFLINE`, `HF_HUB_OFFLINE` | runtime | force offline | `1` |

`MODEL_NAME` = what to download; `MODEL_PATH` = where to load from. The split keeps
the online and offline stages cleanly separated.

## 7. Data flow

1. User uploads a single `.txt` file.
2. `file_utils.read_upload` loads the full bytes into memory (size-guarded).
3. `file_utils.validate_text_file` checks extension + real MIME + UTF-8 decode.
4. Text → `ner_engine.extract_entities`.
5. Tokenizer-aware chunking (≤ model max length, with overlap); `st.progress`
   advances per chunk.
6. Pipeline inference per chunk → offsets remapped to the full document → entities
   merged/deduped across seams.
7. Entities → DataFrame (Class Name, Identified Entity) → table + highlighted text +
   CSV download.

## 8. Error handling

- Invalid type / MIME / non-UTF-8 → `st.error` + toast, stop.
- Empty file → friendly message.
- Oversize file → graceful warning naming `MAX_FILE_MB`.
- Missing model cache → clear error pointing to `download_model.py`.
- Inference error → caught and surfaced.
- Offline is enforced (`local_files_only=True`, `*_OFFLINE=1`) so no accidental
  network calls occur at runtime.

## 9. Offline Docker strategy

- Base `python:3.11-slim`; install `libmagic1` for `python-magic`.
- `pip install` from `requirements.txt` (CPU torch).
- `COPY model_cache/` then `COPY` source.
- `ENV MODEL_PATH=... TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1`.
- `EXPOSE 8501`, `HEALTHCHECK` on Streamlit, `CMD streamlit run app.py`.
- No network access required at build (after model is present) or at runtime.
- `docker-compose.yml` mounts `./model_cache` read-only and sets `MODEL_PATH`, so a
  model swap is an env-var change, not a rebuild.

## 10. CI & testing strategy

- CI installs only `requirements-dev.txt` (flake8, black, pytest, python-magic,
  pandas) plus `libmagic1` — **no torch** — so CI is fast.
- `tests/test_file_utils.py`: accepts `text/plain`, rejects a PNG renamed `.txt`,
  rejects oversize.
- `tests/test_ner_engine.py`: chunk-offset correctness and cross-chunk entity merge,
  driven by a lightweight **stub tokenizer/pipeline** so no model download is needed.
- Workflow: checkout → setup Python 3.11 → install libmagic1 + dev deps →
  `flake8` → `black --check` → `pytest`.

## 11. Local development setup (macOS, Apple Silicon / M4)

- `brew install libmagic` (required so `python-magic` does not crash on macOS).
- `python3 -m venv .venv && source .venv/bin/activate`
- `pip install -r requirements.txt`
- `python download_model.py` (online, one time) → prints the `MODEL_PATH`.
- `MODEL_PATH=./model_cache/<id> streamlit run app.py`

## 12. Atomic commit plan

Branch: `feat/threatextract-app` (the design doc itself lands on `main` first).

1. Scaffolding: `.gitignore`, `.dockerignore`, `pyproject.toml`, README skeleton.
2. `requirements.txt` + `download_model.py`.
3. `src/config.py` + `src/ner_engine.py`.
4. `src/file_utils.py`.
5. `app.py`.
6. `Dockerfile` + `docker-compose.yml`.
7. `tests/` + `requirements-dev.txt`.
8. `.github/workflows/ci.yml`.
9. README finalize → push → PR / merge to `main`.

## 13. Out of scope

- Fork 1 benchmarking (already complete).
- Training / fine-tuning models.
- Multi-file or batch processing (assignment: one file at a time).
- Authentication / multi-tenant concerns (on-prem, single-user tool).
```
