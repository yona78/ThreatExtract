# ThreatExtract

Model-agnostic **Named Entity Recognition (NER)** serving app for cyber-threat
intelligence. Upload a `.txt` report and get a clean, color-coded breakdown of
the entities a Hugging Face NER model finds in it — APT groups, malware,
indicators (IPs, domains, hashes), CVEs, and more — running **fully offline**
inside Docker.

> Fork 2 of the MLE home assignment: the production-grade serving application.
> Fork 1 (benchmarking SecureBERT-NER vs CyNER on the DNRTI dataset) lives
> alongside it in this repo but is independent.

![status](https://img.shields.io/badge/offline-on--prem-success)
![python](https://img.shields.io/badge/python-3.11-blue)

---

## Why it's model-agnostic

The app serves **any** Hugging Face `token-classification` model. Entity classes
are read at runtime from the model's own `config.json` (`id2label`) — nothing is
hardcoded — and the model is selected purely via the `MODEL_PATH` environment
variable. Swap SecureBERT-NER for CyNER (or anything else) by downloading it and
changing one env var; the Python code never changes.

## How it works

```
  download_model.py            Dockerfile                 app.py (Streamlit)
  ─────────────────            ──────────                 ─────────────────
  HF Hub ──(online,    ──►  COPY model_cache/   ──►   load MODEL_PATH offline
  pre-build)── ./model_cache/   into the image          (local_files_only)
                                                          │
   upload .txt ─► validate (ext + real MIME + UTF-8) ─► chunk ─► infer ─► merge
                                                          │
                              results table + CSV + highlighted document
```

- **Pre-build (online, once):** `download_model.py` snapshot-downloads a model
  into `./model_cache/<sanitized_id>/` and verifies it.
- **Build:** the `Dockerfile` only `COPY`s `model_cache/` in — it never reaches
  the network at build or run time.
- **Runtime (offline):** the Streamlit app loads the model from the local cache
  (`local_files_only=True`, plus `TRANSFORMERS_OFFLINE=1`).

---

## Quickstart (Docker — recommended)

Prerequisite: Docker / Docker Compose. You also need Python 3.11 once, to run the
pre-build download step.

```bash
# 1) Download the model into ./model_cache/ (online, one time)
python -m venv .venv && source .venv/bin/activate
pip install huggingface_hub            # (or: pip install -r requirements.txt)
python download_model.py               # default: CyberPeace-Institute/SecureBERT-NER

# 2) Build the offline image and run it
docker compose up --build

# 3) Open the app
open http://localhost:8501             # (Linux: xdg-open)
```

That's it — the container now runs entirely on-prem with no network access.

## Quickstart (local, no Docker)

On macOS (Apple Silicon / M-series) you need the `libmagic` system library so
`python-magic` can sniff file types:

```bash
brew install libmagic                  # macOS — REQUIRED, or python-magic crashes
# (Debian/Ubuntu: sudo apt-get install -y libmagic1)

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python download_model.py               # populates ./model_cache/

MODEL_PATH=./model_cache/CyberPeace-Institute__SecureBERT-NER \
  streamlit run app.py
```

---

## Swapping models (no code changes)

`download_model.py` takes any Hugging Face `token-classification` repo id:

```bash
python download_model.py --model AI4Sec/cyner-xlm-roberta-large
# or:  MODEL_NAME=<repo_id> python download_model.py
```

It prints the `MODEL_PATH` to use. Then either set it in `docker-compose.yml`
(the `MODEL_PATH` env) or pass it at runtime:

```bash
MODEL_PATH=./model_cache/AI4Sec__cyner-xlm-roberta-large streamlit run app.py
```

The sidebar always shows which model is loaded and the classes it can detect.

## Configuration

| Variable | Stage | Meaning | Default |
|---|---|---|---|
| `MODEL_NAME` | pre-build | HF repo id to download | `CyberPeace-Institute/SecureBERT-NER` |
| `MODEL_PATH` | runtime | local model dir to load | `./model_cache/CyberPeace-Institute__SecureBERT-NER` |
| `MAX_FILE_MB` | runtime | upload size guard | `10` |
| `CHUNK_OVERLAP` | runtime | token overlap between chunks | `32` |
| `AGGREGATION_STRATEGY` | runtime | HF pipeline aggregation | `simple` |
| `TRANSFORMERS_OFFLINE`, `HF_HUB_OFFLINE` | runtime | force offline | `1` (in Docker) |

## Features

- **Strict, content-based file validation.** Only `.txt` is accepted, verified by
  extension **and** real MIME type via `libmagic` (not just the name) **and** a
  UTF-8 decode — so a binary renamed to `.txt` is rejected.
- **Handles large files.** The upload is read fully into memory, then split into
  token-aligned overlapping chunks so nothing is lost to the model's token limit;
  entity character-offsets are remapped back to the whole document, and a live
  `st.progress` bar reports chunk-by-chunk progress.
- **Clean entities.** Piecewise model output (`CVE` `-` `2021` …) is stitched into
  whole entities (`CVE-2021-44228`) while whitespace boundaries are preserved.
- **Premium UI.** A dark "threat console" theme, per-class color coding derived
  from the class name (works for any model), a results table (**Class Name** /
  **Identified Entity**), per-class summary chips, latency, **CSV download**, and a
  highlighted source document with inline entity chips.

---

## Development

```bash
pip install -r requirements.txt -r requirements-dev.txt
# lint + format + tests (exactly what CI runs)
ruff check app.py src tests download_model.py
black --check app.py src tests download_model.py
pytest          # enforces ≥85% coverage on src/ automatically
```

The unit tests cover `src/config.py`, `src/file_utils.py`, and `src/ner_engine.py`
(mocking the `transformers` stack so no model or torch is required). They run in
well under a second with **no** torch/transformers installed.

`pytest-cov` enforces a **≥85% coverage gate on `src/`** (excluding `src/ui.py`
and `src/__init__.py`, which are thin Streamlit rendering wrappers). The gate is
configured in `pyproject.toml` under `[tool.pytest.ini_options]`.

## Continuous integration

`.github/workflows/ci.yml` runs on every push to `main` and every pull request:
checkout → Python 3.11 → install `libmagic1` + `requirements-dev.txt` → `ruff`
→ `black --check` → `pytest` (with the ≥85% coverage gate).

## Project structure

```
app.py                  Streamlit UI (thin; delegates to src/)
src/config.py           env-driven settings
src/ner_engine.py       model-agnostic load + chunk + infer + merge
src/file_utils.py       .txt + real-MIME + UTF-8 validation
download_model.py       pre-build model fetch into ./model_cache/
Dockerfile              offline image (COPYs model_cache/)
docker-compose.yml      MODEL_PATH = the dynamic model-swap point
assets/styles.css       "threat console" theme
tests/                  unit tests (no model needed)
.github/workflows/ci.yml  lint + tests
docs/superpowers/specs/   design spec
```

## Notes & considerations

- **Offline guarantee.** Models enter the image only via `COPY model_cache/`. At
  runtime the app loads with `local_files_only=True` and the container sets
  `TRANSFORMERS_OFFLINE=1` / `HF_HUB_OFFLINE=1`, so no network call can occur.
- **One file at a time**, per the assignment.
- **Image size.** `download_model.py` skips the redundant `pytorch_model.bin`
  when a repo also ships `safetensors`, keeping the cache lean; otherwise keep
  only the model(s) you intend to serve in `model_cache/` before building.

## License

For assignment / evaluation use.
