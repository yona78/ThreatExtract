# ThreatExtract

Model-agnostic **Named Entity Recognition (NER)** serving app for cyber-threat
intelligence. Upload a `.txt` report and get a clean table of the entities a
Hugging Face NER model finds in it — running **fully offline** inside Docker.

> Fork 2 of the MLE home assignment: the production-grade serving application.
> Fork 1 (benchmarking SecureBERT-NER vs CyNER on the DNRTI dataset) is separate.

## Why it's model-agnostic

The app serves **any** Hugging Face `token-classification` model. Entity classes
are read at runtime from the model's own `config.json` (`id2label`), and the model
is selected purely via the `MODEL_PATH` environment variable — so you can swap
models without touching a line of code.

## Architecture at a glance

- **Pre-build (online):** `download_model.py` fetches model weights into
  `./model_cache/`.
- **Build:** the `Dockerfile` only `COPY`s `model_cache/` in — it never downloads.
- **Runtime (offline):** Streamlit UI loads the model from the local cache.

## Quickstart

> Full setup, Docker, and usage instructions are documented as the components land
> (see `docs/superpowers/specs/2026-06-21-threatextract-app-design.md` for the
> design). This README is finalized in the last step of the build.

## Fork 1 Research Reproduction

Fork 1 compares frozen SecureBERT-NER and CyNER on DNRTI using the local offline
cache. Keep research-only dependencies out of the Docker runtime; `make
reproduce` installs the pinned research dependencies into `.venv` once, then all
model inference reads local DNRTI/model-cache files:

```bash
make reproduce
```

Useful overrides:

```bash
make reproduce DNRTI_DIR=data/dnrti CACHE_DIR=model_cache/fork1 DEVICE=mps
make e11
```

`DEVICE` defaults to `cpu` for portable reproduction; pass `DEVICE=mps` to
rerun the accuracy sweeps on Apple MPS. The operational target always measures
both CPU and MPS when MPS is visible.

The final paper-style report is `reports/fork1/benchmark_summary.md`; the master
table is `reports/fork1/master_table.jsonl`.

## License

For assignment / evaluation use.
