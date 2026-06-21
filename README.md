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

## License

For assignment / evaluation use.
