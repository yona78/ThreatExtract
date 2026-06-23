# ThreatExtract

Model-agnostic **Named Entity Recognition (NER)** for cyber-threat intelligence.
Upload a `.txt` report and get a color-coded breakdown of the entities a Hugging
Face NER model finds in it — APT groups, malware, indicators (IPs, domains,
hashes), CVEs, and more — running **fully offline** inside Docker.

The served model is **SecureBERT-NER** (`CyberPeace-Institute/SecureBERT-NER`),
but the app is model-agnostic: entity classes are read at runtime from the
model's own `config.json`, and the model is chosen purely via the `MODEL_PATH`
environment variable — no code changes to swap it.

![offline](https://img.shields.io/badge/offline-on--prem-success)
![python](https://img.shields.io/badge/python-3.11-blue)
[![container](https://img.shields.io/badge/ghcr.io-threatextract-2496ED?logo=docker)](https://github.com/yona78/ThreatExtract/pkgs/container/threatextract)

> Fork 2 of the MLE home assignment — the serving app. Fork 1 (benchmarking
> SecureBERT-NER vs CyNER on the DNRTI dataset) lives alongside it in this repo.

---

## Quickstart — run the prebuilt image

The image is published to this repo's **GitHub Container Registry**, with the
SecureBERT-NER model already baked in. One command pulls and runs it — Docker
fetches it once, then it serves fully offline:

```bash
docker run --rm -p 8501:8501 ghcr.io/yona78/threatextract:latest
```

Then open **http://localhost:8501**. `Ctrl-C` to stop.

> **Auth error on pull?** The package is private — either make it public
> (repo → **Packages** → *threatextract* → visibility → Public) or run
> `docker login ghcr.io -u <user>` with a token first.

## Using the app

A sample report ships in both the repo and the image:
[`examples/sample_report.txt`](examples/sample_report.txt).

1. The **sidebar** lists the loaded model and every entity class it can detect —
   the UI adapts to whatever model is loaded (nothing is hardcoded).
2. **Upload** a `.txt` report — drag-and-drop or *Browse files*. One file at a time.
3. The file is validated (extension **+** real MIME type via `libmagic` **+** UTF-8),
   a progress bar runs, then the results render.
4. **Results table** — exactly two columns, **Class Name** and **Identified
   Entity**. The sample yields ~20 entities, e.g.:

   | Class Name | Identified Entity |
   |---|---|
   | APT | APT29 |
   | APT | Cozy Bear |
   | MAL | WellMess |
   | IDTY | government agencies |
   | IP | 203.0.113.42 |
   | VULID | CVE-2021-44228 |
   | URL | http://malicious.example.net/payload.exe |
   | MD5 | 5d41402abc4b2a76b9719d911017c592 |
   | SECTEAM | CERT-EU |
   | TOOL | Nmap |

5. **Download the CSV** (button under the table) and expand **Highlighted
   document** to see every entity color-coded inline.
6. **Guardrail:** rename a binary (say an image) to `something.txt` and upload it —
   the app rejects it, because validation inspects the real content, not the name.

Prefer the terminal? Confirm inference headlessly (container named `threatextract`):

```bash
docker exec -i threatextract python - <<'PY'
import os, sys
sys.path.insert(0, "/app")
from src.ner_engine import NerEngine
engine = NerEngine(os.environ["MODEL_PATH"])
entities = engine.extract_entities(open("/app/examples/sample_report.txt").read())
print(f"{len(entities)} entities")
for ent in entities:
    print(f"  {ent.class_name:8} {ent.text}")
PY
```

## Build it yourself (Docker)

To rebuild from source — e.g. to bake in a different model — download the model
once (online), then build the offline image:

```bash
python -m venv .venv && source .venv/bin/activate
pip install huggingface_hub                 # or: pip install -r requirements.txt
python download_model.py                    # default: CyberPeace-Institute/SecureBERT-NER

docker compose up --build                   # builds the offline image, serves on :8501
```

The model is fetched **before** the build and baked into the image, so the
container never touches the network. Prove it by running with networking off:

```bash
docker run -d --name te --network none threatextract:latest
docker exec te curl -fsS http://localhost:8501/_stcore/health   # -> ok
docker rm -f te
```

(`docker run --rm -p 8501:8501 threatextract:latest` runs the local image without
Compose.)

## Run without Docker

```bash
brew install libmagic                        # macOS (Debian/Ubuntu: apt-get install -y libmagic1)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python download_model.py
MODEL_PATH=./model_cache/CyberPeace-Institute__SecureBERT-NER streamlit run app.py
```

## Swap the model (no code changes)

`download_model.py` accepts any Hugging Face `token-classification` repo id:

```bash
python download_model.py --model AI4Sec/cyner-xlm-roberta-large
# then point MODEL_PATH at the printed directory (docker-compose.yml or the shell)
```

The sidebar always reflects the loaded model and its classes.

## Configuration

| Variable | Stage | Meaning | Default |
|---|---|---|---|
| `MODEL_NAME` | pre-build | HF repo id to download | `CyberPeace-Institute/SecureBERT-NER` |
| `MODEL_PATH` | runtime | local model dir to load | `./model_cache/CyberPeace-Institute__SecureBERT-NER` |
| `MAX_FILE_MB` | runtime | upload size guard | `10` |
| `CHUNK_OVERLAP` | runtime | token overlap between chunks | `32` |
| `AGGREGATION_STRATEGY` | runtime | HF pipeline aggregation | `simple` |
| `TRANSFORMERS_OFFLINE`, `HF_HUB_OFFLINE` | runtime | force offline | `1` (in Docker) |

## How it works

```
download_model.py        Dockerfile                 app.py (Streamlit)
─────────────────        ──────────                 ──────────────────
HF Hub ─(online,   ─►  bake the chosen model  ─►  load MODEL_PATH offline
 once)─ model_cache/    dir into the image          (local_files_only=True)

upload .txt ─► validate ─► chunk (token-aligned, overlapping) ─► infer ─► merge
            ─► results table + CSV + highlighted document
```

- **Offline by construction.** The model is downloaded before the build and
  `COPY`-ed in; at runtime `local_files_only=True` plus `TRANSFORMERS_OFFLINE=1` /
  `HF_HUB_OFFLINE=1` guarantee no network call can occur.
- **Long documents** are split into overlapping, token-aligned chunks so nothing
  is lost to the model's token limit; entity offsets are remapped to the full document.
- **Clean entities.** Piecewise output (`CVE` `-` `2021` …) is stitched into whole
  entities (`CVE-2021-44228`), and surrounding whitespace is trimmed from each span.

## Development

```bash
pip install -r requirements.txt -r requirements-dev.txt
ruff check app.py src tests download_model.py
black --check app.py src tests download_model.py
pytest                                       # enforces a ≥85% coverage gate on src/
```

Unit tests mock the `transformers` stack, so they run in under a second with no
torch or model needed. CI (`.github/workflows/ci.yml`) runs ruff + black + pytest
on every push to `main` and every pull request.

## Project structure

```
app.py              Streamlit UI (thin; delegates to src/)
src/ner_engine.py   model-agnostic load + chunk + infer + merge
src/file_utils.py   .txt + real-MIME + UTF-8 validation
src/config.py       env-driven settings
download_model.py   pre-build model fetch into ./model_cache/
Dockerfile          offline image (bakes only the chosen model)
docker-compose.yml  run config + the model-swap point
examples/           sample threat report
tests/              unit tests (mocked; no torch/model required)
```

## Notes

- **One file at a time**, per the assignment.
- **Image ~2.7 GB.** Only the chosen model's directory is baked in;
  `download_model.py` skips the redundant `pytorch_model.bin` (when `safetensors`
  is present) and training-checkpoint artifacts (optimizer / scheduler / RNG /
  trainer state) — trimming SecureBERT-NER's payload from ~2 GB to ~480 MB. The
  rest is the unavoidable CPU `torch` + `transformers` + Streamlit runtime.

## Fork 1 — benchmarking

Fork 1 compares SecureBERT-NER vs CyNER on the DNRTI dataset, fully offline. See
`make reproduce` and the reports under `reports/fork1/`.

## License

For assignment / evaluation use.
