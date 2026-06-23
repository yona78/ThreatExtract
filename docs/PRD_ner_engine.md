# PRD — NER Engine

**Parent project:** ThreatExtract (see `docs/PRD.md`)
**Version:** 1.00
**Status:** Approved
**Owner:** yona78 (ML Engineer)
**Last updated:** 2026-06-23

---

## 1. Purpose and role in the system

The NER engine is the core inference component. It takes the validated text of an
uploaded report (from the file-validation layer) and returns the named entities
found in it, as spans with character offsets into the original document. The
Streamlit UI is its only consumer; the UI renders the engine's output as a table,
a highlighted document, and a CSV.

Boundary:
- **Consumes:** a decoded UTF-8 string (the full report) plus runtime settings
  (`MODEL_PATH`, aggregation strategy, chunk overlap).
- **Produces:** an ordered list of `Entity(class_name, text, score, start, end)`.
- **Does not:** validate or read files (the file layer does that); render anything
  (the UI does that); choose or evaluate the model (Fork 1 did that).

The engine is **model-agnostic**: it loads whatever Hugging Face
`token-classification` model `MODEL_PATH` points at and derives the entity classes
from that model's own configuration. No class names are hardcoded.

## 2. Theoretical background

The engine wraps a transformer-based token-classification model. Such a model
assigns each input token a label from a fixed set, typically in the **BIO**
scheme: `B-<CLASS>` (beginning of an entity), `I-<CLASS>` (inside/continuation),
and `O` (outside any entity). The default model, `CyberPeace-Institute/
SecureBERT-NER`, emits the DNRTI label set (APT, SECTEAM, MAL, IP, URL, SHA2,
TIME, LOC, …) in BIO form.

Two practical complications drive the design:

1. **Subword tokenization.** Tokenizers split words into subword pieces (`APT29`
   → `APT`, `##29`), so raw model output is per-subword. Hugging Face's
   `token-classification` pipeline with an `aggregation_strategy` regroups
   subwords into word-level spans using the BIO tags.
2. **The model's token limit.** Transformer encoders accept a bounded sequence
   (512 tokens for BERT-family models). Longer documents must be windowed, or
   entities past the limit are silently lost. The engine slices the document on
   **token boundaries** using the fast tokenizer's offset mapping, with overlap so
   an entity near a window edge is wholly contained in at least one window, then
   remaps each entity's offsets back to the full document and de-duplicates spans
   seen in overlapping windows.

A residual issue is that some models tag a contiguous run with repeated `B-`
tags (e.g. `CVE` `-` `2021` `-` `44228` each `B-VULID`), which no aggregation
strategy stitches because each `B-` opens a new entity. The engine therefore adds
a post-aggregation pass that merges **directly adjacent** (zero-gap) spans of the
**same class**, preserving whitespace gaps as boundaries so genuinely distinct
entities are never run together.

## 3. Inputs and outputs

### 3.1 Input contract

| Field | Type | Range / format | Required | Description |
|-------|------|---------------|----------|-------------|
| `text` | str | UTF-8, may exceed the model token limit | Yes | The full report text |
| `model_path` | str/path | Local directory of a HF token-classification model | Yes | Model to load (offline) |
| `aggregation_strategy` | str | one of `simple`/`first`/`average`/`max` | No (default `simple`) | HF pipeline aggregation |
| `chunk_overlap` | int | ≥ 0 tokens | No (default 32) | Overlap between windows |

Validation: blank or whitespace-only `text` returns an empty list (not an error).
A `model_path` that cannot be loaded raises at construction and is surfaced by the
UI as guidance to run the pre-build download.

### 3.2 Output contract

A list of `Entity`, ordered by position, each with:

| Field | Type | Description |
|-------|------|-------------|
| `class_name` | str | Entity class, BIO prefix stripped (e.g. `APT`) |
| `text` | str | The exact substring `full_text[start:end]` |
| `score` | float | Confidence in [0, 1] (min across merged spans) |
| `start`, `end` | int | Character offsets into the original document |

Error shape: construction failures raise an exception with a human-readable
message; per-request extraction does not partially fail — it returns the entities
found or an empty list.

### 3.3 Side effects
- Reads model files from `MODEL_PATH` at construction (no writes).
- No network calls (`local_files_only=True`; the container also sets
  `TRANSFORMERS_OFFLINE=1`/`HF_HUB_OFFLINE=1`).
- Emits no logs of its own beyond the underlying library's; invokes the optional
  `progress_cb(done, total)` callback once per processed chunk.

## 4. Functional requirements

- **FR-engine-1:** Given any HF token-classification model directory, the engine
  loads it offline and exposes its entity classes from `id2label`.
- **FR-engine-2:** Given text within the token limit, the engine returns one entity
  per detected span with correct class and offsets.
- **FR-engine-3:** Given text exceeding the token limit, the engine windows it with
  overlap, remaps offsets to the full document, and returns no fewer entities than
  a per-window view would (no boundary loss).
- **FR-engine-4:** The engine merges directly adjacent same-class spans into one
  entity and de-duplicates identical spans from overlapping windows.
- **FR-engine-5:** Given blank/whitespace-only text, the engine returns `[]`.
- **FR-engine-6:** For every returned entity, `full_text[start:end] == text`.

## 5. Non-functional requirements

### 5.1 Performance
- **Latency:** < 5 s for a ≤ 2,000-char report on CPU; scales roughly linearly in
  the number of 512-token windows.
- **Resource usage:** CPU-only; memory bounded by the model (BERT-base class,
  ~0.5 GB resident) plus the in-memory document.
- **Cold start:** one model load per process; the UI caches the engine
  (`st.cache_resource`) so subsequent requests skip loading.

### 5.2 Quality / accuracy
- Entity quality is a property of the **chosen** model (selected in Fork 1); the
  engine does not alter the model's predictions beyond regrouping and stitching.
- The engine must not introduce span errors: the offset-equality invariant
  (FR-engine-6) is asserted in tests.
- The aggregation strategy is chosen to maximize clean, non-truncated spans (see
  §7.1) rather than a learned metric.

### 5.3 Reliability
- **Failure modes:** unloadable model (construction error); malformed pipeline
  output (guarded by `.get(...)` access).
- **Degradation:** none required — single-process, single-request.

### 5.4 Observability
- Exposes `class_names`, `max_length`, and `aggregation_strategy` for the UI to
  display; invokes `progress_cb` per chunk for the progress bar.

## 6. Constraints

### 6.1 Technical constraints
- CPU-only; no GPU.
- Requires a **fast** tokenizer (offset mapping) — true for BERT- and
  XLM-RoBERTa-family models, including both reference models.
- Pure module: no Streamlit/UI imports, so it is unit-testable in isolation; the
  `transformers` import is lazy so the pure helpers import without torch.

### 6.2 Compliance / policy constraints
- Runs entirely on-prem; report text never leaves the process.
- Model licensing is the operator's responsibility when swapping models.

## 7. Alternatives considered

### 7.1 Pipeline aggregation strategy

Measured on a representative sentence containing `APT29`, `Cozy Bear`, `WellMess`,
`evil-c2.example.com`, and `CVE-2021-44228` against the default model:

| Option | Result | Verdict |
|--------|--------|---------|
| `simple` | Clean spans: `APT29`, `Cozy Bear`, `evil-c2.example.com`, `CVE-2021-44228`; no lost entities | **Chosen** |
| `first` / `average` | Included trailing punctuation (`Cozy Bear,`, `evil-c2.example.com.`) | Rejected — dirty spans |
| `max` | Dropped entities (`Cozy` without `Bear`; domain missing entirely) | Rejected — lower recall |

**Rationale:** with the adjacent-merge pass (§7.3) doing the stitching, `simple`
yields the cleanest boundaries and best recall, and avoids the "tokenizer does not
support real words" fallback the word-level strategies hit on this model. It is
the default but remains overridable via `AGGREGATION_STRATEGY`.

### 7.2 Serving approach: plain pipeline vs. CyNER toolkit

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| Plain HF `token-classification` pipeline | Works for any token-classification model; minimal deps | Needs post-processing for piecewise tags | **Chosen** |
| CyNER toolkit (spaCy + Flair + regex wrapper) | Richer heuristics for one model | Heavy deps; couples the app to one model; breaks model-agnosticism | Rejected |

**Rationale:** the project's primary requirement is model-agnosticism (§G1 of the
main PRD). The CyNER Hugging Face model is itself a standard token-classification
model the generic path already serves, so the toolkit adds dependencies and
coupling for no benefit.

### 7.3 Entity stitching: post-merge vs. aggregation alone

**Options:** (a) rely on the pipeline's BIO aggregation only; (b) add a
zero-gap, same-class merge pass. Aggregation alone leaves runs like
`CVE`/`-`/`2021` fragmented because the model emits repeated `B-VULID`. The merge
pass (b) was **chosen**: it stitches such runs into `CVE-2021-44228` while
preserving whitespace boundaries so `WellMess` and the following word `malware`
remain distinct.

### 7.4 Weight format on download
`download_model.py` prefers `safetensors` and skips the duplicate
`pytorch_model.bin` when both are present, halving the cache/image, while still
downloading `.bin` for models that ship only that format.

## 8. Success criteria and test scenarios

### 8.1 Success criteria
- All FRs in §4 have passing tests; coverage of the engine module is ≥ 85% (it is
  100%, via mocked-transformers tests plus pure-helper tests).
- The offset-equality invariant holds for every entity in every test.

### 8.2 Test scenarios

**Happy path:**
- TS-1: A threat sentence → `APT29` (APT), `WellMess` (MAL), `CVE-2021-44228`
  (VULID) extracted, every span satisfying `full_text[start:end] == text`.
- TS-2: A class label map with `B-/I-` pairs → `class_names` returns the stripped,
  sorted classes excluding `O`.

**Edge cases:**
- TS-10: Blank / whitespace-only input → `[]`.
- TS-11: Input longer than one window → chunked, offsets remapped, coverage of the
  whole document, no duplicate spans.
- TS-12: Adjacent same-class run (`CVE` `-` `2021` `-` `44228`) → single
  `CVE-2021-44228` entity; whitespace-separated same-class spans stay distinct.
- TS-13: Tokenizer reporting a sentinel `model_max_length` (e.g. `1e30`) → falls
  back to the default window length.

**Failure scenarios:**
- TS-20: `MODEL_PATH` missing or unreadable → construction raises; the UI surfaces
  guidance to run `download_model.py`.

**Load scenarios:**
- Out of scope — single-user, one report at a time (see main PRD §3.2).

## 9. Open questions
- **Q1:** Should sharded multi-file safetensors models be range-validated before
  load? — Owner: operator. Target: post-submission.

## 10. References
- Devlin et al., *BERT* (2018) — the encoder family underpinning the models.
- Hugging Face `TokenClassificationPipeline` documentation — aggregation strategies.
- DNRTI dataset — the label set the default model emits.
- `docs/PRD.md` — parent project PRD (links here from its appendix).
