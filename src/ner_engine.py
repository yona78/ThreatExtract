"""Model-agnostic Named Entity Recognition engine.

Loads any Hugging Face ``token-classification`` model from a local directory
(offline), splits long documents into token-aligned overlapping chunks so
nothing is lost to truncation, runs inference, and returns entities whose
character offsets point back into the original document. Entity classes are read
from the model's own config — none are hardcoded.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# NOTE: ``transformers`` is imported lazily inside ``NerEngine.__init__`` so the
# pure helpers (chunk_text, merge_entities, strip_bio) and the dataclasses can be
# imported and unit-tested without the heavy torch/transformers stack — which is
# what keeps CI fast (it installs neither).

_BIO_PREFIXES = ("B-", "I-", "L-", "U-", "E-", "S-")
_DEFAULT_MAX_LENGTH = 512


@dataclass(frozen=True)
class Entity:
    """A single detected entity, with offsets into the original document."""

    class_name: str
    text: str
    score: float
    start: int
    end: int


@dataclass(frozen=True)
class Chunk:
    """A slice of the document plus its character offset within the original."""

    text: str
    offset: int


def strip_bio(label: str) -> str:
    """Strip a BIO/BILOU prefix (e.g. ``B-APT`` -> ``APT``)."""
    if not label:
        return label
    for prefix in _BIO_PREFIXES:
        if label.startswith(prefix):
            return label[len(prefix) :]
    return label


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Shrink ``[start, end)`` inward past any leading/trailing whitespace."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def chunk_text(text: str, tokenizer, max_length: int, overlap: int) -> list[Chunk]:
    """Split ``text`` into token-aligned, overlapping chunks that fit the model.

    Chunks are cut on token boundaries (via the fast tokenizer's offset mapping),
    so each chunk's ``text`` is a clean substring of the original and ``offset``
    is its starting character index — which lets us remap entity spans back to
    the full document.
    """
    special = tokenizer.num_special_tokens_to_add(pair=False)
    window = max(1, max_length - special)
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]
    n = len(offsets)
    if n == 0:
        return []
    if n <= window:
        return [Chunk(text=text, offset=0)]

    step = max(1, window - overlap)
    chunks: list[Chunk] = []
    start_tok = 0
    while start_tok < n:
        end_tok = min(start_tok + window, n)
        char_start = offsets[start_tok][0]
        char_end = offsets[end_tok - 1][1]
        chunks.append(Chunk(text=text[char_start:char_end], offset=char_start))
        if end_tok >= n:
            break
        start_tok += step
    return chunks


def merge_entities(entities: list[Entity], full_text: str) -> list[Entity]:
    """De-duplicate and stitch entities into clean spans.

    1. Drop identical spans (seen in overlapping chunks), keeping the best score.
    2. Merge directly adjacent or overlapping spans of the *same* class into one,
       so subword/punctuation runs the model tags piecewise (``CVE`` ``-``
       ``2021`` ... or ``evil`` ``-`` ``c2`` ``.`` ``com``) become a single
       entity. A whitespace gap is preserved as a boundary, so genuinely distinct
       entities are never run together.
    """
    if not entities:
        return []
    best: dict = {}
    for ent in entities:
        key = (ent.start, ent.end, ent.class_name)
        if key not in best or ent.score > best[key].score:
            best[key] = ent
    ordered = sorted(best.values(), key=lambda e: (e.start, e.end))

    merged: list[Entity] = []
    for ent in ordered:
        prev = merged[-1] if merged else None
        if prev and ent.class_name == prev.class_name and ent.start <= prev.end:
            new_end = max(prev.end, ent.end)
            merged[-1] = Entity(
                class_name=prev.class_name,
                text=full_text[prev.start : new_end],
                score=min(prev.score, ent.score),
                start=prev.start,
                end=new_end,
            )
        else:
            merged.append(ent)
    return merged


class NerEngine:
    """Stateful wrapper around a Hugging Face token-classification pipeline."""

    def __init__(self, model_path, aggregation_strategy: str = "simple", chunk_overlap: int = 32):
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            pipeline,
        )

        self.model_path = str(model_path)
        self.aggregation_strategy = aggregation_strategy
        self.chunk_overlap = chunk_overlap
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_path, local_files_only=True
        )
        self.pipeline = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy=self.aggregation_strategy,
        )
        self.id2label = dict(self.model.config.id2label)
        self.max_length = self._resolve_max_length()

    def _resolve_max_length(self) -> int:
        # Some tokenizers report a sentinel like int(1e30); guard against it.
        candidates = [
            getattr(self.tokenizer, "model_max_length", None),
            getattr(self.model.config, "max_position_embeddings", None),
        ]
        valid = [c for c in candidates if isinstance(c, int) and 0 < c < 100_000]
        return min(valid) if valid else _DEFAULT_MAX_LENGTH

    @property
    def class_names(self) -> list[str]:
        """Sorted entity class names from the model config (BIO-stripped, no O)."""
        names = set()
        for label in self.id2label.values():
            name = strip_bio(str(label))
            if name and name != "O":
                names.add(name)
        return sorted(names)

    def extract_entities(
        self, text: str, progress_cb: Callable[[int, int], None] | None = None
    ) -> list[Entity]:
        """Return merged entities for ``text`` (empty list for blank input)."""
        if not text or not text.strip():
            return []
        chunks = chunk_text(text, self.tokenizer, self.max_length, self.chunk_overlap)
        total = len(chunks)
        entities: list[Entity] = []
        for index, chunk in enumerate(chunks):
            for raw in self.pipeline(chunk.text):
                entity = self._to_entity(raw, chunk.offset, text)
                if entity is not None:
                    entities.append(entity)
            if progress_cb is not None:
                progress_cb(index + 1, total)
        return merge_entities(entities, text)

    def _to_entity(self, raw: dict, offset: int, full_text: str) -> Entity | None:
        start = int(raw["start"]) + offset
        end = int(raw["end"]) + offset
        # Pipeline spans can include leading/trailing whitespace (e.g. a token
        # that opens a new line). Trim it so the entity text is clean and so two
        # spans that differ only by a trailing newline collapse to one in
        # merge_entities. An all-whitespace span is dropped.
        start, end = _trim_span(full_text, start, end)
        if start >= end:
            return None
        label = raw.get("entity_group") or raw.get("entity") or ""
        return Entity(
            class_name=strip_bio(str(label)),
            text=full_text[start:end],
            score=float(raw["score"]),
            start=start,
            end=end,
        )
