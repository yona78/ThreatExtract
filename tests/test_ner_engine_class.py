"""Tests for NerEngine class — mocked transformers, no model download required.

The heavy ``transformers`` stack is patched out via ``sys.modules`` replacement
so ``NerEngine`` can be constructed and exercised in CI without torch or any
model files.  The ``from transformers import ...`` statement inside
``NerEngine.__init__`` is lazy (runs per-call), so replacing ``sys.modules['transformers']``
before each call is the reliable way to intercept it.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from src.ner_engine import (
    _DEFAULT_MAX_LENGTH,
    NerEngine,
    chunk_text,
    merge_entities,
)

# ---------------------------------------------------------------------------
# Shared fake objects
# ---------------------------------------------------------------------------

# A minimal BIO label map for a model with two entity classes.
_FAKE_ID2LABEL = {
    0: "O",
    1: "B-APT",
    2: "I-APT",
    3: "B-MAL",
    4: "I-MAL",
}


def _make_fake_tokenizer(model_max_length=512):
    tok = MagicMock()
    tok.model_max_length = model_max_length
    tok.num_special_tokens_to_add.return_value = 2

    def _tokenize(text, add_special_tokens=False, return_offsets_mapping=False):
        offsets = []
        cursor = 0
        for word in text.split():
            start = text.index(word, cursor)
            offsets.append((start, start + len(word)))
            cursor = start + len(word)
        return {"offset_mapping": offsets}

    tok.side_effect = _tokenize
    tok.__call__ = _tokenize
    return tok


def _make_fake_model(id2label=None, max_position_embeddings=512):
    model = MagicMock()
    model.config.id2label = id2label if id2label is not None else _FAKE_ID2LABEL
    model.config.max_position_embeddings = max_position_embeddings
    return model


@contextmanager
def _fake_transformers(tokenizer, model, pipeline_output):
    """Context manager that replaces sys.modules['transformers'] with a fake.

    This is necessary because ``NerEngine.__init__`` uses::

        from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

    which is a lazy import that executes at call time. Replacing the module in
    ``sys.modules`` before the call ensures the local ``from ... import`` binds
    to our mocks.
    """
    real = sys.modules.get("transformers")
    fake = MagicMock()
    fake.AutoTokenizer.from_pretrained.return_value = tokenizer
    fake.AutoModelForTokenClassification.from_pretrained.return_value = model
    fake.pipeline.return_value = MagicMock(return_value=pipeline_output)
    sys.modules["transformers"] = fake
    try:
        yield fake
    finally:
        if real is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = real


def _build_engine(
    pipeline_output=None,
    tokenizer=None,
    model=None,
    model_path="/fake/model_path",
    aggregation_strategy="simple",
    chunk_overlap=8,
):
    """Construct a NerEngine with transformers completely mocked."""
    if tokenizer is None:
        tokenizer = _make_fake_tokenizer()
    if model is None:
        model = _make_fake_model()
    if pipeline_output is None:
        pipeline_output = []

    with _fake_transformers(tokenizer, model, pipeline_output):
        engine = NerEngine(
            model_path,
            aggregation_strategy=aggregation_strategy,
            chunk_overlap=chunk_overlap,
        )
    return engine


# ---------------------------------------------------------------------------
# Construction & property tests
# ---------------------------------------------------------------------------


def test_ner_engine_constructs_without_error():
    engine = _build_engine()
    assert engine.model_path == "/fake/model_path"
    assert engine.aggregation_strategy == "simple"
    assert engine.chunk_overlap == 8


def test_class_names_strips_bio_and_excludes_o():
    engine = _build_engine()
    # _FAKE_ID2LABEL has B-APT, I-APT, B-MAL, I-MAL, O → classes are APT, MAL
    assert engine.class_names == ["APT", "MAL"]


def test_class_names_sorted():
    engine = _build_engine()
    names = engine.class_names
    assert names == sorted(names)


def test_max_length_uses_tokenizer_model_max_length():
    tok = _make_fake_tokenizer(model_max_length=256)
    model = _make_fake_model(max_position_embeddings=512)
    engine = _build_engine(tokenizer=tok, model=model)
    assert engine.max_length == 256


def test_max_length_takes_minimum_of_valid_candidates():
    """max_length = min(tokenizer.model_max_length, model.max_position_embeddings)."""
    tok = _make_fake_tokenizer(model_max_length=1024)
    model = _make_fake_model(max_position_embeddings=256)
    engine = _build_engine(tokenizer=tok, model=model)
    assert engine.max_length == 256


def test_max_length_falls_back_to_default_when_sentinel():
    """When model_max_length is a huge sentinel value, use _DEFAULT_MAX_LENGTH."""
    tok = _make_fake_tokenizer(model_max_length=int(1e30))

    # Build a model without max_position_embeddings attribute
    model = MagicMock()
    model.config.id2label = _FAKE_ID2LABEL
    # Remove max_position_embeddings so getattr returns None
    del model.config.max_position_embeddings

    engine = _build_engine(tokenizer=tok, model=model)
    assert engine.max_length == _DEFAULT_MAX_LENGTH


def test_max_length_uses_model_config_when_tokenizer_has_sentinel():
    """Falls back to model.config.max_position_embeddings when tokenizer has huge value."""
    tok = _make_fake_tokenizer(model_max_length=int(1e30))
    model = _make_fake_model(max_position_embeddings=512)
    engine = _build_engine(tokenizer=tok, model=model)
    assert engine.max_length == 512


# ---------------------------------------------------------------------------
# extract_entities — blank / empty input
# ---------------------------------------------------------------------------


def test_extract_entities_empty_string_returns_empty():
    engine = _build_engine()
    assert engine.extract_entities("") == []


def test_extract_entities_whitespace_only_returns_empty():
    engine = _build_engine()
    assert engine.extract_entities("   \n\t  ") == []


# ---------------------------------------------------------------------------
# extract_entities — integration with canned pipeline output
# ---------------------------------------------------------------------------


def test_extract_entities_single_chunk_entity():
    """Pipeline returns one entity; extract_entities remaps offsets correctly."""
    raw = [{"start": 0, "end": 5, "entity_group": "B-APT", "score": 0.95}]
    engine = _build_engine(pipeline_output=raw)
    text = "APT29 is a threat actor"
    entities = engine.extract_entities(text)
    assert len(entities) == 1
    assert entities[0].class_name == "APT"
    assert entities[0].text == "APT29"
    assert entities[0].start == 0
    assert entities[0].end == 5
    assert entities[0].score == pytest.approx(0.95)


def test_extract_entities_uses_entity_fallback_key():
    """_to_entity falls back to 'entity' key when 'entity_group' is absent."""
    raw = [{"start": 0, "end": 3, "entity": "B-MAL", "score": 0.80}]
    engine = _build_engine(pipeline_output=raw)
    entities = engine.extract_entities("CVE exploit")
    assert entities[0].class_name == "MAL"


def test_extract_entities_empty_label_becomes_empty_string():
    """If neither entity_group nor entity is present, class_name is ''."""
    raw = [{"start": 0, "end": 3, "score": 0.5}]
    engine = _build_engine(pipeline_output=raw)
    entities = engine.extract_entities("foo bar")
    assert entities[0].class_name == ""


def test_extract_entities_progress_callback_called():
    """progress_cb is called once per chunk with (done, total)."""
    raw = [{"start": 0, "end": 3, "entity_group": "APT", "score": 0.9}]
    engine = _build_engine(pipeline_output=raw)
    calls = []
    engine.extract_entities("one two three", progress_cb=lambda d, t: calls.append((d, t)))
    # Single chunk → one call with (1, 1)
    assert calls == [(1, 1)]


def test_extract_entities_no_progress_callback():
    """When progress_cb is None, no error is raised."""
    raw = [{"start": 0, "end": 3, "entity_group": "APT", "score": 0.9}]
    engine = _build_engine(pipeline_output=raw)
    result = engine.extract_entities("one two three", progress_cb=None)
    assert isinstance(result, list)


def test_extract_entities_merges_duplicate_spans():
    """Duplicate entities from overlapping chunks are merged (best score kept)."""
    raw = [
        {"start": 0, "end": 5, "entity_group": "APT", "score": 0.80},
        {"start": 0, "end": 5, "entity_group": "APT", "score": 0.95},
    ]
    engine = _build_engine(pipeline_output=raw)
    entities = engine.extract_entities("APT29 detected")
    apt_entities = [e for e in entities if e.start == 0 and e.end == 5]
    assert len(apt_entities) == 1
    assert apt_entities[0].score == pytest.approx(0.95)


def test_extract_entities_returns_empty_when_pipeline_empty():
    """No entities from pipeline means empty result."""
    engine = _build_engine(pipeline_output=[])
    entities = engine.extract_entities("no entities here")
    assert entities == []


def test_extract_entities_trims_surrounding_whitespace():
    """Spans that include leading/trailing whitespace are trimmed to clean text."""
    raw = [{"start": 0, "end": 6, "entity_group": "MAL", "score": 0.9}]
    engine = _build_engine(pipeline_output=raw)
    # text[0:6] == "\nWell " → trimmed inward to "Well" at offsets 1..5.
    entities = engine.extract_entities("\nWell more")
    assert len(entities) == 1
    assert entities[0].text == "Well"
    assert entities[0].start == 1
    assert entities[0].end == 5


def test_extract_entities_drops_whitespace_only_span():
    """A span that is entirely whitespace is dropped, not emitted as empty."""
    raw = [{"start": 0, "end": 1, "entity_group": "MAL", "score": 0.9}]
    engine = _build_engine(pipeline_output=raw)
    # text[0:1] == "\n" → empty after trimming → dropped.
    assert engine.extract_entities("\nfoo") == []


# ---------------------------------------------------------------------------
# chunk_text edge cases not covered by existing tests
# ---------------------------------------------------------------------------


def test_chunk_text_empty_text_returns_empty():
    """When tokenizer returns no tokens, chunk_text returns []."""

    class _EmptyTokenizer:
        def num_special_tokens_to_add(self, pair=False):
            return 2

        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            return {"offset_mapping": []}

    result = chunk_text("", _EmptyTokenizer(), max_length=10, overlap=2)
    assert result == []


# ---------------------------------------------------------------------------
# merge_entities edge cases not covered by existing tests
# ---------------------------------------------------------------------------


def test_merge_entities_empty_list_returns_empty():
    assert merge_entities([], "any text") == []
