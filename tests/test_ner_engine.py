"""Tests for the pure NER helpers — no torch/transformers required.

These exercise the chunking, offset remapping, and entity-merge logic with a
lightweight stub tokenizer, which is what lets CI run them without the heavy
model stack.
"""

from src.ner_engine import Chunk, Entity, chunk_text, merge_entities, strip_bio


class _StubTokenizer:
    """Whitespace tokenizer exposing just what ``chunk_text`` needs.

    Each space-separated word becomes one token; ``offset_mapping`` records its
    (start, end) character span in the original text.
    """

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 2

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        offsets = []
        cursor = 0
        for word in text.split(" "):
            if word:
                start = text.index(word, cursor)
                offsets.append((start, start + len(word)))
                cursor = start + len(word)
        return {"offset_mapping": offsets}


def test_strip_bio_variants():
    assert strip_bio("B-APT") == "APT"
    assert strip_bio("I-MAL") == "MAL"
    assert strip_bio("O") == "O"
    assert strip_bio("APT") == "APT"
    assert strip_bio("") == ""


def test_chunk_text_short_stays_single_chunk():
    chunks = chunk_text("alpha beta gamma", _StubTokenizer(), max_length=100, overlap=2)
    assert chunks == [Chunk(text="alpha beta gamma", offset=0)]


def test_chunk_text_splits_with_correct_offsets_and_coverage():
    text = "aa bb cc dd ee ff"  # 6 tokens; window = 4 - 2 special = 2, overlap 1
    chunks = chunk_text(text, _StubTokenizer(), max_length=4, overlap=1)
    assert len(chunks) > 1
    # Every chunk's text is a real substring at its recorded offset.
    for ch in chunks:
        assert text[ch.offset : ch.offset + len(ch.text)] == ch.text
    # The chunks together cover the whole document.
    assert chunks[0].offset == 0
    assert chunks[-1].offset + len(chunks[-1].text) == len(text)


def test_merge_dedupes_identical_spans_keeping_best_score():
    text = "APT29 attacked"
    ents = [
        Entity("APT", "APT29", 0.90, 0, 5),
        Entity("APT", "APT29", 0.95, 0, 5),  # same span from an overlapping chunk
    ]
    merged = merge_entities(ents, text)
    assert len(merged) == 1
    assert merged[0].score == 0.95


def test_merge_stitches_adjacent_same_class():
    text = "CVE-2021-44228 exploited"
    ents = [
        Entity("VULID", "CVE", 0.9, 0, 3),
        Entity("VULID", "-", 0.9, 3, 4),
        Entity("VULID", "2021", 0.9, 4, 8),
        Entity("VULID", "-", 0.9, 8, 9),
        Entity("VULID", "44228", 0.9, 9, 14),
    ]
    merged = merge_entities(ents, text)
    assert len(merged) == 1
    assert merged[0].text == "CVE-2021-44228"
    assert (merged[0].start, merged[0].end) == (0, 14)


def test_merge_keeps_whitespace_separated_entities_distinct():
    text = "WellMess malware"  # space at index 8 is a boundary
    ents = [
        Entity("MAL", "WellMess", 0.9, 0, 8),
        Entity("MAL", "malware", 0.9, 9, 16),
    ]
    merged = merge_entities(ents, text)
    assert len(merged) == 2


def test_merge_does_not_join_different_classes():
    text = "APT29 WellMess"
    ents = [
        Entity("APT", "APT29", 0.9, 0, 5),
        Entity("MAL", "WellMess", 0.9, 6, 14),
    ]
    merged = merge_entities(ents, text)
    assert len(merged) == 2
