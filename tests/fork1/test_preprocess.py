from fork1.data import Sample
from fork1.preprocess import (
    defang,
    detok_punct_aware,
    detok_single_space,
    iter_contexts,
    normalize_text,
    refang,
)


def _sample(sample_id: str, tokens: tuple[str, ...]) -> Sample:
    return Sample(
        sample_id=sample_id,
        split="test",
        index=int(sample_id.rsplit("-", 1)[-1]),
        text=" ".join(tokens),
        tokens=tokens,
        tags=tuple("O" for _ in tokens),
        gold_spans=[],
    )


def test_single_space_matches_legacy() -> None:
    text, offsets = detok_single_space(["APT", "hit", "."])

    assert text == "APT hit ."
    assert offsets[0] == (0, 3)


def test_punct_aware_no_space_before_period() -> None:
    text, offsets = detok_punct_aware(["APT", "hit", "."])

    assert text == "APT hit."
    assert text[offsets[2][0] : offsets[2][1]] == "."


def test_refang_restores_common_ioc_forms() -> None:
    assert refang("hxxp://1.1.1[.]1") == "http://1.1.1.1"


def test_normalize_nfkc_is_idempotent() -> None:
    normalized = normalize_text("ＣＶＥ-２０２１-４４２２８", "nfkc")

    assert normalize_text(normalized, "nfkc") == normalized


def test_normalize_lower_lowercases() -> None:
    assert normalize_text("APT HIT", "lower") == "apt hit"


def test_defang_refang_round_trips_ioc() -> None:
    original = "http://1.1.1.1"

    assert refang(defang(original)) == original


def test_iter_contexts_sentence_mode_returns_one_context_per_sample() -> None:
    samples = [_sample("test-0", ("APT", "hit")), _sample("test-1", ("CVE", "found"))]

    contexts = iter_contexts(samples, "sentence")

    assert [context.text for context in contexts] == ["APT hit", "CVE found"]
    assert contexts[0].token_refs == (("test-0", 0), ("test-0", 1))


def test_iter_contexts_window_mode_concatenates_and_preserves_back_map() -> None:
    samples = [_sample("test-0", ("APT", "hit")), _sample("test-1", ("CVE", "found"))]

    contexts = iter_contexts(samples, "window", window=2, stride=2)

    assert len(contexts) == 1
    assert contexts[0].text == "APT hit CVE found"
    assert contexts[0].token_refs == (
        ("test-0", 0),
        ("test-0", 1),
        ("test-1", 0),
        ("test-1", 1),
    )
    assert contexts[0].token_refs_for_span(8, 11) == [("test-1", 0)]
