from fork1.data import Sample, Span
from fork1.metrics import bootstrap_gap_ci, corpus_f1, mcnemar, score


def _g(start: int, end: int, label: str) -> Span:
    return Span(label=label, start=start, end=end, text="", score=None, source="gold")


def _p(start: int, end: int, label: str, score_value: float = 0.9) -> Span:
    return Span(
        label=label,
        start=start,
        end=end,
        text="",
        score=score_value,
        source="securebert",
    )


def _sample(sample_id: str, gold: list[Span]) -> Sample:
    return Sample(
        sample_id=sample_id,
        split="test",
        index=int(sample_id.rsplit("-", 1)[-1]),
        text="",
        tokens=(),
        tags=(),
        gold_spans=gold,
    )


def test_strict_exact_boundary_and_type() -> None:
    gold = [_g(0, 5, "HackOrg")]
    pred = [_p(0, 5, "APT")]

    out = score(gold, pred, "securebert")

    assert out["strict"]["f1"] == 1.0


def test_partial_credits_overlap_not_strict() -> None:
    gold = [_g(0, 10, "Tool")]
    pred = [_p(0, 5, "MAL")]

    out = score(gold, pred, "securebert")

    assert out["strict"]["f1"] == 0.0
    assert out["partial"]["f1"] > 0.0


def test_type_scheme_ignores_boundary_when_overlap_and_type_ok() -> None:
    gold = [_g(0, 10, "Area")]
    pred = [_p(3, 9, "LOC")]

    out = score(gold, pred, "securebert")

    assert out["type"]["f1"] > 0.0


def test_corpus_f1_micro_aggregates_sentence_counts() -> None:
    samples = [
        _sample("test-0", [_g(0, 5, "HackOrg")]),
        _sample("test-1", [_g(0, 5, "Tool")]),
    ]
    preds = {
        "test-0": [_p(0, 5, "APT")],
        "test-1": [_p(0, 4, "MAL")],
    }

    assert corpus_f1(samples, preds, "securebert", "strict") == 0.5


def test_bootstrap_ci_orders_low_high() -> None:
    samples = [_sample(f"test-{index}", [_g(0, 5, "HackOrg")]) for index in range(5)]
    preds = {sample.sample_id: [_p(0, 5, "APT")] for sample in samples}

    low, high, gap = bootstrap_gap_ci(
        samples,
        preds,
        preds,
        "securebert",
        "securebert",
        "strict",
        n=200,
        seed=1,
    )

    assert low <= gap <= high
    assert gap == 0.0


def test_mcnemar_counts_strict_discordance() -> None:
    samples = [
        _sample("test-0", [_g(0, 5, "HackOrg")]),
        _sample("test-1", [_g(0, 5, "HackOrg")]),
        _sample("test-2", [_g(0, 5, "HackOrg")]),
    ]
    preds_a = {
        "test-0": [_p(0, 5, "APT")],
        "test-1": [_p(0, 5, "APT")],
        "test-2": [],
    }
    preds_b = {
        "test-0": [],
        "test-1": [_p(0, 5, "APT")],
        "test-2": [],
    }

    stat, p_value = mcnemar(samples, preds_a, preds_b, "securebert", "securebert")

    assert stat == 0.0
    assert 0.0 <= p_value <= 1.0
