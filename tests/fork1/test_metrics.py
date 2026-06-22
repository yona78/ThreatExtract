from fork1.data import Span
from fork1.metrics import score


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
