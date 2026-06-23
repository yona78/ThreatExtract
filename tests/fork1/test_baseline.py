from __future__ import annotations

from fork1.baseline import (
    build_gazetteer,
    lexical_baseline_row,
    predict_lexical,
    score_exact,
)
from fork1.data import Sample, Span


def _sample(sid: str, tokens: list[str], spans: list[Span]) -> Sample:
    return Sample(
        sample_id=sid,
        split="x",
        index=0,
        text=" ".join(tokens),
        tokens=tuple(tokens),
        tags=tuple("O" for _ in tokens),
        gold_spans=spans,
    )


def test_build_gazetteer_picks_majority_label() -> None:
    train = [
        _sample("a", ["Tor"], [Span("Tool", 0, 3, "Tor", None, "gold")]),
        _sample("b", ["Tor"], [Span("Tool", 0, 3, "Tor", None, "gold")]),
        _sample("c", ["Tor"], [Span("SamFile", 0, 3, "Tor", None, "gold")]),
    ]
    assert build_gazetteer(train)["tor"] == "Tool"


def test_predict_lexical_greedy_longest_match() -> None:
    gazetteer = {"crowdstrike intelligence": "SecTeam", "crowdstrike": "SecTeam"}
    sample = _sample("a", ["CrowdStrike", "Intelligence", "reported"], [])
    spans = predict_lexical(sample, gazetteer)
    assert len(spans) == 1
    assert spans[0].text == "CrowdStrike Intelligence"
    assert spans[0].label == "SecTeam"


def test_score_exact_counts_span_and_label() -> None:
    sample = _sample(
        "a",
        ["APT28", "hit", "Russia"],
        [
            Span("HackOrg", 0, 5, "APT28", None, "gold"),
            Span("Area", 10, 16, "Russia", None, "gold"),
        ],
    )
    preds = {"a": [Span("HackOrg", 0, 5, "APT28", 1.0, "lexical")]}  # one right, one missed
    out = score_exact([sample], preds)
    assert out["true_positive"] == 1
    assert out["gold"] == 2
    assert out["precision"] == 1.0
    assert out["recall"] == 0.5


def test_lexical_baseline_row_generalizes_to_test() -> None:
    train = [_sample("tr", ["APT28"], [Span("HackOrg", 0, 5, "APT28", None, "gold")])]
    test = [_sample("te", ["APT28"], [Span("HackOrg", 0, 5, "APT28", None, "gold")])]
    row = lexical_baseline_row(train, test)
    assert row["baseline"] == "train_gazetteer"
    assert row["f1"] == 1.0
