import math

from fork1.data import Sample, Span
from fork1.metrics import (
    bootstrap_count_gap_ci,
    bootstrap_gap_ci,
    corpus_f1,
    corpus_scores,
    entity_confusion_rows,
    entity_error_records,
    mcnemar,
    oov_entity_rows,
    per_label_strict_rows,
    sample_muc_counts,
    score,
)


def _g(start: int, end: int, label: str, text: str = "") -> Span:
    return Span(label=label, start=start, end=end, text=text, score=None, source="gold")


def _p(start: int, end: int, label: str, score_value: float = 0.9, text: str = "") -> Span:
    return Span(
        label=label,
        start=start,
        end=end,
        text=text,
        score=score_value,
        source="securebert",
    )


def _sample(sample_id: str, gold: list[Span], text: str = "") -> Sample:
    return Sample(
        sample_id=sample_id,
        split="test",
        index=int(sample_id.rsplit("-", 1)[-1]),
        text=text,
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


def test_corpus_scores_reports_all_four_schemes() -> None:
    samples = [
        _sample("test-0", [_g(0, 5, "HackOrg")]),
        _sample("test-1", [_g(0, 5, "Tool")]),
    ]
    preds = {
        "test-0": [_p(0, 5, "APT")],
        "test-1": [_p(0, 4, "MAL")],
    }

    out = corpus_scores(samples, preds, "securebert")

    assert set(out) == {"strict", "exact", "partial", "type"}
    assert out["strict"]["f1"] == 0.5
    assert out["partial"]["f1"] > out["strict"]["f1"]


def test_sample_muc_counts_preserve_sample_order() -> None:
    samples = [
        _sample("test-0", [_g(0, 5, "HackOrg")]),
        _sample("test-1", [_g(0, 5, "Tool")]),
    ]
    preds = {
        "test-0": [_p(0, 5, "APT")],
        "test-1": [_p(0, 4, "MAL")],
    }

    counts = sample_muc_counts(samples, preds, "securebert", "strict")

    assert [item.cor for item in counts] == [1, 0]
    assert [item.inc for item in counts] == [0, 1]


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


def test_bootstrap_count_gap_ci_orders_low_high() -> None:
    samples = [_sample(f"test-{index}", [_g(0, 5, "HackOrg")]) for index in range(5)]
    preds = {sample.sample_id: [_p(0, 5, "APT")] for sample in samples}
    counts = sample_muc_counts(samples, preds, "securebert", "strict")

    low, high, gap = bootstrap_count_gap_ci(counts, counts, "strict", n=200, seed=1)

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


def test_mcnemar_uses_chi_square_survival_for_df1() -> None:
    samples = [_sample(f"test-{index}", [_g(0, 5, "HackOrg")]) for index in range(3)]
    preds_a = {sample.sample_id: [_p(0, 5, "APT")] for sample in samples}
    preds_b = {sample.sample_id: [] for sample in samples}

    stat, p_value = mcnemar(samples, preds_a, preds_b, "securebert", "securebert")

    assert stat == (3 - 1) ** 2 / 3
    assert p_value == math.erfc(math.sqrt(stat / 2))


def test_entity_error_records_cover_strict_type_boundary_fn_and_fp() -> None:
    samples = [
        _sample("test-0", [_g(0, 3, "HackOrg", "APT")], text="APT"),
        _sample("test-1", [_g(0, 4, "Tool", "MALX")], text="MALX"),
        _sample("test-2", [_g(0, 5, "Tool", "MALXX")], text="MALXX"),
        _sample("test-3", [_g(0, 3, "Org", "ORG")], text="ORG"),
        _sample("test-4", [], text="ROME"),
    ]
    preds = {
        "test-0": [_p(0, 3, "APT", text="APT")],
        "test-1": [_p(0, 4, "LOC", text="MALX")],
        "test-2": [_p(0, 4, "MAL", text="MALX")],
        "test-3": [],
        "test-4": [_p(0, 4, "LOC", text="ROME")],
    }

    rows = entity_error_records(samples, preds, "securebert")

    assert [row["bucket"] for row in rows] == [
        "strict_correct",
        "type_error",
        "boundary_error",
        "strict_drop_fn",
        "spurious_fp",
    ]
    assert rows[1]["gold_label"] == "Tool"
    assert rows[1]["predicted_label"] == "Area"
    assert rows[3]["predicted_label"] == "O/MISSED"
    assert rows[4]["gold_label"] == "SPURIOUS"


def test_entity_confusion_rows_include_missed_and_spurious() -> None:
    samples = [
        _sample("test-0", [_g(0, 3, "HackOrg", "APT")], text="APT"),
        _sample("test-1", [_g(0, 4, "Tool", "MALX")], text="MALX"),
        _sample("test-2", [_g(0, 3, "Org", "ORG")], text="ORG"),
        _sample("test-3", [], text="ROME"),
    ]
    preds = {
        "test-0": [_p(0, 3, "APT", text="APT")],
        "test-1": [_p(0, 4, "LOC", text="MALX")],
        "test-2": [],
        "test-3": [_p(0, 4, "LOC", text="ROME")],
    }

    rows = entity_confusion_rows(samples, preds, "securebert")
    by_pair = {(row["gold_label"], row["predicted_label"]): row["count"] for row in rows}

    assert by_pair[("HackOrg", "HackOrg")] == 1
    assert by_pair[("Tool", "Area")] == 1
    assert by_pair[("Org", "O/MISSED")] == 1
    assert by_pair[("SPURIOUS", "Area")] == 1


def test_per_label_strict_rows_report_projection_aware_prf() -> None:
    samples = [
        _sample("test-0", [_g(0, 3, "HackOrg", "APT")], text="APT"),
        _sample("test-1", [_g(0, 4, "Tool", "MALX")], text="MALX"),
        _sample("test-2", [_g(0, 5, "Tool", "MALXX")], text="MALXX"),
    ]
    preds = {
        "test-0": [_p(0, 3, "APT", text="APT")],
        "test-1": [_p(0, 4, "LOC", text="MALX")],
        "test-2": [_p(0, 4, "MAL", text="MALX")],
    }

    rows = per_label_strict_rows(samples, preds, "securebert")
    by_label = {row["label"]: row for row in rows}

    assert by_label["HackOrg"]["true_positive"] == 1
    assert by_label["HackOrg"]["f1"] == 1.0
    assert by_label["Tool"]["false_negative"] == 2
    assert by_label["Tool"]["false_positive"] == 1
    assert by_label["Area"]["false_positive"] == 1


def test_oov_entity_rows_split_seen_and_unseen_surfaces() -> None:
    train = [
        _sample("train-0", [_g(0, 3, "HackOrg", "APT")], text="APT"),
    ]
    test = [
        _sample("test-0", [_g(0, 3, "HackOrg", "APT")], text="APT"),
        _sample("test-1", [_g(0, 6, "HackOrg", "NEWAPT")], text="NEWAPT"),
    ]
    preds = {
        "test-0": [_p(0, 3, "APT", text="APT")],
        "test-1": [],
    }

    rows = oov_entity_rows(train, test, preds, "securebert")
    by_status = {row["surface_status"]: row for row in rows}

    assert by_status["seen"]["support"] == 1
    assert by_status["seen"]["recall"] == 1.0
    assert by_status["unseen"]["support"] == 1
    assert by_status["unseen"]["recall"] == 0.0
