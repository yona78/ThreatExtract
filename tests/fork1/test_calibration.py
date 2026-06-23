from fork1.calibration import entity_records, expected_calibration_error, threshold_sweep
from fork1.data import Sample, Span


def _sample() -> Sample:
    return Sample(
        sample_id="test-0",
        split="test",
        index=0,
        text="APT CVE",
        tokens=("APT", "CVE"),
        tags=("B-HackOrg", "B-Exp"),
        gold_spans=[
            Span(label="HackOrg", start=0, end=3, text="APT", score=None, source="gold"),
            Span(label="Exp", start=4, end=7, text="CVE", score=None, source="gold"),
        ],
    )


def test_entity_records_marks_strict_true_positive_once() -> None:
    predictions = {
        "test-0": [
            Span(label="APT", start=0, end=3, text="APT", score=0.9, source="securebert"),
            Span(label="APT", start=0, end=3, text="APT", score=0.8, source="securebert"),
            Span(label="VULID", start=4, end=7, text="CVE", score=0.7, source="securebert"),
        ]
    }

    records = entity_records([_sample()], predictions, "securebert")

    assert records == [
        {"score": 0.9, "is_correct": True},
        {"score": 0.8, "is_correct": False},
        {"score": 0.7, "is_correct": True},
    ]


def test_expected_calibration_error_is_zero_for_matching_confidence_and_precision() -> None:
    records = [
        {"score": 0.5, "is_correct": True},
        {"score": 0.5, "is_correct": False},
    ]

    assert expected_calibration_error(records, bins=1) == 0.0


def test_threshold_sweep_reports_precision_and_gold_recall() -> None:
    records = [
        {"score": 0.9, "is_correct": True},
        {"score": 0.8, "is_correct": False},
        {"score": 0.7, "is_correct": True},
    ]

    rows = threshold_sweep(records, [0.0, 0.85], total_gold=4)

    assert rows[0]["precision"] == 2 / 3
    assert rows[0]["recall"] == 0.5
    assert rows[1]["precision"] == 1.0
    assert rows[1]["recall"] == 0.25
