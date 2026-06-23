from pathlib import Path

import pytest

from benchmark import (
    Span,
    compute_match_counts,
    extract_bio_spans,
    load_dnrti_split,
    map_model_label_to_dnrti,
    relaxed_iou_match,
    select_winner,
    write_final_selection,
)


def write_split(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_load_dnrti_split_skips_tag_only_o_lines(tmp_path: Path) -> None:
    split = write_split(
        tmp_path,
        "test.txt",
        "APT B-HackOrg\n" "28 O\n" "O\n" "Poison B-Tool\n" "Ivy I-Tool\n" "\n",
    )

    samples, warnings = load_dnrti_split(split, split_name="test")

    assert len(samples) == 1
    assert samples[0].text == "APT 28 Poison Ivy"
    assert warnings == ["test.txt:3: skipped tag-only line 'O'"]
    assert samples[0].gold_spans == [
        Span(label="HackOrg", start=0, end=3, text="APT", score=None, source="gold"),
        Span(label="Tool", start=7, end=17, text="Poison Ivy", score=None, source="gold"),
    ]


def test_extract_bio_spans_closes_on_new_b_tag_and_label_change() -> None:
    tokens = ["APT", "28", "CVE-1", "Poison", "Ivy"]
    tags = ["B-HackOrg", "I-HackOrg", "I-Exp", "B-Tool", "I-Tool"]
    text = "APT 28 CVE-1 Poison Ivy"
    offsets = [(0, 3), (4, 6), (7, 12), (13, 19), (20, 23)]

    assert extract_bio_spans(tokens, tags, text, offsets) == [
        Span(label="HackOrg", start=0, end=6, text="APT 28", score=None, source="gold"),
        Span(label="Exp", start=7, end=12, text="CVE-1", score=None, source="gold"),
        Span(label="Tool", start=13, end=23, text="Poison Ivy", score=None, source="gold"),
    ]


@pytest.mark.parametrize(
    ("model", "label", "expected"),
    [
        ("securebert", "APT", {"HackOrg"}),
        ("securebert", "SECTEAM", {"SecTeam"}),
        ("securebert", "IDTY", {"Idus", "Org"}),
        ("securebert", "ACT", {"OffAct", "Way"}),
        ("securebert", "OS", {"OffAct", "Way"}),
        ("securebert", "TOOL", {"OffAct", "Way"}),
        ("securebert", "MAL", {"Tool"}),
        ("securebert", "FILE", {"SamFile"}),
        ("securebert", "LOC", {"Area"}),
        ("cyner", "Organization", {"HackOrg", "SecTeam", "Idus", "Org"}),
        ("cyner", "System", {"OffAct", "Way"}),
        ("cyner", "Vulnerability", {"Exp"}),
        ("cyner", "Malware", {"Tool"}),
        ("cyner", "Indicator", {"SamFile"}),
    ],
)
def test_assignment_label_mapping(model: str, label: str, expected: set[str]) -> None:
    assert map_model_label_to_dnrti(model, label) == expected


def test_compute_match_counts_exact_handles_one_to_many_mapping() -> None:
    gold = [
        Span(label="HackOrg", start=0, end=3, text="APT", score=None, source="gold"),
        Span(label="Org", start=8, end=17, text="Microsoft", score=None, source="gold"),
    ]
    preds = [
        Span(label="Organization", start=0, end=3, text="APT", score=0.9, source="cyner"),
        Span(label="Organization", start=8, end=16, text="Microsof", score=0.9, source="cyner"),
    ]

    counts = compute_match_counts(gold, preds, model_name="cyner", relaxed=False)

    assert counts.true_positive == 1
    assert counts.false_positive == 1
    assert counts.false_negative == 1


def test_relaxed_iou_match_requires_overlap_and_mapped_label() -> None:
    gold = Span(label="Org", start=10, end=20, text="Microsoft", score=None, source="gold")
    close_pred = Span(
        label="Organization", start=12, end=20, text="crosoft", score=0.8, source="cyner"
    )
    wrong_label = Span(label="System", start=12, end=20, text="crosoft", score=0.8, source="cyner")

    assert relaxed_iou_match(gold, close_pred, model_name="cyner", threshold=0.5)
    assert not relaxed_iou_match(gold, wrong_label, model_name="cyner", threshold=0.5)


def test_select_winner_prefers_all_subset_exact_f1() -> None:
    rows = [
        {
            "model": "cyner",
            "samples": 10,
            "subset_size": "10",
            "metrics": {"exact": {"f1": 0.5}},
        },
        {
            "model": "securebert",
            "samples": 664,
            "subset_size": "all",
            "metrics": {"exact": {"f1": 0.28}},
        },
        {
            "model": "cyner",
            "samples": 664,
            "subset_size": "all",
            "metrics": {"exact": {"f1": 0.10}},
        },
    ]

    assert select_winner(rows)["winner"] == "securebert"


def test_write_final_selection_reports_evidence_without_deploy_decision(tmp_path: Path) -> None:
    rows = [
        {
            "model": "securebert",
            "samples": 664,
            "subset_size": "all",
            "elapsed_seconds": 1.0,
            "rss_peak_mb": 10.0,
            "metrics": {
                "exact": {"f1": 0.28, "precision": 0.22, "recall": 0.38},
                "relaxed": {"f1": 0.5},
            },
        },
        {
            "model": "cyner",
            "samples": 664,
            "subset_size": "all",
            "elapsed_seconds": 2.0,
            "rss_peak_mb": 20.0,
            "metrics": {
                "exact": {"f1": 0.10, "precision": 0.12, "recall": 0.09},
                "relaxed": {"f1": 0.26},
            },
        },
    ]

    write_final_selection(tmp_path / "final_selection.md", rows, model_loads={})

    text = (tmp_path / "final_selection.md").read_text(encoding="utf-8")
    assert "Evidence leader: **securebert**." in text
    assert "strict entity F1" in text
    assert "exact micro-F1" not in text
    assert "Decision: deploy" not in text
