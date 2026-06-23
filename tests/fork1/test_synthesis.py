import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from fork1.synthesis import (
    bias_adjustment,
    jsonable,
    run_synthesis,
    summarize_decisions,
    write_jsonl,
)


def test_summarize_decisions_counts_securebert_wins_ties_and_losses() -> None:
    rows = [
        {
            "source": "protocol",
            "config": "pdf_mapping",
            "model": "securebert",
            "scheme": "strict",
            "f1": 0.3,
            "gap_vs_other": 0.1,
            "ci_low": 0.01,
            "ci_high": 0.2,
        },
        {
            "source": "subset",
            "config": "subset=random,size=10,seed=1",
            "model": "securebert",
            "scheme": "strict",
            "f1": 0.2,
            "gap_vs_other": 0.02,
            "ci_low": -0.01,
            "ci_high": 0.08,
        },
        {
            "source": "preprocessing",
            "config": "context=document",
            "model": "securebert",
            "scheme": "strict",
            "f1": 0.01,
            "gap_vs_other": 0.007,
            "ci_low": 0.002,
            "ci_high": 0.014,
            "mcnemar_p": 0.06,
        },
        {
            "source": "stress",
            "config": "counterexample",
            "model": "securebert",
            "scheme": "strict",
            "f1": 0.1,
            "gap_vs_other": -0.05,
            "ci_low": -0.09,
            "ci_high": -0.01,
        },
    ]

    summary = summarize_decisions(rows)

    assert summary == {
        "securebert_wins": 1,
        "ties": 2,
        "cyner_wins": 1,
        "total": 4,
        "flips": [
            "subset:subset=random,size=10,seed=1",
            "preprocessing:context=document",
        ],
        "losses": ["stress:counterexample"],
    }


def test_bias_adjustment_subtracts_oracle_ceiling_gap() -> None:
    protocol_rows = [{"protocol": "pdf_mapping", "model": "securebert", "gap_vs_other": 0.1785}]
    intrinsic_rows = [
        {"model": "securebert", "oracle_f1": 0.9483},
        {"model": "cyner", "oracle_f1": 0.8490},
    ]

    adjusted = bias_adjustment(protocol_rows, intrinsic_rows)

    assert adjusted["raw_gap"] == pytest.approx(0.1785)
    assert adjusted["oracle_gap"] == pytest.approx(0.0993)
    assert adjusted["residual_gap"] == pytest.approx(0.0792)
    assert adjusted["ceiling_share"] == pytest.approx(0.5563, rel=1e-3)


def test_jsonable_converts_dataclasses_before_metadata_write() -> None:
    @dataclass
    class LoaderStats:
        malformed_lines: int

    assert jsonable({"stats": LoaderStats(malformed_lines=16)}) == {
        "stats": {"malformed_lines": 16}
    }


def test_run_synthesis_writes_master_report_metadata_and_bias_files(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    write_jsonl(
        reports / "protocol_comparison.jsonl",
        [
            {
                "protocol": "pdf_mapping",
                "config": "pdf_mapping",
                "model": "securebert",
                "scheme": "strict",
                "strict_f1": 0.28,
                "gap_vs_other": 0.17,
                "ci_low": 0.1,
                "ci_high": 0.2,
                "mcnemar_p": 0.001,
                "flip": False,
            },
            {
                "protocol": "pdf_mapping",
                "config": "pdf_mapping",
                "model": "cyner",
                "scheme": "strict",
                "strict_f1": 0.11,
                "gap_vs_other": -0.17,
                "ci_low": -0.2,
                "ci_high": -0.1,
                "mcnemar_p": 0.001,
                "flip": False,
            },
        ],
    )
    write_jsonl(
        reports / "intrinsic_metrics.jsonl",
        [
            {
                "model": "securebert",
                "oracle_f1": 0.95,
                "oracle_recall": 0.9,
                "expressible_labels": ["Area"],
            },
            {
                "model": "cyner",
                "oracle_f1": 0.85,
                "oracle_recall": 0.7,
                "expressible_labels": ["Exp"],
            },
        ],
    )
    write_jsonl(
        reports / "operational_envelope.jsonl",
        [
            {
                "device": "cpu",
                "model": "securebert",
                "token_length": 256,
                "batch_size": 1,
                "p50_ms": 75.0,
                "rss_after_load_mb": 300.0,
                "cache_size_mb": 950.0,
            },
            {
                "device": "cpu",
                "model": "cyner",
                "token_length": 256,
                "batch_size": 1,
                "p50_ms": 125.0,
                "rss_after_load_mb": 800.0,
                "cache_size_mb": 1072.0,
            },
        ],
    )
    write_jsonl(
        reports / "calibration_summary.jsonl",
        [
            {"model": "securebert", "ece": 0.65, "recommended_threshold": None},
            {"model": "cyner", "ece": 0.70, "recommended_threshold": None},
        ],
    )

    run_synthesis(reports_dir=reports, out_dir=reports, dnrti_dir=None, cache_dir=Path("cache"))

    assert (reports / "master_table.jsonl").is_file()
    assert (reports / "master_table.md").is_file()
    assert (reports / "benchmark_summary.md").is_file()
    assert (reports / "run_metadata.json").is_file()
    assert (reports / "leakage_bias.md").is_file()
    assert (reports / "figures" / "bias_chain.svg").is_file()
    text = (reports / "benchmark_summary.md").read_text(encoding="utf-8")
    assert "bias-adjusted residual" in text
    assert "CPU deployment note" in text
    metadata = json.loads((reports / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["reproduce"]["command"] == "make reproduce"
