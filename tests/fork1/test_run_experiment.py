from pathlib import Path
import subprocess
import sys

import fork1.run_experiment as run_experiment
import pytest
from fork1.config import PRESETS, ExperimentConfig
from fork1.data import Sample, Span
from fork1.run_experiment import (
    apply_alignment_policy,
    bio_tags_from_label_sets,
    iter_preprocessing_sweep_configs,
    iter_protocol_configs,
    iter_subset_study_configs,
    min_faithful_subset_by_strategy,
    mapping_coverage_for_model,
    prepare_samples_for_config,
    predict_samples,
    run_protocol_comparison,
    run_subset_study,
    seqeval_cross_check,
    subset_rows_from_predictions,
    summarize_subset_cells,
    write_protocol_comparison_report,
    write_robustness_report,
    write_subset_study_report,
    write_preprocessing_tornado,
    write_preprocessing_report,
)


def _sample() -> Sample:
    return Sample(
        sample_id="test-0",
        split="test",
        index=0,
        text="APT hit .",
        tokens=("APT", "hit", "."),
        tags=("B-HackOrg", "O", "O"),
        gold_spans=[],
    )


def test_preprocessing_sweep_configs_vary_one_lever_from_pdf_mapping() -> None:
    configs = list(iter_preprocessing_sweep_configs(PRESETS["pdf_mapping"]))
    names = {config.name for config in configs}

    assert "detok=punct_aware" in names
    assert "context=window" in names
    assert "max_length=128" in names
    assert "normalization=lower" in names
    assert "alignment=contained" in names
    assert all(
        sum(
            getattr(config, field) != getattr(PRESETS["pdf_mapping"], field)
            for field in ("detok", "context", "max_length", "normalization", "alignment")
        )
        <= 1
        for config in configs
    )


def test_subset_study_configs_cover_strategy_size_seed_grid() -> None:
    configs = list(iter_subset_study_configs(PRESETS["pdf_mapping"]))

    assert len(configs) == 75
    assert {(config.subset_strategy, config.subset_size, config.seed) for config in configs} >= {
        ("random", "10", 1),
        ("label_stratified", "100", 2),
        ("density", "250", 3),
        ("length", "all", 1),
        ("hardness", "50", 3),
    }
    assert all(config.detok == "single_space" for config in configs)
    assert all(config.alignment == "overlap" for config in configs)


def test_protocol_configs_compare_pdf_mapping_and_paper_native() -> None:
    configs = list(iter_protocol_configs())

    assert [config.name for config in configs] == ["pdf_mapping", "paper_native"]
    assert configs[0].detok == "single_space"
    assert configs[1].detok == "punct_aware"
    assert configs[1].max_length == 128


def test_prepare_samples_rebuilds_text_offsets_for_config_detok() -> None:
    samples = [_sample()]

    prepared = prepare_samples_for_config(
        samples,
        ExperimentConfig(name="punct", detok="punct_aware"),
    )

    assert prepared[0].text == "APT hit."
    assert prepared[0].gold_spans == [
        Span(label="HackOrg", start=0, end=3, text="APT", score=None, source="gold")
    ]


def test_prepare_samples_applies_named_perturbation_before_detok() -> None:
    sample = Sample(
        sample_id="test-0",
        split="test",
        index=0,
        text="",
        tokens=("http://1.1.1.1",),
        tags=("B-SamFile",),
        gold_spans=[],
    )

    prepared = prepare_samples_for_config(
        [sample],
        ExperimentConfig(name="defang", perturbation="defang"),
    )

    assert prepared[0].text == "hxxp://1[.]1[.]1[.]1"
    assert prepared[0].gold_spans[0].text == "hxxp://1[.]1[.]1[.]1"


def test_prepare_samples_applies_subset_config() -> None:
    samples = [
        Sample(
            sample_id=f"test-{index}",
            split="test",
            index=index,
            text=f"token{index}",
            tokens=(f"token{index}",),
            tags=("O",),
            gold_spans=[],
        )
        for index in range(6)
    ]

    prepared = prepare_samples_for_config(
        samples,
        ExperimentConfig(
            name="subset",
            subset_strategy="random",
            subset_size="2",
            seed=11,
        ),
    )

    assert len(prepared) == 2
    assert [sample.index for sample in prepared] == sorted(sample.index for sample in prepared)


def test_subset_rows_from_predictions_scores_selected_samples_only() -> None:
    samples = [
        Sample(
            sample_id=f"test-{index}",
            split="test",
            index=index,
            text="APT",
            tokens=("APT",),
            tags=("B-HackOrg",),
            gold_spans=[
                Span(label="HackOrg", start=0, end=3, text="APT", score=None, source="gold")
            ],
        )
        for index in range(4)
    ]
    predictions = {
        "securebert": {
            "test-0": [
                Span(label="APT", start=0, end=3, text="APT", score=0.9, source="securebert")
            ],
            "test-1": [
                Span(label="APT", start=0, end=3, text="APT", score=0.9, source="securebert")
            ],
            "test-2": [],
            "test-3": [],
        },
        "cyner": {sample.sample_id: [] for sample in samples},
    }

    rows = subset_rows_from_predictions(
        ExperimentConfig(
            name="subset=random,size=2,seed=3",
            subset_strategy="random",
            subset_size="2",
            seed=3,
            bootstrap=0,
        ),
        samples,
        predictions,
    )

    assert {row["model"] for row in rows} == {"securebert", "cyner"}
    assert {row["samples"] for row in rows} == {2}
    assert all(row["subset_strategy"] == "random" for row in rows)
    assert all(row["subset_size"] == "2" for row in rows)


def test_bio_tags_from_label_sets_emits_b_i_boundaries() -> None:
    tags = bio_tags_from_label_sets(
        [
            {"HackOrg"},
            {"HackOrg"},
            set(),
            {"Tool"},
            {"Tool", "Exp"},
            {"Tool"},
        ]
    )

    assert tags == ["B-HackOrg", "I-HackOrg", "O", "B-Tool", "O", "B-Tool"]


def test_mapping_coverage_counts_unique_label_subset() -> None:
    samples = [
        Sample(
            sample_id="test-0",
            split="test",
            index=0,
            text="APT action",
            tokens=("APT", "action"),
            tags=("B-HackOrg", "B-Way"),
            gold_spans=[
                Span(label="HackOrg", start=0, end=3, text="APT", score=None, source="gold"),
                Span(label="Way", start=4, end=10, text="action", score=None, source="gold"),
            ],
        )
    ]

    coverage = mapping_coverage_for_model(samples, "securebert")

    assert coverage["total_gold_spans"] == 2
    assert coverage["unique_gold_spans"] == 1
    assert coverage["non_unique_gold_spans"] == 1
    assert coverage["non_unique_labels"] == {"Way": 1}


def test_seqeval_cross_check_matches_strict_on_unique_labels() -> None:
    pytest.importorskip("seqeval")
    samples = [
        Sample(
            sample_id="test-0",
            split="test",
            index=0,
            text="APT",
            tokens=("APT",),
            tags=("B-HackOrg",),
            gold_spans=[
                Span(label="HackOrg", start=0, end=3, text="APT", score=None, source="gold")
            ],
        )
    ]
    predictions = {
        "test-0": [Span(label="APT", start=0, end=3, text="APT", score=0.9, source="securebert")]
    }

    check = seqeval_cross_check(
        samples,
        predictions,
        "securebert",
        ExperimentConfig(name="pdf_mapping", bootstrap=0),
    )

    assert check["our_f1"] == 1.0
    assert check["seqeval_f1"] == 1.0
    assert check["delta"] == 0.0


class FakeRunner:
    def predict(self, text: str, max_length: int | None = None):
        assert text == "APT hit."
        return [Span(label="APT", start=7, end=8, text=".", score=0.9, source="securebert")]


def test_predict_samples_projects_context_spans_with_active_detok_offsets() -> None:
    prepared = prepare_samples_for_config(
        [_sample()],
        ExperimentConfig(name="punct", detok="punct_aware"),
    )

    predictions = predict_samples(
        prepared,
        FakeRunner(),
        ExperimentConfig(name="punct", detok="punct_aware"),
    )

    assert predictions["test-0"][0].start == 7
    assert predictions["test-0"][0].end == 8
    assert predictions["test-0"][0].text == "."


class BoundaryRunner:
    def predict(self, text: str, max_length: int | None = None):
        assert text == "APT hit ."
        return [Span(label="APT", start=1, end=3, text="PT", score=0.9, source="securebert")]


def test_predict_samples_preserves_raw_char_boundaries_in_sentence_context() -> None:
    prepared = prepare_samples_for_config(
        [_sample()],
        ExperimentConfig(name="single", detok="single_space"),
    )

    predictions = predict_samples(
        prepared,
        BoundaryRunner(),
        ExperimentConfig(name="single", detok="single_space"),
    )

    assert predictions["test-0"][0].start == 1
    assert predictions["test-0"][0].end == 3
    assert predictions["test-0"][0].text == "PT"


def test_contained_alignment_drops_partial_token_prediction() -> None:
    prepared = prepare_samples_for_config(
        [_sample()],
        ExperimentConfig(name="single", detok="single_space"),
    )[0]
    partial = Span(label="APT", start=1, end=3, text="PT", score=0.9, source="securebert")

    aligned = apply_alignment_policy(
        prepared,
        [partial],
        ExperimentConfig(name="contained", alignment="contained"),
    )

    assert aligned == []


def test_write_preprocessing_report_includes_ci_and_flip_columns(tmp_path: Path) -> None:
    rows = [
        {
            "config": "detok=single_space",
            "model": "securebert",
            "strict_f1": 0.28,
            "gap_vs_other": 0.17,
            "ci_low": 0.1,
            "ci_high": 0.2,
            "flip": False,
        }
    ]

    write_preprocessing_report(tmp_path / "preprocessing.md", rows)

    text = (tmp_path / "preprocessing.md").read_text(encoding="utf-8")
    assert "| Config | Model | Strict F1 | Gap | 95% CI | Flip? |" in text
    assert "detok=single_space" in text


def test_write_preprocessing_tornado_groups_swing_by_lever(tmp_path: Path) -> None:
    rows = [
        {"config": "detok=single_space", "model": "securebert", "strict_f1": 0.28},
        {"config": "detok=punct_aware", "model": "securebert", "strict_f1": 0.27},
        {"config": "detok=single_space", "model": "cyner", "strict_f1": 0.10},
        {"config": "detok=punct_aware", "model": "cyner", "strict_f1": 0.09},
    ]

    write_preprocessing_tornado(tmp_path / "tornado.svg", rows)

    text = (tmp_path / "tornado.svg").read_text(encoding="utf-8")
    assert "<svg" in text
    assert "detok" in text
    assert "securebert" in text


def test_write_robustness_report_includes_delta_f1(tmp_path: Path) -> None:
    rows = [
        {
            "perturbation": "defang",
            "model": "securebert",
            "clean_f1": 0.28,
            "noisy_f1": 0.20,
            "delta_f1": -0.08,
            "ci_low": -0.10,
            "ci_high": -0.05,
            "gap_vs_other": 0.17,
            "gap_ci_low": 0.10,
            "gap_ci_high": 0.20,
            "flip": False,
        }
    ]

    write_robustness_report(tmp_path / "robustness.md", rows)

    text = (tmp_path / "robustness.md").read_text(encoding="utf-8")
    assert (
        "| Perturbation | Model | Clean F1 | Noisy F1 | Delta F1 | Delta 95% CI | "
        "Gap vs Other | Gap 95% CI | Flip? |"
    ) in text
    assert "defang" in text


def test_summarize_subset_cells_reports_variance() -> None:
    rows = [
        {
            "subset_strategy": "random",
            "subset_size": "10",
            "model": "securebert",
            "strict_f1": 0.2,
        },
        {
            "subset_strategy": "random",
            "subset_size": "10",
            "model": "securebert",
            "strict_f1": 0.4,
        },
    ]

    summary = summarize_subset_cells(rows)

    assert summary == [
        {
            "subset_strategy": "random",
            "subset_size": "10",
            "model": "securebert",
            "seeds": 2,
            "mean_f1": 0.30000000000000004,
            "variance_f1": 0.020000000000000004,
            "min_f1": 0.2,
            "max_f1": 0.4,
        }
    ]


def test_min_faithful_subset_requires_all_three_seeds_to_exclude_zero() -> None:
    rows = []
    for size, low in (("10", -0.01), ("50", 0.02)):
        for seed in (1, 2, 3):
            rows.append(
                {
                    "subset_strategy": "random",
                    "subset_size": size,
                    "subset_seed": seed,
                    "model": "securebert",
                    "gap_vs_other": 0.10,
                    "ci_low": low,
                    "ci_high": 0.20,
                }
            )
    for seed in (1, 2, 3):
        rows.append(
            {
                "subset_strategy": "random",
                "subset_size": "all",
                "subset_seed": seed,
                "model": "securebert",
                "gap_vs_other": 0.10,
                "ci_low": 0.04,
                "ci_high": 0.18,
            }
        )

    assert min_faithful_subset_by_strategy(rows, full_winner="securebert") == {"random": "50"}


def test_write_subset_study_report_includes_summary_and_min_faithful(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "subset_strategy": "random",
            "subset_size": "10",
            "subset_seed": 1,
            "samples": 10,
            "model": "securebert",
            "strict_f1": 0.2,
            "gap_vs_other": 0.1,
            "ci_low": 0.02,
            "ci_high": 0.2,
            "flip": False,
        }
    ]

    write_subset_study_report(
        tmp_path / "subset_study.md",
        rows,
        min_faithful={"random": "10"},
    )

    text = (tmp_path / "subset_study.md").read_text(encoding="utf-8")
    assert "| Strategy | Size | Model | Seeds | Mean F1 | Variance | Min F1 | Max F1 |" in text
    assert "| random | 10 |" in text
    assert "min-faithful subset" in text


def test_write_protocol_comparison_report_includes_rows_checks_and_coverage(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "protocol": "pdf_mapping",
            "model": "securebert",
            "strict_f1": 0.28,
            "gap_vs_other": 0.17,
            "ci_low": 0.10,
            "ci_high": 0.20,
            "flip": False,
        }
    ]
    checks = [
        {
            "protocol": "pdf_mapping",
            "model": "securebert",
            "our_f1": 1.0,
            "seqeval_f1": 1.0,
            "delta": 0.0,
            "unique_gold_spans": 1,
        }
    ]
    coverage = [
        {
            "model": "securebert",
            "total_gold_spans": 2,
            "unique_gold_spans": 1,
            "non_unique_gold_spans": 1,
            "coverage": 0.5,
            "non_unique_labels": {"Way": 1},
        }
    ]

    write_protocol_comparison_report(
        tmp_path / "protocol_comparison.md",
        rows,
        checks,
        coverage,
    )

    text = (tmp_path / "protocol_comparison.md").read_text(encoding="utf-8")
    assert "| Protocol | Model | Strict F1 | Gap | 95% CI | Flip? |" in text
    assert "Seqeval Cross-Check" in text
    assert "Unique-Label Coverage" in text


def test_run_subset_study_reuses_full_predictions(monkeypatch, tmp_path: Path) -> None:
    samples = [
        Sample(
            sample_id=f"test-{index}",
            split="test",
            index=index,
            text="APT",
            tokens=("APT",),
            tags=("B-HackOrg",),
            gold_spans=[
                Span(label="HackOrg", start=0, end=3, text="APT", score=None, source="gold")
            ],
        )
        for index in range(4)
    ]
    predictions = {
        "securebert": {
            sample.sample_id: [
                Span(label="APT", start=0, end=3, text="APT", score=0.9, source="securebert")
            ]
            for sample in samples
        },
        "cyner": {sample.sample_id: [] for sample in samples},
    }
    calls = []

    def fake_load_dnrti_dataset(dnrti_dir, split):
        return samples, [], []

    def fake_run_config_with_predictions(config, samples_arg, *, device, offline, cache_dir):
        calls.append(config)
        return samples_arg, predictions, []

    monkeypatch.setattr(run_experiment, "load_dnrti_dataset", fake_load_dnrti_dataset)
    monkeypatch.setattr(
        run_experiment,
        "run_config_with_predictions",
        fake_run_config_with_predictions,
    )
    monkeypatch.setattr(
        run_experiment,
        "iter_subset_study_configs",
        lambda: [
            ExperimentConfig(
                name="subset=random,size=10,seed=1",
                subset_strategy="random",
                subset_size="10",
                seed=1,
                bootstrap=0,
            )
        ],
    )

    rows = run_subset_study(
        dnrti_dir=tmp_path / "dnrti",
        out_dir=tmp_path / "reports",
        device="mps",
        offline=True,
        cache_dir=tmp_path / "cache",
        bootstrap=0,
    )

    assert len(calls) == 1
    assert rows
    assert (tmp_path / "reports" / "subset_study.jsonl").is_file()
    assert (tmp_path / "reports" / "subset_study.md").is_file()
    assert (tmp_path / "reports" / "figures" / "subset_random.svg").is_file()


def test_run_protocol_comparison_writes_report_and_jsonl(monkeypatch, tmp_path: Path) -> None:
    samples = [
        Sample(
            sample_id="test-0",
            split="test",
            index=0,
            text="APT",
            tokens=("APT",),
            tags=("B-HackOrg",),
            gold_spans=[
                Span(label="HackOrg", start=0, end=3, text="APT", score=None, source="gold")
            ],
        )
    ]
    predictions = {
        "securebert": {
            "test-0": [
                Span(label="APT", start=0, end=3, text="APT", score=0.9, source="securebert")
            ]
        },
        "cyner": {"test-0": []},
    }
    rows = [
        {
            "config": "pdf_mapping",
            "model": "securebert",
            "scheme": "strict",
            "strict_f1": 1.0,
            "gap_vs_other": 1.0,
            "ci_low": 1.0,
            "ci_high": 1.0,
            "mcnemar_stat": 0.0,
            "mcnemar_p": 1.0,
            "flip": False,
        },
        {
            "config": "pdf_mapping",
            "model": "cyner",
            "scheme": "strict",
            "strict_f1": 0.0,
            "gap_vs_other": -1.0,
            "ci_low": -1.0,
            "ci_high": -1.0,
            "mcnemar_stat": 0.0,
            "mcnemar_p": 1.0,
            "flip": False,
        },
    ]

    def fake_load_dnrti_dataset(dnrti_dir, split):
        return samples, [], []

    def fake_run_config_with_predictions(config, samples_arg, *, device, offline, cache_dir):
        return samples_arg, predictions, rows

    monkeypatch.setattr(run_experiment, "load_dnrti_dataset", fake_load_dnrti_dataset)
    monkeypatch.setattr(
        run_experiment,
        "run_config_with_predictions",
        fake_run_config_with_predictions,
    )
    monkeypatch.setattr(
        run_experiment,
        "seqeval_cross_check",
        lambda prepared, preds, model, config: {
            "protocol": config.name,
            "model": model,
            "our_f1": 1.0,
            "seqeval_f1": 1.0,
            "delta": 0.0,
            "unique_gold_spans": 1,
        },
    )

    out = run_protocol_comparison(
        dnrti_dir=tmp_path / "dnrti",
        out_dir=tmp_path / "reports",
        device="mps",
        offline=True,
        cache_dir=tmp_path / "cache",
        bootstrap=0,
    )

    assert out
    assert (tmp_path / "reports" / "protocol_comparison.jsonl").is_file()
    assert (tmp_path / "reports" / "protocol_seqeval_crosscheck.jsonl").is_file()
    assert (tmp_path / "reports" / "protocol_comparison.md").is_file()


def test_run_experiment_module_help_works_from_repo_root() -> None:
    root = Path(__file__).resolve().parents[2]

    result = subprocess.run(
        [sys.executable, "-m", "fork1.run_experiment", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--sweep" in result.stdout
    assert "subset" in result.stdout
    assert "protocol" in result.stdout
