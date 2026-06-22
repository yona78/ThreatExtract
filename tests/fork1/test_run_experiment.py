from pathlib import Path
import subprocess
import sys

from fork1.config import PRESETS, ExperimentConfig
from fork1.data import Sample, Span
from fork1.run_experiment import (
    iter_preprocessing_sweep_configs,
    prepare_samples_for_config,
    predict_samples,
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
