from fork1.config import PRESETS, ExperimentConfig


def test_pdf_mapping_preset_matches_default_experiment_config() -> None:
    expected = ExperimentConfig(name="pdf_mapping")

    assert PRESETS["pdf_mapping"] == expected
    assert expected.models == ("securebert", "cyner")
    assert expected.bootstrap == 10000
    assert expected.perturbation == "none"


def test_paper_native_preset_uses_paper_protocol_settings() -> None:
    preset = PRESETS["paper_native"]

    assert preset.detok == "punct_aware"
    assert preset.context == "sentence"
    assert preset.max_length == 128
    assert preset.alignment == "overlap"
