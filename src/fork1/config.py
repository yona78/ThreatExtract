from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    models: tuple[str, ...] = ("securebert", "cyner")
    detok: str = "single_space"
    context: str = "sentence"
    max_length: int = 256
    normalization: str = "none"
    alignment: str = "overlap"
    perturbation: str = "none"
    scheme: str = "strict"
    subset_strategy: str = "all"
    subset_size: str = "all"
    seed: int = 20260621
    bootstrap: int = 10000


PRESETS: dict[str, ExperimentConfig] = {
    "pdf_mapping": ExperimentConfig(name="pdf_mapping"),
    "paper_native": ExperimentConfig(
        name="paper_native",
        detok="punct_aware",
        context="sentence",
        max_length=128,
        alignment="overlap",
    ),
}
