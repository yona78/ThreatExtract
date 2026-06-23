from __future__ import annotations

from collections.abc import Iterable

from fork1.data import Sample
from fork1.mapping import CYNER_TO_DNRTI, SECUREBERT_TO_DNRTI


CYBER_PROBE_WORDS = (
    "ransomware",
    "C2",
    "powershell",
    "T1059.001",
    "CVE-2021-44228",
    "mimikatz",
    "hxxp",
)


def tokenizer_fertility(tokenizer, words: Iterable[str]) -> float:
    counts = [len(tokenizer.tokenize(word)) for word in words]
    return sum(counts) / len(counts) if counts else 0.0


def domain_coverage(tokenizer, words: Iterable[str]) -> float:
    counts = [len(tokenizer.tokenize(word)) for word in words]
    if not counts:
        return 0.0
    return sum(1 for count in counts if count == 1) / len(counts)


def expressible_dnrti_labels(model_name: str) -> set[str]:
    normalized = model_name.lower()
    if normalized.startswith("securebert"):
        mapping = SECUREBERT_TO_DNRTI
    elif normalized.startswith("cyner"):
        mapping = CYNER_TO_DNRTI
    else:
        raise ValueError(f"unknown model for label expressibility: {model_name}")
    return set().union(*mapping.values())


def oracle_upper_bound(samples: list[Sample], model_name: str) -> dict[str, float | int]:
    expressible = expressible_dnrti_labels(model_name)
    total = sum(len(sample.gold_spans) for sample in samples)
    true_positive = sum(
        1 for sample in samples for span in sample.gold_spans if span.label in expressible
    )
    false_negative = total - true_positive
    precision = 1.0 if total else 0.0
    recall = true_positive / total if total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": 0,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "expressible_labels": sorted(expressible),
    }


def entity_surface_words(samples: list[Sample]) -> list[str]:
    words = []
    for sample in samples:
        for span in sample.gold_spans:
            words.extend(part for part in span.text.split() if part)
    return words
