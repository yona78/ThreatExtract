from __future__ import annotations

from collections import Counter

from fork1.data import Sample, Span
from fork1.subsets import sample_subset


def _sample(index: int, *, labels: tuple[str, ...] = (), tokens: int = 4) -> Sample:
    spans = [
        Span(
            label=label,
            start=0,
            end=1,
            text=label,
            score=None,
            source="gold",
        )
        for label in labels
    ]
    return Sample(
        sample_id=f"test-{index:05d}",
        split="test",
        index=index,
        text=" ".join(f"t{index}_{i}" for i in range(tokens)),
        tokens=tuple(f"t{index}_{i}" for i in range(tokens)),
        tags=tuple("O" for _ in range(tokens)),
        gold_spans=spans,
    )


def _label_share(samples: list[Sample], label: str) -> float:
    counts = Counter(span.label for sample in samples for span in sample.gold_spans)
    total = sum(counts.values())
    return counts[label] / total if total else 0.0


def test_random_subset_is_seeded_and_preserves_dataset_order() -> None:
    samples = [_sample(index) for index in range(20)]

    first = sample_subset(samples, "random", "5", seed=7)
    second = sample_subset(samples, "random", "5", seed=7)

    assert [sample.sample_id for sample in first] == [sample.sample_id for sample in second]
    assert [sample.index for sample in first] == sorted(sample.index for sample in first)
    assert len(first) == 5


def test_label_stratified_preserves_top_label_share() -> None:
    samples = [
        *[_sample(index, labels=("Malware",)) for index in range(80)],
        *[_sample(index, labels=("Tool",)) for index in range(80, 100)],
    ]

    subset = sample_subset(samples, "label_stratified", "50", seed=1)

    assert len(subset) == 50
    assert abs(_label_share(subset, "Malware") - _label_share(samples, "Malware")) <= 0.05


def test_density_subset_samples_evenly_across_entity_count_buckets() -> None:
    samples = []
    index = 0
    for labels in ((), ("Malware",), ("Malware", "Tool"), ("Malware", "Tool", "Org")):
        for _ in range(10):
            samples.append(_sample(index, labels=labels))
            index += 1

    subset = sample_subset(samples, "density", "8", seed=2)

    buckets = Counter(min(len(sample.gold_spans), 3) for sample in subset)
    assert buckets == {0: 2, 1: 2, 2: 2, 3: 2}


def test_length_subset_samples_evenly_across_length_quartiles() -> None:
    samples = [_sample(index, tokens=index + 1) for index in range(16)]

    subset = sample_subset(samples, "length", "8", seed=3)

    short = sum(1 for sample in subset if len(sample.tokens) <= 4)
    medium_short = sum(1 for sample in subset if 5 <= len(sample.tokens) <= 8)
    medium_long = sum(1 for sample in subset if 9 <= len(sample.tokens) <= 12)
    long = sum(1 for sample in subset if len(sample.tokens) >= 13)
    assert (short, medium_short, medium_long, long) == (2, 2, 2, 2)


def test_hardness_subset_prefers_rare_labels_then_length() -> None:
    easy = [_sample(index, labels=("Malware",), tokens=4) for index in range(10)]
    hard = [_sample(index + 10, labels=("Way",), tokens=20) for index in range(5)]
    samples = easy + hard

    subset = sample_subset(samples, "hardness", "5", seed=4)

    assert {sample.sample_id for sample in subset} == {sample.sample_id for sample in hard}
