from __future__ import annotations

import random
from collections import Counter, defaultdict
from math import floor

from fork1.data import Sample


STRATEGIES = frozenset({"random", "label_stratified", "density", "length", "hardness"})
RARE_LABELS = frozenset({"Way", "Purp", "Features"})


def sample_subset(samples: list[Sample], strategy: str, size: str | int, seed: int) -> list[Sample]:
    if strategy == "all":
        return list(samples)
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown subset strategy: {strategy}")

    requested = _parse_size(size)
    if requested >= len(samples):
        return list(samples)
    if requested <= 0:
        return []

    rng = random.Random(seed)
    if strategy == "random":
        return _sample_indices(samples, rng.sample(range(len(samples)), requested))
    if strategy == "label_stratified":
        return _label_stratified_subset(samples, requested, rng)
    if strategy == "density":
        return _bucketed_subset(
            samples,
            requested,
            rng,
            key=lambda sample: min(len(sample.gold_spans), 3),
        )
    if strategy == "length":
        return _bucketed_subset(
            samples,
            requested,
            rng,
            key=_length_quartile_keys(samples),
        )
    return _hardness_subset(samples, requested, rng)


def _parse_size(size: str | int) -> int:
    if size == "all":
        return 10**18
    return int(size)


def _sample_indices(samples: list[Sample], indices: list[int]) -> list[Sample]:
    selected = set(indices)
    return [sample for index, sample in enumerate(samples) if index in selected]


def _proportional_quotas(counts: Counter[str], requested: int) -> dict[str, int]:
    total = sum(counts.values())
    if total == 0:
        return {}
    raw = {label: requested * count / total for label, count in counts.items()}
    quotas = {label: floor(value) for label, value in raw.items()}
    remaining = requested - sum(quotas.values())
    remainders = sorted(
        ((raw[label] - quotas[label], label) for label in counts),
        reverse=True,
    )
    for _remainder, label in remainders[:remaining]:
        quotas[label] += 1
    return quotas


def _label_stratified_subset(
    samples: list[Sample],
    requested: int,
    rng: random.Random,
) -> list[Sample]:
    label_counts = Counter(span.label for sample in samples for span in sample.gold_spans)
    quotas = _proportional_quotas(label_counts, requested)
    if not quotas:
        return _sample_indices(samples, rng.sample(range(len(samples)), requested))

    shuffled_indices = list(range(len(samples)))
    rng.shuffle(shuffled_indices)
    selected: set[int] = set()

    while len(selected) < requested and any(value > 0 for value in quotas.values()):
        best_index = None
        best_score = 0
        for index in shuffled_indices:
            if index in selected:
                continue
            labels = {span.label for span in samples[index].gold_spans}
            score = sum(quotas[label] for label in labels if quotas.get(label, 0) > 0)
            if score > best_score:
                best_index = index
                best_score = score
        if best_index is None:
            break
        selected.add(best_index)
        for label in {span.label for span in samples[best_index].gold_spans}:
            if quotas.get(label, 0) > 0:
                quotas[label] -= 1

    _fill_random(selected, len(samples), requested, rng)
    return _sample_indices(samples, list(selected))


def _even_quotas(keys: list[object], requested: int) -> dict[object, int]:
    buckets = sorted(set(keys), key=str)
    base = requested // len(buckets)
    remainder = requested % len(buckets)
    return {bucket: base + (1 if index < remainder else 0) for index, bucket in enumerate(buckets)}


def _bucketed_subset(
    samples: list[Sample],
    requested: int,
    rng: random.Random,
    key,
) -> list[Sample]:
    buckets: dict[object, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        buckets[key(sample)].append(index)

    quotas = _even_quotas(list(buckets), requested)
    selected: set[int] = set()
    for bucket, indices in buckets.items():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        selected.update(shuffled[: min(quotas[bucket], len(shuffled))])

    _fill_random(selected, len(samples), requested, rng)
    return _sample_indices(samples, list(selected))


def _length_quartile_keys(samples: list[Sample]):
    ordered = sorted(
        range(len(samples)),
        key=lambda index: (
            len(samples[index].tokens),
            samples[index].index,
            samples[index].sample_id,
        ),
    )
    by_identity = {
        id(samples[sample_index]): min(rank * 4 // len(samples), 3)
        for rank, sample_index in enumerate(ordered)
    }
    return lambda sample: by_identity[id(sample)]


def _hardness_subset(
    samples: list[Sample],
    requested: int,
    rng: random.Random,
) -> list[Sample]:
    scored = []
    for index, sample in enumerate(samples):
        labels = {span.label for span in sample.gold_spans}
        rare_hits = len(labels & RARE_LABELS)
        score = rare_hits * 1000 + len(sample.tokens)
        scored.append((score, rng.random(), index))
    scored.sort(reverse=True)
    return _sample_indices(samples, [index for _score, _tie, index in scored[:requested]])


def _fill_random(
    selected: set[int],
    sample_count: int,
    requested: int,
    rng: random.Random,
) -> None:
    if len(selected) >= requested:
        return
    remaining = [index for index in range(sample_count) if index not in selected]
    selected.update(rng.sample(remaining, requested - len(selected)))
