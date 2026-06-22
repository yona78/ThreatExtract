from __future__ import annotations

import math
import random
from dataclasses import dataclass

from fork1.mapping import map_model_label_to_dnrti


@dataclass
class MucCounts:
    cor: int = 0
    inc: int = 0
    par: int = 0
    mis: int = 0
    spu: int = 0


def _overlap(left, right) -> bool:
    return max(left.start, right.start) < min(left.end, right.end)


def _exact(left, right) -> bool:
    return left.start == right.start and left.end == right.end


def _type_ok(gold, pred, model_name: str) -> bool:
    return gold.label in map_model_label_to_dnrti(model_name, pred.label)


def muc_counts(gold_spans, pred_spans, model_name: str) -> dict[str, MucCounts]:
    schemes = {key: MucCounts() for key in ("strict", "exact", "partial", "type")}
    ordered_preds = sorted(pred_spans, key=lambda span: span.score or 0.0, reverse=True)

    for scheme, counts in schemes.items():
        used_gold: set[int] = set()
        for pred in ordered_preds:
            best = None
            for gold_index, gold in enumerate(gold_spans):
                if gold_index in used_gold or not _overlap(gold, pred):
                    continue
                best = (gold_index, gold)
                break

            if best is None:
                counts.spu += 1
                continue

            gold_index, gold = best
            used_gold.add(gold_index)
            boundaries_match = _exact(gold, pred)
            type_matches = _type_ok(gold, pred, model_name)

            if scheme == "strict":
                ok = boundaries_match and type_matches
                counts.cor += int(ok)
                counts.inc += int(not ok)
            elif scheme == "exact":
                counts.cor += int(boundaries_match)
                counts.inc += int(not boundaries_match)
            elif scheme == "type":
                counts.cor += int(type_matches)
                counts.inc += int(not type_matches)
            else:
                if boundaries_match and type_matches:
                    counts.cor += 1
                else:
                    counts.par += 1

        counts.mis = len(gold_spans) - len(used_gold)

    return schemes


def prf(counts: MucCounts, scheme: str) -> dict[str, float | int]:
    possible = counts.cor + counts.inc + counts.par + counts.mis
    actual = counts.cor + counts.inc + counts.par + counts.spu
    numerator = counts.cor + (0.5 * counts.par if scheme == "partial" else 0.0)
    precision = numerator / actual if actual else 0.0
    recall = numerator / possible if possible else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"p": precision, "r": recall, "f1": f1, **counts.__dict__}


def score(gold_spans, pred_spans, model_name: str) -> dict[str, dict[str, float | int]]:
    counts_by_scheme = muc_counts(gold_spans, pred_spans, model_name)
    return {scheme: prf(counts, scheme) for scheme, counts in counts_by_scheme.items()}


def _sum_counts(items: list[MucCounts]) -> MucCounts:
    total = MucCounts()
    for item in items:
        total.cor += item.cor
        total.inc += item.inc
        total.par += item.par
        total.mis += item.mis
        total.spu += item.spu
    return total


def sample_muc_counts(samples, preds, model_name: str, scheme: str) -> list[MucCounts]:
    return [
        muc_counts(sample.gold_spans, preds.get(sample.sample_id, []), model_name)[scheme]
        for sample in samples
    ]


def corpus_f1(samples, preds, model_name: str, scheme: str) -> float:
    per_sample = sample_muc_counts(samples, preds, model_name, scheme)
    return float(prf(_sum_counts(per_sample), scheme)["f1"])


def _f1_from_sample_counts(
    counts: list[MucCounts], scheme: str, indices: list[int] | None = None
) -> float:
    if indices is None:
        selected = counts
    else:
        selected = [counts[index] for index in indices]
    return float(prf(_sum_counts(selected), scheme)["f1"])


def bootstrap_gap_ci(
    samples,
    preds_a,
    preds_b,
    model_a: str,
    model_b: str,
    scheme: str,
    n: int,
    seed: int,
) -> tuple[float, float, float]:
    counts_a = sample_muc_counts(samples, preds_a, model_a, scheme)
    counts_b = sample_muc_counts(samples, preds_b, model_b, scheme)
    gap = _f1_from_sample_counts(counts_a, scheme) - _f1_from_sample_counts(counts_b, scheme)
    if not samples or n <= 0:
        return gap, gap, gap

    rng = random.Random(seed)
    gaps: list[float] = []
    for _ in range(n):
        indices = [rng.randrange(len(samples)) for _ in samples]
        gaps.append(
            _f1_from_sample_counts(counts_a, scheme, indices)
            - _f1_from_sample_counts(counts_b, scheme, indices)
        )

    try:
        import numpy as np
    except ImportError:
        ordered = sorted(gaps)
        low_index = max(0, min(len(ordered) - 1, int(0.025 * (len(ordered) - 1))))
        high_index = max(0, min(len(ordered) - 1, int(0.975 * (len(ordered) - 1))))
        return ordered[low_index], ordered[high_index], gap

    low, high = np.percentile(gaps, [2.5, 97.5])
    return float(low), float(high), gap


def _strict_gold_correctness(gold_spans, pred_spans, model_name: str) -> list[bool]:
    correct = [False for _ in gold_spans]
    used_gold: set[int] = set()
    for pred in sorted(pred_spans, key=lambda span: span.score or 0.0, reverse=True):
        for gold_index, gold in enumerate(gold_spans):
            if gold_index in used_gold:
                continue
            if _exact(gold, pred) and _type_ok(gold, pred, model_name):
                correct[gold_index] = True
                used_gold.add(gold_index)
                break
    return correct


def mcnemar(samples, preds_a, preds_b, model_a: str, model_b: str) -> tuple[float, float]:
    a_only = 0
    b_only = 0
    for sample in samples:
        correct_a = _strict_gold_correctness(
            sample.gold_spans,
            preds_a.get(sample.sample_id, []),
            model_a,
        )
        correct_b = _strict_gold_correctness(
            sample.gold_spans,
            preds_b.get(sample.sample_id, []),
            model_b,
        )
        for is_a_correct, is_b_correct in zip(correct_a, correct_b):
            if is_a_correct and not is_b_correct:
                a_only += 1
            elif is_b_correct and not is_a_correct:
                b_only += 1

    discordant = a_only + b_only
    if discordant == 0:
        return 0.0, 1.0

    statistic = (abs(a_only - b_only) - 1) ** 2 / discordant
    p_value = math.exp(-statistic / 2)
    return statistic, p_value
