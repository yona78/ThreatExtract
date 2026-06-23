from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass

from fork1.mapping import DNRTI_LABELS, map_model_label_to_dnrti


@dataclass
class MucCounts:
    cor: int = 0
    inc: int = 0
    par: int = 0
    mis: int = 0
    spu: int = 0


def _overlap(left, right) -> bool:
    return max(left.start, right.start) < min(left.end, right.end)


def _overlap_amount(left, right) -> int:
    return max(0, min(left.end, right.end) - max(left.start, right.start))


def _exact(left, right) -> bool:
    return left.start == right.start and left.end == right.end


def _type_ok(gold, pred, model_name: str) -> bool:
    return gold.label in map_model_label_to_dnrti(model_name, pred.label)


def _mapped_label(model_name: str, pred, gold_label: str | None = None) -> str:
    labels = sorted(map_model_label_to_dnrti(model_name, pred.label))
    if gold_label is not None and gold_label in labels:
        return gold_label
    return "|".join(labels) if labels else "O/UNMAPPED"


def muc_counts(gold_spans, pred_spans, model_name: str) -> dict[str, MucCounts]:
    schemes = {key: MucCounts() for key in ("strict", "exact", "partial", "type")}
    ordered_preds = sorted(pred_spans, key=lambda span: span.score or 0.0, reverse=True)

    for scheme, counts in schemes.items():
        used_gold: set[int] = set()
        for pred in ordered_preds:
            best = None
            best_overlap = 0
            for gold_index, gold in enumerate(gold_spans):
                if gold_index in used_gold:
                    continue
                amount = _overlap_amount(gold, pred)
                if amount > best_overlap:
                    best_overlap = amount
                    best = (gold_index, gold)

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


def corpus_scores(samples, preds, model_name: str) -> dict[str, dict[str, float | int]]:
    out = {}
    for scheme in ("strict", "exact", "partial", "type"):
        per_sample = sample_muc_counts(samples, preds, model_name, scheme)
        out[scheme] = prf(_sum_counts(per_sample), scheme)
    return out


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
    return bootstrap_count_gap_ci(counts_a, counts_b, scheme, n, seed)


def bootstrap_count_gap_ci(
    counts_a: list[MucCounts],
    counts_b: list[MucCounts],
    scheme: str,
    n: int,
    seed: int,
) -> tuple[float, float, float]:
    gap = _f1_from_sample_counts(counts_a, scheme) - _f1_from_sample_counts(counts_b, scheme)
    if not counts_a or n <= 0:
        return gap, gap, gap

    rng = random.Random(seed)
    gaps: list[float] = []
    for _ in range(n):
        indices = [rng.randrange(len(counts_a)) for _ in counts_a]
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
        for is_a_correct, is_b_correct in zip(correct_a, correct_b, strict=False):
            if is_a_correct and not is_b_correct:
                a_only += 1
            elif is_b_correct and not is_a_correct:
                b_only += 1

    discordant = a_only + b_only
    if discordant == 0:
        return 0.0, 1.0

    statistic = (abs(a_only - b_only) - 1) ** 2 / discordant
    p_value = math.erfc(math.sqrt(statistic / 2))
    return statistic, p_value


def _span_text(sample, span) -> str:
    if span is None:
        return ""
    if span.text:
        return span.text
    return sample.text[span.start : span.end]


def _surface_key(text: str) -> str:
    return " ".join(text.casefold().split())


def _matched_error_records_for_sample(
    sample, pred_spans, model_name: str
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    used_gold: set[int] = set()
    used_pred: set[int] = set()
    ordered_preds = sorted(
        enumerate(pred_spans),
        key=lambda item: (-(item[1].score or 0.0), item[1].start, item[1].end),
    )

    for pred_index, pred in ordered_preds:
        best = None
        best_overlap = 0
        for gold_index, gold in enumerate(sample.gold_spans):
            if gold_index in used_gold:
                continue
            amount = _overlap_amount(gold, pred)
            if amount > best_overlap:
                best_overlap = amount
                best = (gold_index, gold)

        if best is None:
            continue

        gold_index, gold = best
        used_gold.add(gold_index)
        used_pred.add(pred_index)
        boundaries_match = _exact(gold, pred)
        type_matches = _type_ok(gold, pred, model_name)
        if boundaries_match and type_matches:
            bucket = "strict_correct"
        elif not type_matches:
            bucket = "type_error"
        else:
            bucket = "boundary_error"

        records.append(
            {
                "model": model_name,
                "sample_id": sample.sample_id,
                "bucket": bucket,
                "gold_label": gold.label,
                "predicted_label": _mapped_label(model_name, pred, gold.label),
                "gold_text": _span_text(sample, gold),
                "predicted_text": _span_text(sample, pred),
                "gold_start": gold.start,
                "gold_end": gold.end,
                "pred_start": pred.start,
                "pred_end": pred.end,
                "score": pred.score,
            }
        )

    for gold_index, gold in enumerate(sample.gold_spans):
        if gold_index in used_gold:
            continue
        records.append(
            {
                "model": model_name,
                "sample_id": sample.sample_id,
                "bucket": "strict_drop_fn",
                "gold_label": gold.label,
                "predicted_label": "O/MISSED",
                "gold_text": _span_text(sample, gold),
                "predicted_text": "",
                "gold_start": gold.start,
                "gold_end": gold.end,
                "pred_start": None,
                "pred_end": None,
                "score": None,
            }
        )

    for pred_index, pred in enumerate(pred_spans):
        if pred_index in used_pred:
            continue
        records.append(
            {
                "model": model_name,
                "sample_id": sample.sample_id,
                "bucket": "spurious_fp",
                "gold_label": "SPURIOUS",
                "predicted_label": _mapped_label(model_name, pred),
                "gold_text": "",
                "predicted_text": _span_text(sample, pred),
                "gold_start": None,
                "gold_end": None,
                "pred_start": pred.start,
                "pred_end": pred.end,
                "score": pred.score,
            }
        )

    return records


def entity_error_records(samples, preds, model_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample in samples:
        rows.extend(
            _matched_error_records_for_sample(
                sample,
                preds.get(sample.sample_id, []),
                model_name,
            )
        )
    return rows


def entity_confusion_rows(samples, preds, model_name: str) -> list[dict[str, object]]:
    counts = Counter(
        (str(row["gold_label"]), str(row["predicted_label"]))
        for row in entity_error_records(samples, preds, model_name)
    )
    return [
        {
            "model": model_name,
            "gold_label": gold_label,
            "predicted_label": predicted_label,
            "count": count,
        }
        for (gold_label, predicted_label), count in sorted(counts.items())
    ]


def _predicted_labels_for_counts(predicted_label: str) -> list[str]:
    if predicted_label in {"O/MISSED", "O/UNMAPPED"}:
        return []
    return predicted_label.split("|")


def per_label_strict_rows(samples, preds, model_name: str) -> list[dict[str, object]]:
    true_positive: Counter[str] = Counter()
    false_positive: Counter[str] = Counter()
    false_negative: Counter[str] = Counter()

    for row in entity_error_records(samples, preds, model_name):
        bucket = row["bucket"]
        gold_label = str(row["gold_label"])
        predicted_labels = _predicted_labels_for_counts(str(row["predicted_label"]))
        # A single prediction must contribute at most one false positive. For a
        # one-to-many model->DNRTI projection (e.g. CyNER "Organization" ->
        # {HackOrg, Idus, Org, SecTeam}) the prediction is one decision, so it is
        # charged once to a representative (first sorted) mapped label instead of
        # once per mapped label, which previously over-counted CyNER FPs ~4x and
        # understated its per-label precision.
        fp_label = predicted_labels[0] if predicted_labels else None
        if bucket == "strict_correct":
            true_positive[gold_label] += 1
        elif bucket == "strict_drop_fn":
            false_negative[gold_label] += 1
        elif bucket in {"type_error", "boundary_error"}:
            false_negative[gold_label] += 1
            if fp_label is not None:
                false_positive[fp_label] += 1
        elif bucket == "spurious_fp":
            if fp_label is not None:
                false_positive[fp_label] += 1

    labels = sorted(DNRTI_LABELS | set(true_positive) | set(false_positive) | set(false_negative))
    out = []
    for label in labels:
        tp = true_positive[label]
        fp = false_positive[label]
        fn = false_negative[label]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out.append(
            {
                "model": model_name,
                "label": label,
                "support": tp + fn,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return out


def ambiguous_surfaces(samples) -> set[str]:
    """Gold entity surfaces that appear with more than one DNRTI label.

    These are genuinely ambiguous mentions (e.g. "ransomware" tagged as both
    SamFile and Tool, "EternalBlue" as both Exp and SamFile) where the correct
    label depends on context, so they are the hardest cases for type assignment.
    """
    labels_by_surface: dict[str, set[str]] = {}
    for sample in samples:
        for span in sample.gold_spans:
            key = _surface_key(_span_text(sample, span))
            if key:
                labels_by_surface.setdefault(key, set()).add(span.label)
    return {key for key, labels in labels_by_surface.items() if len(labels) > 1}


def ambiguity_rows(samples, preds, model_name: str) -> list[dict[str, object]]:
    """Strict P/R/F1 split by whether the gold surface is ambiguous (>1 gold label)."""
    ambiguous = ambiguous_surfaces(samples)
    counts = {
        "ambiguous": Counter({"tp": 0, "fp": 0, "fn": 0}),
        "unambiguous": Counter({"tp": 0, "fp": 0, "fn": 0}),
    }

    def status(text: str) -> str:
        return "ambiguous" if _surface_key(text) in ambiguous else "unambiguous"

    for row in entity_error_records(samples, preds, model_name):
        bucket = row["bucket"]
        if bucket == "strict_correct":
            counts[status(str(row["gold_text"]))]["tp"] += 1
        elif bucket == "strict_drop_fn":
            counts[status(str(row["gold_text"]))]["fn"] += 1
        elif bucket in {"type_error", "boundary_error"}:
            counts[status(str(row["gold_text"]))]["fn"] += 1
            counts[status(str(row["predicted_text"]))]["fp"] += 1
        elif bucket == "spurious_fp":
            counts[status(str(row["predicted_text"]))]["fp"] += 1

    out = []
    for surface_class in ("ambiguous", "unambiguous"):
        tp = counts[surface_class]["tp"]
        fp = counts[surface_class]["fp"]
        fn = counts[surface_class]["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out.append(
            {
                "model": model_name,
                "surface_class": surface_class,
                "support": tp + fn,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return out


def oov_entity_rows(train_samples, test_samples, preds, model_name: str) -> list[dict[str, object]]:
    train_surfaces = {
        _surface_key(_span_text(sample, span))
        for sample in train_samples
        for span in sample.gold_spans
        if _surface_key(_span_text(sample, span))
    }
    counts = {
        "seen": Counter({"tp": 0, "fp": 0, "fn": 0}),
        "unseen": Counter({"tp": 0, "fp": 0, "fn": 0}),
    }

    def status(text: str) -> str:
        return "seen" if _surface_key(text) in train_surfaces else "unseen"

    for row in entity_error_records(test_samples, preds, model_name):
        bucket = row["bucket"]
        if bucket == "strict_correct":
            counts[status(str(row["gold_text"]))]["tp"] += 1
        elif bucket == "strict_drop_fn":
            counts[status(str(row["gold_text"]))]["fn"] += 1
        elif bucket in {"type_error", "boundary_error"}:
            counts[status(str(row["gold_text"]))]["fn"] += 1
            counts[status(str(row["predicted_text"]))]["fp"] += 1
        elif bucket == "spurious_fp":
            counts[status(str(row["predicted_text"]))]["fp"] += 1

    out = []
    for surface_status in ("seen", "unseen"):
        tp = counts[surface_status]["tp"]
        fp = counts[surface_status]["fp"]
        fn = counts[surface_status]["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out.append(
            {
                "model": model_name,
                "surface_status": surface_status,
                "support": tp + fn,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return out
