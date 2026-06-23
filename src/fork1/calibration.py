from __future__ import annotations

from fork1.mapping import map_model_label_to_dnrti


def entity_records(samples, predictions, model_name: str) -> list[dict[str, float | bool]]:
    records: list[dict[str, float | bool]] = []
    for sample in samples:
        used_gold: set[int] = set()
        gold_spans = sample.gold_spans
        for pred in sorted(
            predictions.get(sample.sample_id, []),
            key=lambda span: span.score or 0.0,
            reverse=True,
        ):
            matched_gold = None
            for gold_index, gold in enumerate(gold_spans):
                if gold_index in used_gold:
                    continue
                if (
                    gold.start == pred.start
                    and gold.end == pred.end
                    and gold.label in map_model_label_to_dnrti(model_name, pred.label)
                ):
                    matched_gold = gold_index
                    break
            is_correct = matched_gold is not None
            if matched_gold is not None:
                used_gold.add(matched_gold)
            records.append({"score": float(pred.score or 0.0), "is_correct": is_correct})
    return records


def expected_calibration_error(records: list[dict[str, float | bool]], bins: int = 10) -> float:
    if not records:
        return 0.0
    ece = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        if index == bins - 1:
            bucket = [record for record in records if low <= float(record["score"]) <= high]
        else:
            bucket = [record for record in records if low <= float(record["score"]) < high]
        if not bucket:
            continue
        confidence = sum(float(record["score"]) for record in bucket) / len(bucket)
        precision = sum(bool(record["is_correct"]) for record in bucket) / len(bucket)
        ece += (len(bucket) / len(records)) * abs(precision - confidence)
    return ece


def threshold_sweep(
    records: list[dict[str, float | bool]],
    thresholds: list[float],
    *,
    total_gold: int,
) -> list[dict[str, float | int]]:
    rows = []
    for threshold in thresholds:
        selected = [record for record in records if float(record["score"]) >= threshold]
        true_positive = sum(bool(record["is_correct"]) for record in selected)
        precision = true_positive / len(selected) if selected else 0.0
        recall = true_positive / total_gold if total_gold else 0.0
        rows.append(
            {
                "threshold": threshold,
                "predictions": len(selected),
                "true_positive": true_positive,
                "precision": precision,
                "recall": recall,
            }
        )
    return rows
