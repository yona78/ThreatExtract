"""Non-neural lexical baselines for the DNRTI benchmark.

A dictionary (gazetteer) baseline gives a sanity floor: a model that only memorises
train entity surfaces and tags any exact re-occurrence in the test split. Both
neural models should comfortably beat it; if they did not, the benchmark wiring
would be suspect. This is the "does this number even make sense?" check.

Pure Python, no model dependencies, fully deterministic.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from fork1.data import Sample, Span, reconstruct_text


def _surface_key(text: str) -> str:
    return " ".join(text.casefold().split())


def build_gazetteer(samples: list[Sample]) -> dict[str, str]:
    """Map each gold entity surface to its majority DNRTI label across `samples`."""
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        for span in sample.gold_spans:
            key = _surface_key(span.text)
            if key:
                label_counts[key][span.label] += 1
    return {surface: counts.most_common(1)[0][0] for surface, counts in label_counts.items()}


def predict_lexical(sample: Sample, gazetteer: dict[str, str]) -> list[Span]:
    """Greedy longest-match tagging of known surfaces in a sample."""
    tokens = list(sample.tokens)
    _text, offsets = reconstruct_text(tokens)
    max_len = max((len(surface.split()) for surface in gazetteer), default=1)
    spans: list[Span] = []
    i = 0
    while i < len(tokens):
        matched = False
        for span_len in range(min(max_len, len(tokens) - i), 0, -1):
            surface = _surface_key(" ".join(tokens[i : i + span_len]))
            label = gazetteer.get(surface)
            if label is not None:
                start = offsets[i][0]
                end = offsets[i + span_len - 1][1]
                spans.append(
                    Span(
                        label=label,
                        start=start,
                        end=end,
                        text=sample.text[start:end],
                        score=1.0,
                        source="lexical",
                    )
                )
                i += span_len
                matched = True
                break
        if not matched:
            i += 1
    return spans


def score_exact(
    samples: list[Sample], preds_by_id: dict[str, list[Span]]
) -> dict[str, float | int]:
    """Exact span+label P/R/F1 (no taxonomy mapping; labels are already DNRTI)."""
    tp = pred_total = gold_total = 0
    for sample in samples:
        gold = {(s.start, s.end, s.label) for s in sample.gold_spans}
        pred = {(s.start, s.end, s.label) for s in preds_by_id.get(sample.sample_id, [])}
        gold_total += len(gold)
        pred_total += len(pred)
        tp += len(gold & pred)
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "predicted": pred_total,
        "gold": gold_total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def lexical_baseline_row(
    train_samples: list[Sample], test_samples: list[Sample]
) -> dict[str, object]:
    """Train-gazetteer baseline evaluated on the test split."""
    gazetteer = build_gazetteer(train_samples)
    preds = {sample.sample_id: predict_lexical(sample, gazetteer) for sample in test_samples}
    result = score_exact(test_samples, preds)
    return {"baseline": "train_gazetteer", "gazetteer_surfaces": len(gazetteer), **result}
