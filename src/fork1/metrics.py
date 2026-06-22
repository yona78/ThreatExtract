from __future__ import annotations

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
