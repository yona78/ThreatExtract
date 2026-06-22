from __future__ import annotations

from fork1.mapping import map_model_label_to_dnrti


def _overlap_chars(token_start: int, token_end: int, span) -> int:
    return max(0, min(token_end, span.end) - max(token_start, span.start))


def _matches_policy(token_start: int, token_end: int, span, policy: str) -> bool:
    overlap = _overlap_chars(token_start, token_end, span)
    if policy == "overlap":
        return overlap > 0
    if policy == "majority":
        return overlap > ((token_end - token_start) / 2)
    if policy == "contained":
        return span.start <= token_start and token_end <= span.end
    raise ValueError(f"unknown alignment policy: {policy}")


def align_pred_to_tokens(
    token_offsets: list[tuple[int, int]],
    pred_spans,
    model_name: str,
    policy: str,
) -> list[set[str]]:
    if policy not in {"overlap", "majority", "contained"}:
        raise ValueError(f"unknown alignment policy: {policy}")

    aligned = [set() for _ in token_offsets]
    for pred in pred_spans:
        labels = map_model_label_to_dnrti(model_name, pred.label)
        if not labels:
            continue
        for token_index, (token_start, token_end) in enumerate(token_offsets):
            if _matches_policy(token_start, token_end, pred, policy):
                aligned[token_index].update(labels)
    return aligned
