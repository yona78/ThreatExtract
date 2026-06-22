from __future__ import annotations

from collections.abc import Callable


Detokenizer = Callable[[list[str]], tuple[str, list[tuple[int, int]]]]

NO_SPACE_BEFORE = set(".,;:!?)]}%") | {"'s", "n't", "'re", "'ll", "'ve", "'m", "'d"}
NO_SPACE_AFTER = set("([{")


def join_with_spacing(
    tokens: list[str],
    space_before: Callable[[str | None, str], bool],
) -> tuple[str, list[tuple[int, int]]]:
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    previous: str | None = None
    for index, token in enumerate(tokens):
        if index and space_before(previous, token):
            parts.append(" ")
            cursor += 1
        start = cursor
        parts.append(token)
        cursor += len(token)
        offsets.append((start, cursor))
        previous = token
    return "".join(parts), offsets


def detok_single_space(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    return join_with_spacing(tokens, lambda previous, token: True)


def detok_punct_aware(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    def space_before(previous: str | None, token: str) -> bool:
        if token in NO_SPACE_BEFORE:
            return False
        if previous in NO_SPACE_AFTER:
            return False
        return True

    return join_with_spacing(tokens, space_before)


DETOKENIZERS: dict[str, Detokenizer] = {
    "single_space": detok_single_space,
    "punct_aware": detok_punct_aware,
}
