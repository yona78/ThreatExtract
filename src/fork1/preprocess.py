from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

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


@dataclass(frozen=True)
class ContextDoc:
    text: str
    token_offsets: tuple[tuple[int, int], ...]
    token_refs: tuple[tuple[str, int], ...]
    sample_ids: tuple[str, ...]
    sample_char_ranges: tuple[tuple[str, int, int], ...]

    def token_refs_for_span(self, start: int, end: int) -> list[tuple[str, int]]:
        return [
            ref
            for (token_start, token_end), ref in zip(
                self.token_offsets,
                self.token_refs,
                strict=False,
            )
            if max(start, token_start) < min(end, token_end)
        ]


def _context_from_samples(samples, detokenizer: Detokenizer = detok_single_space) -> ContextDoc:
    parts: list[str] = []
    token_offsets: list[tuple[int, int]] = []
    token_refs: list[tuple[str, int]] = []
    sample_char_ranges: list[tuple[str, int, int]] = []
    cursor = 0

    for sample in samples:
        if parts:
            parts.append(" ")
            cursor += 1

        sentence_text, sentence_offsets = detokenizer(list(sample.tokens))
        if not sentence_text:
            sentence_text = sample.text
        base = cursor
        parts.append(sentence_text)
        cursor += len(sentence_text)
        sample_char_ranges.append((sample.sample_id, base, cursor))

        for token_index, (start, end) in enumerate(sentence_offsets):
            token_offsets.append((base + start, base + end))
            token_refs.append((sample.sample_id, token_index))

    return ContextDoc(
        text="".join(parts),
        token_offsets=tuple(token_offsets),
        token_refs=tuple(token_refs),
        sample_ids=tuple(sample.sample_id for sample in samples),
        sample_char_ranges=tuple(sample_char_ranges),
    )


def iter_contexts(
    samples,
    mode: str,
    window: int = 3,
    stride: int = 2,
    detokenizer: Detokenizer = detok_single_space,
) -> list[ContextDoc]:
    if mode == "sentence":
        return [_context_from_samples([sample], detokenizer) for sample in samples]
    if mode == "document":
        return [_context_from_samples(samples, detokenizer)] if samples else []
    if mode == "window":
        if window < 1:
            raise ValueError("window must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        return [
            _context_from_samples(samples[start : start + window], detokenizer)
            for start in range(0, len(samples), stride)
            if samples[start : start + window]
        ]
    raise ValueError(f"unknown context mode: {mode}")


def refang(text: str) -> str:
    output = re.sub("hxxps", "https", text, flags=re.IGNORECASE)
    output = re.sub("hxxp", "http", output, flags=re.IGNORECASE)
    output = re.sub(r"\[\.\]|\[dot\]|\(dot\)", ".", output, flags=re.IGNORECASE)
    output = re.sub(r"\[at\]|\(at\)", "@", output, flags=re.IGNORECASE)
    return output


def defang(text: str) -> str:
    output = text.replace("https://", "hxxps://").replace("http://", "hxxp://")
    output = output.replace("@", "[at]")
    output = output.replace(".", "[.]")
    return output


def normalize_text(text: str, mode: str) -> str:
    if mode == "none":
        return text
    if mode == "nfc":
        return unicodedata.normalize("NFC", text)
    if mode == "nfkc":
        return unicodedata.normalize("NFKC", text)
    if mode == "refang":
        return refang(text)
    if mode == "lower":
        return text.lower()
    raise ValueError(f"unknown normalization mode: {mode}")
