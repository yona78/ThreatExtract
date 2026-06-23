from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class Span:
    label: str
    start: int
    end: int
    text: str
    score: float | None
    source: str


@dataclass(frozen=True)
class Sample:
    sample_id: str
    split: str
    index: int
    text: str
    tokens: tuple[str, ...]
    tags: tuple[str, ...]
    gold_spans: list[Span]


@dataclass
class MatchCounts:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positive + self.false_positive
        return self.true_positive / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positive + self.false_negative
        return self.true_positive / denom if denom else 0.0

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return 2 * self.precision * self.recall / denom if denom else 0.0


@dataclass
class DatasetStats:
    split: str
    sentences: int
    tokens: int
    labeled_tokens: int
    spans: int
    label_counts: dict[str, int]
    malformed_lines: int


def reconstruct_text(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    text_parts: list[str] = []
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for index, token in enumerate(tokens):
        if index:
            text_parts.append(" ")
            cursor += 1
        start = cursor
        text_parts.append(token)
        cursor += len(token)
        offsets.append((start, cursor))
    return "".join(text_parts), offsets


def extract_bio_spans(
    tokens: list[str],
    tags: list[str],
    text: str,
    offsets: list[tuple[int, int]],
    source: str = "gold",
) -> list[Span]:
    spans: list[Span] = []
    current_start: int | None = None
    current_end: int | None = None
    current_label: str | None = None

    for index, tag in enumerate(list(tags) + ["O"]):
        if tag == "O":
            prefix = "O"
            label = None
        elif "-" in tag:
            prefix, label = tag.split("-", 1)
        else:
            prefix = "B"
            label = tag

        boundary = prefix == "B" or prefix == "O" or label != current_label
        if current_start is not None and boundary:
            assert current_label is not None
            assert current_end is not None
            spans.append(
                Span(
                    label=current_label,
                    start=current_start,
                    end=current_end,
                    text=text[current_start:current_end],
                    score=None,
                    source=source,
                )
            )
            current_start = None
            current_end = None
            current_label = None

        if prefix in {"B", "I"} and label is not None:
            if current_start is None:
                current_start = offsets[index][0]
                current_label = label
            current_end = offsets[index][1]

    return spans


def load_dnrti_split(path: Path, split_name: str | None = None) -> tuple[list[Sample], list[str]]:
    split = split_name or path.stem
    samples: list[Sample] = []
    warnings: list[str] = []
    tokens: list[str] = []
    tags: list[str] = []

    def flush() -> None:
        if not tokens:
            return
        text, offsets = reconstruct_text(tokens)
        gold_spans = extract_bio_spans(tokens, tags, text, offsets)
        index = len(samples)
        samples.append(
            Sample(
                sample_id=f"{split}-{index:05d}",
                split=split,
                index=index,
                text=text,
                tokens=tuple(tokens),
                tags=tuple(tags),
                gold_spans=gold_spans,
            )
        )
        tokens.clear()
        tags.clear()

    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        line = raw.strip()
        if not line:
            flush()
            continue
        parts = line.rsplit(maxsplit=1)
        if len(parts) != 2:
            if line == "O" or line.startswith(("B-", "I-")):
                warnings.append(f"{path.name}:{line_number}: skipped tag-only line {line!r}")
                continue
            warnings.append(f"{path.name}:{line_number}: malformed line {raw!r}")
            continue
        token, tag = parts
        tokens.append(token)
        tags.append(tag)
    flush()
    return samples, warnings


def load_dnrti_dataset(
    dnrti_dir: Path, split: str
) -> tuple[list[Sample], list[str], list[DatasetStats]]:
    requested = DEFAULT_SPLITS if split == "all" else (split,)
    all_samples: list[Sample] = []
    all_warnings: list[str] = []
    stats: list[DatasetStats] = []
    for split_name in requested:
        path = dnrti_dir / f"{split_name}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"missing DNRTI split file: {path}")
        samples, warnings = load_dnrti_split(path, split_name=split_name)
        label_counts: Counter[str] = Counter()
        labeled_tokens = 0
        token_count = 0
        for sample in samples:
            token_count += len(sample.tokens)
            labeled_tokens += sum(1 for tag in sample.tags if tag != "O")
            label_counts.update(span.label for span in sample.gold_spans)
        stats.append(
            DatasetStats(
                split=split_name,
                sentences=len(samples),
                tokens=token_count,
                labeled_tokens=labeled_tokens,
                spans=sum(len(sample.gold_spans) for sample in samples),
                label_counts=dict(sorted(label_counts.items())),
                malformed_lines=len(warnings),
            )
        )
        all_samples.extend(samples)
        all_warnings.extend(warnings)
    return all_samples, all_warnings, stats


def summarize_gold(samples: list[Sample]) -> dict[str, object]:
    labels = Counter()
    token_count = 0
    for sample in samples:
        token_count += len(sample.tokens)
        labels.update(span.label for span in sample.gold_spans)
    return {
        "samples": len(samples),
        "tokens": token_count,
        "gold_spans": sum(labels.values()),
        "label_counts": dict(sorted(labels.items())),
    }
