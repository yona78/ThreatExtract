from __future__ import annotations

import argparse
import html
import json
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import mean, variance

from fork1.align import align_pred_to_tokens
from fork1.config import PRESETS, ExperimentConfig
from fork1.data import Sample, extract_bio_spans, load_dnrti_dataset
from fork1.intrinsic import (
    CYBER_PROBE_WORDS,
    domain_coverage,
    entity_surface_words,
    oracle_upper_bound,
    tokenizer_fertility,
)
from fork1.mapping import map_model_label_to_dnrti, strip_bio, unique_mapped_dnrti_labels
from fork1.metrics import (
    bootstrap_count_gap_ci,
    bootstrap_gap_ci,
    corpus_f1,
    mcnemar,
    sample_muc_counts,
)
from fork1.operational import make_workload, sample_power
from fork1.perturb import PERTURBATIONS, keyboard_typo, random_case
from fork1.preprocess import DETOKENIZERS, iter_contexts, normalize_text
from fork1.runner import (
    MODEL_ALIASES,
    HfTokenClassificationRunner,
    directory_size_bytes,
    hf_cache_model_dir,
)
from fork1.subsets import sample_subset


SWEEP_LEVELS = {
    "detok": ("single_space", "punct_aware"),
    "context": ("sentence", "document", "window"),
    "max_length": (128, 256, "full"),
    "normalization": ("none", "nfkc", "refang", "lower"),
    "alignment": ("overlap", "majority", "contained"),
}
SUBSET_STRATEGIES = ("random", "label_stratified", "density", "length", "hardness")
SUBSET_SIZES = ("10", "50", "100", "250", "all")
SUBSET_SEEDS = (1, 2, 3)


def iter_preprocessing_sweep_configs(base: ExperimentConfig = PRESETS["pdf_mapping"]):
    seen: set[ExperimentConfig] = set()
    for field, values in SWEEP_LEVELS.items():
        for value in values:
            kwargs = {field: value}
            config = replace(base, name=f"{field}={value}", **kwargs)
            if config in seen:
                continue
            seen.add(config)
            yield config


def iter_subset_study_configs(base: ExperimentConfig = PRESETS["pdf_mapping"]):
    for strategy in SUBSET_STRATEGIES:
        for size in SUBSET_SIZES:
            for seed in SUBSET_SEEDS:
                yield replace(
                    base,
                    name=f"subset={strategy},size={size},seed={seed}",
                    subset_strategy=strategy,
                    subset_size=size,
                    seed=seed,
                )


def iter_protocol_configs():
    yield PRESETS["pdf_mapping"]
    yield PRESETS["paper_native"]


def prepare_samples_for_config(samples: list[Sample], config: ExperimentConfig) -> list[Sample]:
    samples = sample_subset(samples, config.subset_strategy, config.subset_size, config.seed)
    detokenizer = DETOKENIZERS[config.detok]
    perturbation = _resolve_perturbation(config.perturbation, config.seed)
    prepared: list[Sample] = []
    for sample in samples:
        normalized_tokens = tuple(
            normalize_text(token, config.normalization) for token in sample.tokens
        )
        if perturbation is not None:
            normalized_tokens = tuple(perturbation(token) for token in normalized_tokens)
        text, offsets = detokenizer(list(normalized_tokens))
        gold_spans = extract_bio_spans(
            list(normalized_tokens),
            list(sample.tags),
            text,
            offsets,
        )
        prepared.append(
            Sample(
                sample_id=sample.sample_id,
                split=sample.split,
                index=sample.index,
                text=text,
                tokens=normalized_tokens,
                tags=sample.tags,
                gold_spans=gold_spans,
            )
        )
    return prepared


def _resolve_perturbation(name: str, seed: int):
    if name == "none":
        return None
    if name == "random_case":
        return random_case(seed=seed)
    if name == "keyboard_typo":
        return keyboard_typo(rate=0.05, seed=seed)
    if name in PERTURBATIONS:
        return PERTURBATIONS[name]
    raise ValueError(f"unknown perturbation: {name}")


def _token_offsets_for_config(sample: Sample, config: ExperimentConfig) -> list[tuple[int, int]]:
    _, offsets = DETOKENIZERS[config.detok](list(sample.tokens))
    return offsets


def _token_matches_policy(token_start: int, token_end: int, span, policy: str) -> bool:
    overlap = max(0, min(token_end, span.end) - max(token_start, span.start))
    if policy == "overlap":
        return overlap > 0
    if policy == "majority":
        return overlap > ((token_end - token_start) / 2)
    if policy == "contained":
        return span.start <= token_start and token_end <= span.end
    raise ValueError(f"unknown alignment policy: {policy}")


def apply_alignment_policy(
    sample: Sample,
    pred_spans,
    config: ExperimentConfig,
) -> list:
    if config.alignment == "overlap":
        return list(pred_spans)

    token_offsets = _token_offsets_for_config(sample, config)
    aligned = []
    for pred in pred_spans:
        selected = [
            (start, end)
            for start, end in token_offsets
            if _token_matches_policy(start, end, pred, config.alignment)
        ]
        if not selected:
            continue
        start = min(item[0] for item in selected)
        end = max(item[1] for item in selected)
        aligned.append(
            type(pred)(
                label=pred.label,
                start=start,
                end=end,
                text=sample.text[start:end],
                score=pred.score,
                source=pred.source,
            )
        )
    return aligned


def _project_context_predictions(
    samples: list[Sample],
    context,
    pred_spans,
    config: ExperimentConfig,
) -> dict[str, list]:
    by_id = {sample.sample_id: sample for sample in samples}
    projected = {sample.sample_id: [] for sample in samples}
    for pred in pred_spans:
        for sample_id, sample_start, sample_end in context.sample_char_ranges:
            overlap_start = max(pred.start, sample_start)
            overlap_end = min(pred.end, sample_end)
            if overlap_start >= overlap_end:
                continue
            start = overlap_start - sample_start
            end = overlap_end - sample_start
            sample = by_id[sample_id]
            projected[sample_id].append(
                type(pred)(
                    label=pred.label,
                    start=start,
                    end=end,
                    text=sample.text[start:end],
                    score=pred.score,
                    source=pred.source,
                )
            )
    return projected


def predict_samples(samples: list[Sample], runner, config: ExperimentConfig) -> dict[str, list]:
    predictions = {sample.sample_id: [] for sample in samples}
    max_length = None if config.max_length == "full" else int(config.max_length)
    for context in iter_contexts(
        samples,
        config.context,
        detokenizer=DETOKENIZERS[config.detok],
    ):
        pred_spans = runner.predict(context.text, max_length=max_length)
        projected = _project_context_predictions(samples, context, pred_spans, config)
        for sample_id, spans in projected.items():
            predictions[sample_id].extend(spans)
    by_id = {sample.sample_id: sample for sample in samples}
    if config.alignment != "overlap":
        predictions = {
            sample_id: apply_alignment_policy(by_id[sample_id], spans, config)
            for sample_id, spans in predictions.items()
        }
    return predictions


def run_config(
    config: ExperimentConfig,
    samples: list[Sample],
    *,
    device: str,
    offline: bool,
    cache_dir: Path | None,
) -> list[dict[str, object]]:
    return run_config_with_predictions(
        config,
        samples,
        device=device,
        offline=offline,
        cache_dir=cache_dir,
    )[2]


def run_config_with_predictions(
    config: ExperimentConfig,
    samples: list[Sample],
    *,
    device: str,
    offline: bool,
    cache_dir: Path | None,
) -> tuple[list[Sample], dict[str, dict[str, list]], list[dict[str, object]]]:
    prepared = prepare_samples_for_config(samples, config)
    predictions_by_model = {}
    for model_name in config.models:
        runner = HfTokenClassificationRunner(
            name=model_name,
            model_id=MODEL_ALIASES[model_name],
            device_request=device,
            allow_device_fallback=False,
            offline=offline,
            cache_dir=cache_dir,
        )
        runner.load()
        predictions_by_model[model_name] = predict_samples(prepared, runner, config)

    rows = _comparison_rows(config, prepared, predictions_by_model)
    for row in rows:
        row.update(
            {
                "subset_strategy": config.subset_strategy,
                "subset_size": config.subset_size,
                "subset_seed": config.seed,
                "samples": len(prepared),
            }
        )
    return prepared, predictions_by_model, rows


def _comparison_rows(
    config: ExperimentConfig,
    prepared: list[Sample],
    predictions_by_model: dict[str, dict[str, list]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if len(config.models) == 2:
        model_a, model_b = config.models
        lo, hi, gap = bootstrap_gap_ci(
            prepared,
            predictions_by_model[model_a],
            predictions_by_model[model_b],
            model_a,
            model_b,
            config.scheme,
            config.bootstrap,
            config.seed,
        )
        stat, p_value = mcnemar(
            prepared,
            predictions_by_model[model_a],
            predictions_by_model[model_b],
            model_a,
            model_b,
        )
        gap_by_model = {model_a: gap, model_b: -gap}
        ci_by_model = {model_a: (lo, hi), model_b: (-hi, -lo)}
        for model_name in config.models:
            ci_low, ci_high = ci_by_model[model_name]
            rows.append(
                {
                    "config": config.name,
                    "model": model_name,
                    "scheme": config.scheme,
                    "strict_f1": corpus_f1(
                        prepared,
                        predictions_by_model[model_name],
                        model_name,
                        config.scheme,
                    ),
                    "gap_vs_other": gap_by_model[model_name],
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "mcnemar_stat": stat,
                    "mcnemar_p": p_value,
                    "flip": (ci_low <= 0 <= ci_high),
                }
            )
    return rows


def subset_rows_from_predictions(
    config: ExperimentConfig,
    prepared: list[Sample],
    predictions_by_model: dict[str, dict[str, list]],
) -> list[dict[str, object]]:
    subset = sample_subset(prepared, config.subset_strategy, config.subset_size, config.seed)
    rows = _comparison_rows(config, subset, predictions_by_model)
    for row in rows:
        row.update(
            {
                "subset_strategy": config.subset_strategy,
                "subset_size": config.subset_size,
                "subset_seed": config.seed,
                "samples": len(subset),
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_preprocessing_report(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Preprocessing Sensitivity",
        "",
        "![Preprocessing tornado](figures/preprocessing_tornado.svg)",
        "",
        "| Config | Model | Strict F1 | Gap | 95% CI | Flip? |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['config']} | {row['model']} | {float(row['strict_f1']):.4f} | "
            f"{float(row['gap_vs_other']):.4f} | "
            f"[{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}] | "
            f"{'yes' if row['flip'] else 'no'} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_preprocessing_tornado(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    swings: list[tuple[str, str, float]] = []
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        config = str(row["config"])
        lever = config.split("=", 1)[0]
        model = str(row["model"])
        grouped.setdefault((lever, model), []).append(float(row["strict_f1"]))
    for (lever, model), values in grouped.items():
        if not values:
            continue
        swings.append((lever, model, max(values) - min(values)))
    swings.sort(key=lambda item: item[2], reverse=True)

    width = 760
    row_height = 28
    left = 210
    top = 34
    chart_width = 480
    height = max(120, top + row_height * len(swings) + 24)
    max_swing = max((swing for _, _, swing in swings), default=1.0) or 1.0
    palette = {"securebert": "#2f6fbb", "cyner": "#c45746"}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        'viewBox="0 0 {0} {1}">'.format(width, height),
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="24" font-family="Arial" font-size="16" font-weight="700">'
        "Preprocessing F1 Swing</text>",
    ]
    for index, (lever, model, swing) in enumerate(swings):
        y = top + index * row_height
        bar_width = int((swing / max_swing) * chart_width)
        label = html.escape(f"{lever} / {model}")
        color = palette.get(model, "#777777")
        parts.append(f'<text x="20" y="{y + 17}" font-family="Arial" font-size="12">{label}</text>')
        parts.append(
            f'<rect x="{left}" y="{y + 5}" width="{bar_width}" height="16" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{left + bar_width + 8}" y="{y + 17}" font-family="Arial" '
            f'font-size="12">{swing:.4f}</text>'
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def summarize_subset_cells(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["subset_strategy"]),
                str(row["subset_size"]),
                str(row["model"]),
            )
        ].append(float(row["strict_f1"]))

    out = []
    for strategy, size, model in sorted(
        grouped,
        key=lambda item: (
            SUBSET_STRATEGIES.index(item[0]) if item[0] in SUBSET_STRATEGIES else 999,
            _subset_size_sort_key(item[1]),
            item[2],
        ),
    ):
        values = grouped[(strategy, size, model)]
        out.append(
            {
                "subset_strategy": strategy,
                "subset_size": size,
                "model": model,
                "seeds": len(values),
                "mean_f1": mean(values),
                "variance_f1": variance(values) if len(values) > 1 else 0.0,
                "min_f1": min(values),
                "max_f1": max(values),
            }
        )
    return out


def min_faithful_subset_by_strategy(
    rows: list[dict[str, object]],
    *,
    full_winner: str,
) -> dict[str, str]:
    strategies = [
        strategy
        for strategy in SUBSET_STRATEGIES
        if any(row["subset_strategy"] == strategy for row in rows)
    ]
    if full_winner == "tie":
        return {strategy: "none" for strategy in strategies}

    out: dict[str, str] = {}
    for strategy in strategies:
        out[strategy] = "none"
        for size in SUBSET_SIZES:
            winner_rows = [
                row
                for row in rows
                if row["subset_strategy"] == strategy
                and row["subset_size"] == size
                and row["model"] == full_winner
            ]
            if {int(row["subset_seed"]) for row in winner_rows} != set(SUBSET_SEEDS):
                continue
            if all(float(row["ci_low"]) > 0 for row in winner_rows):
                out[strategy] = size
                break
    return out


def full_split_winner(rows: list[dict[str, object]]) -> str:
    for row in rows:
        if row["subset_size"] != "all":
            continue
        if float(row["ci_low"]) > 0:
            return str(row["model"])
    return "tie"


def write_subset_study_report(
    path: Path,
    rows: list[dict[str, object]],
    *,
    min_faithful: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_subset_cells(rows)
    lines = [
        "# Subset Study",
        "",
        "The min-faithful subset is the smallest size where the full-split winner wins "
        "with a 95% CI excluding 0 for all three seeds.",
        "",
        "## Min-Faithful Subset",
        "",
        "| Strategy | Min-faithful subset |",
        "|---|---|",
    ]
    for strategy in SUBSET_STRATEGIES:
        lines.append(f"| {strategy} | {min_faithful.get(strategy, 'none')} |")

    lines.extend(
        [
            "",
            "## F1 Variance By Cell",
            "",
            "| Strategy | Size | Model | Seeds | Mean F1 | Variance | Min F1 | Max F1 |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['subset_strategy']} | {row['subset_size']} | {row['model']} | "
            f"{int(row['seeds'])} | {float(row['mean_f1']):.4f} | "
            f"{float(row['variance_f1']):.6f} | {float(row['min_f1']):.4f} | "
            f"{float(row['max_f1']):.4f} |"
        )

    lines.extend(
        [
            "",
            "## Per-Seed Decisions",
            "",
            "| Strategy | Size | Seed | Model | Samples | Strict F1 | Gap | 95% CI | Flip? |",
            "|---|---|---:|---|---:|---:|---:|---|---|",
        ]
    )
    for row in _sort_subset_rows(rows):
        lines.append(
            f"| {row['subset_strategy']} | {row['subset_size']} | {int(row['subset_seed'])} | "
            f"{row['model']} | {int(row['samples'])} | {float(row['strict_f1']):.4f} | "
            f"{float(row['gap_vs_other']):.4f} | "
            f"[{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}] | "
            f"{'yes' if row['flip'] else 'no'} |"
        )

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_subset_curve(path: Path, rows: list[dict[str, object]], strategy: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = [row for row in summarize_subset_cells(rows) if row["subset_strategy"] == strategy]
    width = 760
    height = 300
    left = 70
    right = 30
    top = 30
    bottom = 50
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_f1 = max((float(row["max_f1"]) for row in summary), default=1.0) or 1.0
    x_positions = {
        size: left + (plot_width * index / (len(SUBSET_SIZES) - 1))
        for index, size in enumerate(SUBSET_SIZES)
    }
    palette = {"securebert": "#2f6fbb", "cyner": "#c45746"}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="22" font-family="Arial" font-size="16" font-weight="700">'
        f"{html.escape(strategy)} subset F1 vs size</text>",
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" '
        f'y2="{height - bottom}" stroke="#444"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#444"/>',
    ]
    for size, x in x_positions.items():
        parts.append(
            f'<text x="{x - 12}" y="{height - 24}" font-family="Arial" '
            f'font-size="11">{html.escape(size)}</text>'
        )
    for model in ("securebert", "cyner"):
        points = []
        for row in summary:
            if row["model"] != model:
                continue
            x = x_positions[str(row["subset_size"])]
            y = height - bottom - (float(row["mean_f1"]) / max_f1 * plot_height)
            points.append((x, y))
        if not points:
            continue
        point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(
            f'<polyline fill="none" stroke="{palette[model]}" stroke-width="2" '
            f'points="{point_text}"/>'
        )
        for x, y in points:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{palette[model]}"/>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def _sort_subset_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            (
                SUBSET_STRATEGIES.index(str(row["subset_strategy"]))
                if str(row["subset_strategy"]) in SUBSET_STRATEGIES
                else 999
            ),
            _subset_size_sort_key(str(row["subset_size"])),
            int(row["subset_seed"]),
            str(row["model"]),
        ),
    )


def _subset_size_sort_key(size: str) -> int:
    return 10**9 if size == "all" else int(size)


def bio_tags_from_label_sets(label_sets: list[set[str]]) -> list[str]:
    tags = []
    previous_label = None
    for labels in label_sets:
        if len(labels) != 1:
            tags.append("O")
            previous_label = None
            continue
        label = next(iter(labels))
        prefix = "I" if label == previous_label else "B"
        tags.append(f"{prefix}-{label}")
        previous_label = label
    return tags


def _gold_bio_for_unique_labels(sample: Sample, unique_labels: set[str]) -> list[str]:
    label_sets = []
    for tag in sample.tags:
        label = strip_bio(tag)
        label_sets.append({label} if tag != "O" and label in unique_labels else set())
    return bio_tags_from_label_sets(label_sets)


def _pred_bio_for_unique_labels(
    sample: Sample,
    pred_spans,
    model_name: str,
    config: ExperimentConfig,
    unique_labels: set[str],
) -> list[str]:
    _text, offsets = DETOKENIZERS[config.detok](list(sample.tokens))
    aligned = align_pred_to_tokens(offsets, pred_spans, model_name, config.alignment)
    return bio_tags_from_label_sets([labels & unique_labels for labels in aligned])


def _filter_predictions_to_unique_labels(predictions: dict[str, list], model_name: str):
    return {
        sample_id: [
            span for span in spans if len(map_model_label_to_dnrti(model_name, span.label)) == 1
        ]
        for sample_id, spans in predictions.items()
    }


def _bio_spans(tags: list[str]) -> set[tuple[int, int, str]]:
    spans: set[tuple[int, int, str]] = set()
    start: int | None = None
    current_label: str | None = None
    for index, tag in enumerate([*tags, "O"]):
        if tag == "O":
            label = None
            starts_new = False
        else:
            prefix, label = tag.split("-", 1)
            starts_new = prefix == "B" or label != current_label

        if current_label is not None and (label is None or starts_new):
            assert start is not None
            spans.add((start, index, current_label))
            start = None
            current_label = None

        if label is not None and start is None:
            start = index
            current_label = label
    return spans


def _strict_f1_from_bio(gold_bio: list[list[str]], pred_bio: list[list[str]]) -> float:
    gold = {
        (sentence_index, *span)
        for sentence_index, tags in enumerate(gold_bio)
        for span in _bio_spans(tags)
    }
    pred = {
        (sentence_index, *span)
        for sentence_index, tags in enumerate(pred_bio)
        for span in _bio_spans(tags)
    }
    true_positive = len(gold & pred)
    false_positive = len(pred - gold)
    false_negative = len(gold - pred)
    precision = true_positive / (true_positive + false_positive) if pred else 0.0
    recall = true_positive / (true_positive + false_negative) if gold else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def mapping_coverage_for_model(samples: list[Sample], model_name: str) -> dict[str, object]:
    unique_labels = unique_mapped_dnrti_labels(model_name)
    counts = Counter(span.label for sample in samples for span in sample.gold_spans)
    unique_count = sum(count for label, count in counts.items() if label in unique_labels)
    total = sum(counts.values())
    non_unique_labels = {
        label: count for label, count in sorted(counts.items()) if label not in unique_labels
    }
    return {
        "model": model_name,
        "total_gold_spans": total,
        "unique_gold_spans": unique_count,
        "non_unique_gold_spans": total - unique_count,
        "coverage": unique_count / total if total else 0.0,
        "unique_labels": sorted(unique_labels),
        "non_unique_labels": non_unique_labels,
    }


def seqeval_cross_check(
    samples: list[Sample],
    predictions: dict[str, list],
    model_name: str,
    config: ExperimentConfig,
) -> dict[str, object]:
    from seqeval.metrics import f1_score as seqeval_f1_score
    from seqeval.scheme import IOB2

    unique_labels = unique_mapped_dnrti_labels(model_name)
    filtered_predictions = _filter_predictions_to_unique_labels(predictions, model_name)
    gold_bio = [_gold_bio_for_unique_labels(sample, unique_labels) for sample in samples]
    pred_bio = [
        _pred_bio_for_unique_labels(
            sample,
            filtered_predictions.get(sample.sample_id, []),
            model_name,
            config,
            unique_labels,
        )
        for sample in samples
    ]
    our_f1 = _strict_f1_from_bio(gold_bio, pred_bio)
    seqeval_f1 = float(seqeval_f1_score(gold_bio, pred_bio, mode="strict", scheme=IOB2))
    unique_gold_spans = sum(
        1 for sample in samples for span in sample.gold_spans if span.label in unique_labels
    )
    return {
        "protocol": config.name,
        "model": model_name,
        "our_f1": our_f1,
        "seqeval_f1": seqeval_f1,
        "delta": abs(our_f1 - seqeval_f1),
        "unique_gold_spans": unique_gold_spans,
        "unique_labels": sorted(unique_labels),
    }


def write_protocol_comparison_report(
    path: Path,
    rows: list[dict[str, object]],
    checks: list[dict[str, object]],
    coverage: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Protocol Comparison",
        "",
        "| Protocol | Model | Strict F1 | Gap | 95% CI | Flip? |",
        "|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['protocol']} | {row['model']} | {float(row['strict_f1']):.4f} | "
            f"{float(row['gap_vs_other']):.4f} | "
            f"[{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}] | "
            f"{'yes' if row['flip'] else 'no'} |"
        )

    securebert_gaps = {
        str(row["protocol"]): float(row["gap_vs_other"])
        for row in rows
        if row["model"] == "securebert"
    }
    if {"pdf_mapping", "paper_native"} <= set(securebert_gaps):
        movement = securebert_gaps["paper_native"] - securebert_gaps["pdf_mapping"]
        lines.extend(
            [
                "",
                "## Gap Movement",
                "",
                f"SecureBERT gap movement from PDF-mapping to paper-native: {movement:.4f}.",
            ]
        )

    lines.extend(
        [
            "",
            "## Seqeval Cross-Check",
            "",
            "| Protocol | Model | Our unique-label F1 | Seqeval F1 | Delta | Unique gold spans |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for check in checks:
        lines.append(
            f"| {check['protocol']} | {check['model']} | {float(check['our_f1']):.4f} | "
            f"{float(check['seqeval_f1']):.4f} | {float(check['delta']):.4f} | "
            f"{int(check['unique_gold_spans'])} |"
        )

    lines.extend(
        [
            "",
            "## Unique-Label Coverage",
            "",
            "| Model | Total gold spans | Unique-label spans | Non-unique spans | Coverage | Non-unique labels |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for item in coverage:
        non_unique = ", ".join(
            f"{label}:{count}" for label, count in item["non_unique_labels"].items()
        )
        lines.append(
            f"| {item['model']} | {int(item['total_gold_spans'])} | "
            f"{int(item['unique_gold_spans'])} | {int(item['non_unique_gold_spans'])} | "
            f"{float(item['coverage']):.4f} | {non_unique} |"
        )

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def load_hf_tokenizer(model_id: str, cache_dir: Path | None, offline: bool):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_files_only=offline,
    )


def model_parameter_count(model_id: str, cache_dir: Path | None, offline: bool) -> int:
    from transformers import AutoModelForTokenClassification

    model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        cache_dir=str(cache_dir) if cache_dir else None,
        local_files_only=offline,
    )
    return sum(parameter.numel() for parameter in model.parameters())


def write_intrinsic_report(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Intrinsic Metrics",
        "",
        "| Model | Entity fertility | Probe fertility | Probe 1-token coverage | "
        "Parameters | Cache MB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {float(row['entity_fertility']):.4f} | "
            f"{float(row['probe_fertility']):.4f} | "
            f"{float(row['probe_single_token_coverage']):.4f} | "
            f"{int(row['parameter_count'])} | {float(row['cache_size_mb']):.1f} |"
        )

    lines.extend(
        [
            "",
            "## Oracle upper bound",
            "",
            "| Model | Oracle precision | Oracle recall | Oracle F1 | Expressible labels |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        labels = ", ".join(row["expressible_labels"])
        lines.append(
            f"| {row['model']} | {float(row['oracle_precision']):.4f} | "
            f"{float(row['oracle_recall']):.4f} | {float(row['oracle_f1']):.4f} | "
            f"{labels} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def operational_devices(requested: str) -> list[dict[str, str]]:
    devices = []
    if requested in {"cpu", "both"}:
        devices.append({"device": "cpu", "status": "available"})
    if requested in {"mps", "both"}:
        try:
            import torch

            available = torch.backends.mps.is_available()
        except ImportError:
            available = False
        devices.append(
            {
                "device": "mps",
                "status": "available" if available else "unavailable",
            }
        )
    return devices


def benchmark_runner_batch(
    runner,
    workload: list[str],
    batch_size: int,
    warmup: int,
    repeats: int,
) -> dict[str, float | int]:
    for _ in range(warmup):
        _run_batch(runner, workload, batch_size)

    latencies = []
    total_tokens = 0
    for _ in range(repeats):
        start = time.perf_counter()
        _run_batch(runner, workload, batch_size)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        total_tokens += sum(len(text.split()) for text in workload)
    latency_ms = [value * 1000 for value in latencies]
    total_seconds = sum(latencies)
    sentences = len(workload) * repeats
    return {
        "sentences": sentences,
        "tokens": total_tokens,
        "total_seconds": total_seconds,
        "mean_ms": sum(latency_ms) / len(latency_ms) if latency_ms else 0.0,
        "p50_ms": _percentile(latency_ms, 50),
        "p95_ms": _percentile(latency_ms, 95),
        "p99_ms": _percentile(latency_ms, 99),
        "sent_per_s": sentences / total_seconds if total_seconds else 0.0,
        "tok_per_s": total_tokens / total_seconds if total_seconds else 0.0,
    }


def _run_batch(runner, workload: list[str], batch_size: int) -> None:
    if getattr(runner, "pipe", None) is not None:
        list(runner.pipe(workload, batch_size=batch_size))
        return
    for text in workload:
        runner.predict(text)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((percentile / 100) * (len(ordered) - 1))
    return ordered[index]


def write_operational_report(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Operational Envelope",
        "",
        "CPU is deployment-relevant because the offline Docker image ships CPU-only torch; "
        "MPS is the local-dev ceiling.",
        "",
        "![Latency](figures/operational_latency.svg)",
        "",
        "![Throughput](figures/operational_throughput.svg)",
        "",
        "| Device | Model | Tokens | Batch | p50 ms | p95 ms | p99 ms | sent/s | tok/s | RSS MB | Load s | Energy J | Energy source | Cache MB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        if row.get("status") != "available":
            lines.append(
                f"| {row['device']} | {row['model']} | - | - | - | - | - | - | - | - | - | - | "
                f"{row.get('status', 'unavailable')} | - |"
            )
            continue
        lines.append(
            f"| {row['device']} | {row['model']} | {int(row['token_length'])} | "
            f"{int(row['batch_size'])} | {float(row['p50_ms']):.2f} | "
            f"{float(row['p95_ms']):.2f} | {float(row['p99_ms']):.2f} | "
            f"{float(row['sent_per_s']):.2f} | {float(row['tok_per_s']):.2f} | "
            f"{float(row['rss_after_load_mb']):.1f} | {float(row['load_seconds']):.2f} | "
            f"{float(row['energy_j']):.2f} | {row['energy_source']} | "
            f"{float(row['cache_size_mb']):.1f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_operational_plot(
    path: Path,
    rows: list[dict[str, object]],
    metric: str,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    available = [row for row in rows if row.get("status") == "available"]
    width = 820
    height = 360
    left = 80
    bottom = 54
    top = 34
    plot_width = width - left - 40
    plot_height = height - top - bottom
    max_value = max((float(row[metric]) for row in available), default=1.0) or 1.0
    token_lengths = sorted({int(row["token_length"]) for row in available})
    if not token_lengths:
        token_lengths = [1]
    x_positions = {
        token_length: left + index * (plot_width / max(1, len(token_lengths) - 1))
        for index, token_length in enumerate(token_lengths)
    }
    palette = {
        ("cpu", "securebert"): "#2f6fbb",
        ("cpu", "cyner"): "#c45746",
        ("mps", "securebert"): "#6aa6e8",
        ("mps", "cyner"): "#e08a7c",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="24" font-family="Arial" font-size="16" font-weight="700">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - 40}" y2="{height - bottom}" stroke="#444"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#444"/>',
    ]
    for token_length, x in x_positions.items():
        parts.append(
            f'<text x="{x - 10}" y="{height - 28}" font-family="Arial" font-size="11">{token_length}</text>'
        )
    for device in ("cpu", "mps"):
        for model in ("securebert", "cyner"):
            model_rows = [
                row
                for row in available
                if row["device"] == device and row["model"] == model and int(row["batch_size"]) == 1
            ]
            if not model_rows:
                continue
            points = []
            for row in sorted(model_rows, key=lambda item: int(item["token_length"])):
                x = x_positions[int(row["token_length"])]
                y = height - bottom - (float(row[metric]) / max_value * plot_height)
                points.append((x, y))
            point_text = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            color = palette[(device, model)]
            parts.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{point_text}"/>'
            )
            label_x, label_y = points[-1]
            parts.append(
                f'<text x="{label_x + 6:.1f}" y="{label_y:.1f}" font-family="Arial" '
                f'font-size="11" fill="{color}">{device}/{model}</text>'
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def run_intrinsic_eval(
    *,
    dnrti_dir: Path,
    out_dir: Path,
    offline: bool,
    cache_dir: Path | None,
) -> list[dict[str, object]]:
    samples, _warnings, _stats = load_dnrti_dataset(dnrti_dir, "test")
    entity_words = entity_surface_words(samples)
    rows: list[dict[str, object]] = []
    for model_name in PRESETS["pdf_mapping"].models:
        model_id = MODEL_ALIASES[model_name]
        tokenizer = load_hf_tokenizer(model_id, cache_dir, offline)
        oracle = oracle_upper_bound(samples, model_name)
        cache_size = directory_size_bytes(hf_cache_model_dir(cache_dir, model_id))
        rows.append(
            {
                "model": model_name,
                "model_id": model_id,
                "entity_words": len(entity_words),
                "probe_words": len(CYBER_PROBE_WORDS),
                "entity_fertility": tokenizer_fertility(tokenizer, entity_words),
                "entity_single_token_coverage": domain_coverage(tokenizer, entity_words),
                "probe_fertility": tokenizer_fertility(tokenizer, CYBER_PROBE_WORDS),
                "probe_single_token_coverage": domain_coverage(tokenizer, CYBER_PROBE_WORDS),
                "parameter_count": model_parameter_count(model_id, cache_dir, offline),
                "cache_size_mb": (cache_size or 0) / (1024 * 1024),
                "oracle_precision": oracle["precision"],
                "oracle_recall": oracle["recall"],
                "oracle_f1": oracle["f1"],
                "oracle_true_positive": oracle["true_positive"],
                "oracle_false_negative": oracle["false_negative"],
                "expressible_labels": oracle["expressible_labels"],
            }
        )
    write_jsonl(out_dir / "intrinsic_metrics.jsonl", rows)
    write_intrinsic_report(out_dir / "intrinsic_metrics.md", rows)
    return rows


def run_operational_eval(
    *,
    dnrti_dir: Path,
    out_dir: Path,
    offline: bool,
    cache_dir: Path | None,
    requested_device: str,
    token_lengths: tuple[int, ...] = (16, 32, 64, 128, 256),
    batch_sizes: tuple[int, ...] = (1, 8, 32),
    warmup: int = 1,
    repeats: int = 1,
    assumed_watts: float = 25.0,
) -> list[dict[str, object]]:
    samples, _warnings, _stats = load_dnrti_dataset(dnrti_dir, "test")
    token_pool = [token for sample in samples for token in sample.tokens] or None
    rows: list[dict[str, object]] = []
    for device_info in operational_devices(requested_device):
        device = device_info["device"]
        for model_name in PRESETS["pdf_mapping"].models:
            if device_info["status"] != "available":
                rows.append(
                    {"device": device, "model": model_name, "status": device_info["status"]}
                )
                continue
            runner = HfTokenClassificationRunner(
                name=model_name,
                model_id=MODEL_ALIASES[model_name],
                device_request=device,
                allow_device_fallback=False,
                offline=offline,
                cache_dir=cache_dir,
            )
            try:
                load_info = runner.load()
            except RuntimeError as exc:
                rows.append(
                    {
                        "device": device,
                        "model": model_name,
                        "status": f"error: {exc}",
                    }
                )
                continue
            cache_size = directory_size_bytes(
                hf_cache_model_dir(cache_dir, MODEL_ALIASES[model_name])
            )
            for token_length in token_lengths:
                for batch_size in batch_sizes:
                    workload = make_workload(
                        [token_length],
                        batch_size,
                        seed=20260621 + token_length + batch_size,
                        token_pool=token_pool,
                    )
                    metrics = benchmark_runner_batch(
                        runner,
                        workload,
                        batch_size=batch_size,
                        warmup=warmup,
                        repeats=repeats,
                    )
                    watts = sample_power(float(metrics["total_seconds"]))
                    energy_source = "powermetrics" if watts is not None else "estimated"
                    watts = watts if watts is not None else assumed_watts
                    rows.append(
                        {
                            "device": device,
                            "model": model_name,
                            "status": "available",
                            "token_length": token_length,
                            "batch_size": batch_size,
                            **metrics,
                            "load_seconds": load_info.get("load_seconds") or 0.0,
                            "rss_after_load_mb": load_info.get("rss_after_load_mb") or 0.0,
                            "energy_j": float(watts) * float(metrics["total_seconds"]),
                            "energy_source": energy_source,
                            "cache_size_mb": (cache_size or 0) / (1024 * 1024),
                        }
                    )
    write_jsonl(out_dir / "operational_envelope.jsonl", rows)
    write_operational_plot(
        out_dir / "figures" / "operational_latency.svg",
        rows,
        metric="p50_ms",
        title="Operational p50 latency by token length",
    )
    write_operational_plot(
        out_dir / "figures" / "operational_throughput.svg",
        rows,
        metric="tok_per_s",
        title="Operational throughput by token length",
    )
    write_operational_report(out_dir / "operational_envelope.md", rows)
    return rows


def write_robustness_report(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Robustness Perturbations",
        "",
        "| Perturbation | Model | Clean F1 | Noisy F1 | Delta F1 | Delta 95% CI | "
        "Gap vs Other | Gap 95% CI | Flip? |",
        "|---|---|---:|---:|---:|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['perturbation']} | {row['model']} | {float(row['clean_f1']):.4f} | "
            f"{float(row['noisy_f1']):.4f} | {float(row['delta_f1']):.4f} | "
            f"[{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}] | "
            f"{float(row['gap_vs_other']):.4f} | "
            f"[{float(row['gap_ci_low']):.4f}, {float(row['gap_ci_high']):.4f}] | "
            f"{'yes' if row['flip'] else 'no'} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def iter_robustness_configs(base: ExperimentConfig = PRESETS["pdf_mapping"]):
    yield replace(base, name="clean", perturbation="none")
    for perturbation in ("defang", "refang", "random_case", "keyboard_typo"):
        yield replace(
            base,
            name=f"perturbation={perturbation}",
            perturbation=perturbation,
        )


def robustness_rows_from_config_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    clean = {str(row["model"]): row for row in rows if row["config"] == "clean"}
    out: list[dict[str, object]] = []
    for row in rows:
        config = str(row["config"])
        if not config.startswith("perturbation="):
            continue
        model = str(row["model"])
        clean_row = clean[model]
        delta = float(row["strict_f1"]) - float(clean_row["strict_f1"])
        out.append(
            {
                "perturbation": config.split("=", 1)[1],
                "model": model,
                "clean_f1": clean_row["strict_f1"],
                "noisy_f1": row["strict_f1"],
                "delta_f1": delta,
                "ci_low": delta,
                "ci_high": delta,
                "gap_vs_other": row["gap_vs_other"],
                "gap_ci_low": row["ci_low"],
                "gap_ci_high": row["ci_high"],
                "flip": row["flip"],
            }
        )
    return out


def robustness_rows_from_outputs(
    clean_prepared: list[Sample],
    clean_predictions: dict[str, dict[str, list]],
    noisy_outputs: list[
        tuple[ExperimentConfig, list[Sample], dict[str, dict[str, list]], list[dict[str, object]]]
    ],
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for config, noisy_prepared, noisy_predictions, noisy_rows in noisy_outputs:
        perturbation = config.perturbation
        for noisy_row in noisy_rows:
            model = str(noisy_row["model"])
            clean_counts = sample_muc_counts(
                clean_prepared,
                clean_predictions[model],
                model,
                config.scheme,
            )
            noisy_counts = sample_muc_counts(
                noisy_prepared,
                noisy_predictions[model],
                model,
                config.scheme,
            )
            ci_low, ci_high, delta = bootstrap_count_gap_ci(
                noisy_counts,
                clean_counts,
                config.scheme,
                config.bootstrap,
                config.seed,
            )
            clean_f1 = corpus_f1(
                clean_prepared,
                clean_predictions[model],
                model,
                config.scheme,
            )
            out.append(
                {
                    "perturbation": perturbation,
                    "model": model,
                    "clean_f1": clean_f1,
                    "noisy_f1": noisy_row["strict_f1"],
                    "delta_f1": delta,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "gap_vs_other": noisy_row["gap_vs_other"],
                    "gap_ci_low": noisy_row["ci_low"],
                    "gap_ci_high": noisy_row["ci_high"],
                    "flip": noisy_row["flip"],
                }
            )
    return out


def run_preprocessing_sweep(
    *,
    dnrti_dir: Path,
    out_dir: Path,
    device: str,
    offline: bool,
    cache_dir: Path | None,
    bootstrap: int | None = None,
) -> list[dict[str, object]]:
    samples, _warnings, _stats = load_dnrti_dataset(dnrti_dir, "test")
    rows: list[dict[str, object]] = []
    for config in iter_preprocessing_sweep_configs():
        if bootstrap is not None:
            config = replace(config, bootstrap=bootstrap)
        rows.extend(
            run_config(
                config,
                samples,
                device=device,
                offline=offline,
                cache_dir=cache_dir,
            )
        )
    write_jsonl(out_dir / "preprocessing_sensitivity.jsonl", rows)
    write_preprocessing_tornado(out_dir / "figures" / "preprocessing_tornado.svg", rows)
    write_preprocessing_report(out_dir / "preprocessing_sensitivity.md", rows)
    return rows


def run_robustness_eval(
    *,
    dnrti_dir: Path,
    out_dir: Path,
    device: str,
    offline: bool,
    cache_dir: Path | None,
    bootstrap: int | None = None,
) -> list[dict[str, object]]:
    samples, _warnings, _stats = load_dnrti_dataset(dnrti_dir, "test")
    configs = list(iter_robustness_configs())
    if bootstrap is not None:
        configs = [replace(config, bootstrap=bootstrap) for config in configs]

    clean_config = configs[0]
    clean_prepared, clean_predictions, _clean_rows = run_config_with_predictions(
        clean_config,
        samples,
        device=device,
        offline=offline,
        cache_dir=cache_dir,
    )
    noisy_outputs = []
    for config in configs[1:]:
        prepared, predictions, rows = run_config_with_predictions(
            config,
            samples,
            device=device,
            offline=offline,
            cache_dir=cache_dir,
        )
        noisy_outputs.append((config, prepared, predictions, rows))

    rows = robustness_rows_from_outputs(clean_prepared, clean_predictions, noisy_outputs)
    write_jsonl(out_dir / "robustness.jsonl", rows)
    write_robustness_report(out_dir / "robustness.md", rows)
    return rows


def run_subset_study(
    *,
    dnrti_dir: Path,
    out_dir: Path,
    device: str,
    offline: bool,
    cache_dir: Path | None,
    bootstrap: int | None = None,
) -> list[dict[str, object]]:
    samples, _warnings, _stats = load_dnrti_dataset(dnrti_dir, "test")
    full_config = replace(
        PRESETS["pdf_mapping"],
        name="subset_full_prediction_base",
        subset_strategy="all",
        subset_size="all",
        seed=SUBSET_SEEDS[0],
    )
    if bootstrap is not None:
        full_config = replace(full_config, bootstrap=bootstrap)

    prepared, predictions, _rows = run_config_with_predictions(
        full_config,
        samples,
        device=device,
        offline=offline,
        cache_dir=cache_dir,
    )

    rows: list[dict[str, object]] = []
    configs = list(iter_subset_study_configs())
    if bootstrap is not None:
        configs = [replace(config, bootstrap=bootstrap) for config in configs]
    for config in configs:
        rows.extend(subset_rows_from_predictions(config, prepared, predictions))

    write_jsonl(out_dir / "subset_study.jsonl", rows)
    full_winner = full_split_winner(rows)
    min_faithful = min_faithful_subset_by_strategy(rows, full_winner=full_winner)
    write_subset_study_report(
        out_dir / "subset_study.md",
        rows,
        min_faithful=min_faithful,
    )
    for strategy in SUBSET_STRATEGIES:
        if any(row["subset_strategy"] == strategy for row in rows):
            write_subset_curve(out_dir / "figures" / f"subset_{strategy}.svg", rows, strategy)
    return rows


def run_protocol_comparison(
    *,
    dnrti_dir: Path,
    out_dir: Path,
    device: str,
    offline: bool,
    cache_dir: Path | None,
    bootstrap: int | None = None,
) -> list[dict[str, object]]:
    samples, _warnings, _stats = load_dnrti_dataset(dnrti_dir, "test")
    rows: list[dict[str, object]] = []
    checks: list[dict[str, object]] = []
    configs = list(iter_protocol_configs())
    if bootstrap is not None:
        configs = [replace(config, bootstrap=bootstrap) for config in configs]

    for config in configs:
        prepared, predictions, config_rows = run_config_with_predictions(
            config,
            samples,
            device=device,
            offline=offline,
            cache_dir=cache_dir,
        )
        for row in config_rows:
            enriched = dict(row)
            enriched["protocol"] = config.name
            rows.append(enriched)
        for model_name in config.models:
            checks.append(
                seqeval_cross_check(
                    prepared,
                    predictions[model_name],
                    model_name,
                    config,
                )
            )

    coverage = [
        mapping_coverage_for_model(samples, model) for model in PRESETS["pdf_mapping"].models
    ]
    write_jsonl(out_dir / "protocol_comparison.jsonl", rows)
    write_jsonl(out_dir / "protocol_seqeval_crosscheck.jsonl", checks)
    write_jsonl(out_dir / "protocol_unique_label_coverage.jsonl", coverage)
    write_protocol_comparison_report(out_dir / "protocol_comparison.md", rows, checks, coverage)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Fork 1 experiments.")
    parser.add_argument(
        "--sweep",
        choices=["preprocessing", "subset", "protocol", "intrinsic", "operational", "robustness"],
        required=True,
    )
    parser.add_argument("--dnrti-dir", type=Path, default=Path("data/dnrti"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/fork1"))
    parser.add_argument("--device", choices=["cpu", "mps", "both"], default="mps")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("model_cache/fork1"))
    parser.add_argument("--bootstrap", type=int, default=None)
    args = parser.parse_args(argv)

    if args.sweep == "preprocessing":
        run_preprocessing_sweep(
            dnrti_dir=args.dnrti_dir,
            out_dir=args.out_dir,
            device=args.device,
            offline=args.offline,
            cache_dir=args.cache_dir,
            bootstrap=args.bootstrap,
        )
    elif args.sweep == "subset":
        run_subset_study(
            dnrti_dir=args.dnrti_dir,
            out_dir=args.out_dir,
            device=args.device,
            offline=args.offline,
            cache_dir=args.cache_dir,
            bootstrap=args.bootstrap,
        )
    elif args.sweep == "protocol":
        run_protocol_comparison(
            dnrti_dir=args.dnrti_dir,
            out_dir=args.out_dir,
            device=args.device,
            offline=args.offline,
            cache_dir=args.cache_dir,
            bootstrap=args.bootstrap,
        )
    elif args.sweep == "intrinsic":
        run_intrinsic_eval(
            dnrti_dir=args.dnrti_dir,
            out_dir=args.out_dir,
            offline=args.offline,
            cache_dir=args.cache_dir,
        )
    elif args.sweep == "operational":
        run_operational_eval(
            dnrti_dir=args.dnrti_dir,
            out_dir=args.out_dir,
            offline=args.offline,
            cache_dir=args.cache_dir,
            requested_device=args.device,
        )
    elif args.sweep == "robustness":
        run_robustness_eval(
            dnrti_dir=args.dnrti_dir,
            out_dir=args.out_dir,
            device=args.device,
            offline=args.offline,
            cache_dir=args.cache_dir,
            bootstrap=args.bootstrap,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
