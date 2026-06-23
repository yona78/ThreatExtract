from __future__ import annotations

import argparse
import html
import json
from dataclasses import replace
from pathlib import Path

from fork1.config import PRESETS, ExperimentConfig
from fork1.data import Sample, extract_bio_spans, load_dnrti_dataset
from fork1.metrics import (
    bootstrap_count_gap_ci,
    bootstrap_gap_ci,
    corpus_f1,
    mcnemar,
    sample_muc_counts,
)
from fork1.perturb import PERTURBATIONS, keyboard_typo, random_case
from fork1.preprocess import DETOKENIZERS, iter_contexts, normalize_text
from fork1.runner import MODEL_ALIASES, HfTokenClassificationRunner


SWEEP_LEVELS = {
    "detok": ("single_space", "punct_aware"),
    "context": ("sentence", "document", "window"),
    "max_length": (128, 256, "full"),
    "normalization": ("none", "nfkc", "refang", "lower"),
    "alignment": ("overlap", "majority", "contained"),
}


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


def prepare_samples_for_config(samples: list[Sample], config: ExperimentConfig) -> list[Sample]:
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
    return prepared, predictions_by_model, rows


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


def write_robustness_report(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Robustness Perturbations",
        "",
        "| Perturbation | Model | Clean F1 | Noisy F1 | Delta F1 | 95% CI |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['perturbation']} | {row['model']} | {float(row['clean_f1']):.4f} | "
            f"{float(row['noisy_f1']):.4f} | {float(row['delta_f1']):.4f} | "
            f"[{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}] |"
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Fork 1 experiments.")
    parser.add_argument("--sweep", choices=["preprocessing", "robustness"], required=True)
    parser.add_argument("--dnrti-dir", type=Path, default=Path("data/dnrti"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/fork1"))
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
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
