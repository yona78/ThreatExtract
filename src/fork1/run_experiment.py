from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from fork1.config import PRESETS, ExperimentConfig
from fork1.data import Sample, extract_bio_spans, load_dnrti_dataset
from fork1.metrics import bootstrap_gap_ci, corpus_f1, mcnemar
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
    prepared: list[Sample] = []
    for sample in samples:
        normalized_tokens = tuple(
            normalize_text(token, config.normalization) for token in sample.tokens
        )
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
    return predictions


def run_config(
    config: ExperimentConfig,
    samples: list[Sample],
    *,
    device: str,
    offline: bool,
    cache_dir: Path | None,
) -> list[dict[str, object]]:
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
    write_preprocessing_report(out_dir / "preprocessing_sensitivity.md", rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Fork 1 experiments.")
    parser.add_argument("--sweep", choices=["preprocessing"], required=True)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
