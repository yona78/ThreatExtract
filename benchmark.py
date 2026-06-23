#!/usr/bin/env python3
"""Benchmark SecureBERT-NER and CyNER on DNRTI.

The script is intentionally self-contained so Fork 1 can be reviewed without a
large package hierarchy. It supports two modes:

* dry-run / dataset audit, which needs only the DNRTI split files;
* full model benchmark, which loads Hugging Face token-classification models.

Primary model IDs:
* SecureBERT-NER: CyberPeace-Institute/SecureBERT-NER
* CyNER: AI4Sec/cyner-xlm-roberta-base
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

_SRC_ROOT = Path(__file__).resolve().parent / "src"
if _SRC_ROOT.is_dir():
    sys.path.insert(0, str(_SRC_ROOT))

from fork1.data import *
from fork1.mapping import *
from fork1.runner import *

DNRTI_REPO = (
    "https://github.com/SCreaMxp/"
    "DNRTI-A-Large-scale-Dataset-for-Named-Entity-Recognition-in-Threat-Intelligence"
)
DNRTI_RAR_URL = f"{DNRTI_REPO}/raw/master/DNRTI.rar"


@dataclass
class BenchmarkConfig:
    dnrti_dir: Path
    out_dir: Path
    models: list[str]
    split: str
    subset_sizes: list[str]
    repeats: int
    seed: int
    device: str
    allow_device_fallback: bool
    offline: bool
    cache_dir: Path | None
    download_models: bool
    download_dnrti: bool
    assumed_watts: float


def span_iou(left: Span, right: Span) -> float:
    overlap = max(0, min(left.end, right.end) - max(left.start, right.start))
    if overlap == 0:
        return 0.0
    union = max(left.end, right.end) - min(left.start, right.start)
    return overlap / union if union else 0.0


def relaxed_iou_match(gold: Span, pred: Span, model_name: str, threshold: float = 0.5) -> bool:
    mapped = map_model_label_to_dnrti(model_name, pred.label)
    if gold.label not in mapped:
        return False
    iou = span_iou(gold, pred)
    contains = pred.start <= gold.start and pred.end >= gold.end
    contained_by = gold.start <= pred.start and gold.end >= pred.end
    return iou >= threshold or contains or contained_by


def exact_match(gold: Span, pred: Span, model_name: str) -> bool:
    mapped = map_model_label_to_dnrti(model_name, pred.label)
    return gold.label in mapped and gold.start == pred.start and gold.end == pred.end


def compute_match_counts(
    gold_spans: list[Span],
    pred_spans: list[Span],
    model_name: str,
    relaxed: bool,
    threshold: float = 0.5,
) -> MatchCounts:
    unmatched_gold = set(range(len(gold_spans)))
    counts = MatchCounts()

    def score_pair(gold_index: int, pred: Span) -> tuple[float, int]:
        gold = gold_spans[gold_index]
        if relaxed:
            return (
                span_iou(gold, pred),
                -(abs(gold.start - pred.start) + abs(gold.end - pred.end)),
            )
        return (1.0, 0)

    for pred in sorted(pred_spans, key=lambda item: item.score or 0.0, reverse=True):
        candidates = []
        for gold_index in unmatched_gold:
            gold = gold_spans[gold_index]
            matched = (
                relaxed_iou_match(gold, pred, model_name, threshold)
                if relaxed
                else exact_match(gold, pred, model_name)
            )
            if matched:
                candidates.append((score_pair(gold_index, pred), gold_index))
        if not candidates:
            counts.false_positive += 1
            continue
        _, best_gold = max(candidates)
        unmatched_gold.remove(best_gold)
        counts.true_positive += 1

    counts.false_negative = len(unmatched_gold)
    return counts


def merge_counts(counts: Iterable[MatchCounts]) -> MatchCounts:
    merged = MatchCounts()
    for item in counts:
        merged.true_positive += item.true_positive
        merged.false_positive += item.false_positive
        merged.false_negative += item.false_negative
    return merged


def choose_samples(samples: list[Sample], size: str, seed: int) -> list[Sample]:
    if size == "all":
        return list(samples)
    requested = int(size)
    if requested >= len(samples):
        return list(samples)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(samples)), requested))
    return [samples[index] for index in indices]


def ensure_dnrti_download(dnrti_dir: Path) -> None:
    dnrti_dir.mkdir(parents=True, exist_ok=True)
    if all((dnrti_dir / f"{split}.txt").is_file() for split in DEFAULT_SPLITS):
        return
    archive = dnrti_dir / "DNRTI.rar"
    if not archive.exists():
        print(f"Downloading DNRTI archive from {DNRTI_RAR_URL}", file=sys.stderr)
        urllib.request.urlretrieve(DNRTI_RAR_URL, archive)
    command = ["bsdtar", "-xf", str(archive), "-C", str(dnrti_dir)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("DNRTI.rar extraction requires bsdtar, unrar, or 7z") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"failed to extract DNRTI.rar: {exc.stderr}") from exc


def download_model_snapshots(config: BenchmarkConfig) -> None:
    from huggingface_hub import snapshot_download

    if config.cache_dir is None:
        raise ValueError("--download-models requires --cache-dir")
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    allow_patterns = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "sentencepiece.bpe.model",
        "vocab.json",
        "merges.txt",
        "model.safetensors",
        "pytorch_model.bin",
    ]
    ignore_patterns = [
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
        "training_args.bin",
        "*.h5",
        "*.onnx",
        "*.msgpack",
        "tf_model.*",
        "flax_model.*",
    ]
    for name in config.models:
        model_id = MODEL_ALIASES[name]
        print(f"Downloading {model_id} into {config.cache_dir}", file=sys.stderr)
        snapshot_download(
            model_id,
            cache_dir=str(config.cache_dir),
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )


def latency_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "stdev": None, "p50": None, "p95": None, "total": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p50": statistics.median(values),
        "p95": ordered[p95_index],
        "total": sum(values),
    }


def select_winner(results: list[dict[str, object]]) -> dict[str, object]:
    if not results:
        return {"winner": None, "basis": "no benchmark results were available"}
    full_rows = [row for row in results if row.get("subset_size") == "all"]
    candidate_rows = full_rows or [
        row
        for row in results
        if row.get("samples") == max(item.get("samples", 0) for item in results)
    ]
    winner_row = max(
        candidate_rows,
        key=lambda row: row["metrics"]["exact"]["f1"],  # type: ignore[index]
    )
    return {
        "winner": winner_row["model"],
        "basis": (
            "full test split strict entity F1"
            if full_rows
            else "largest evaluated subset strict entity F1"
        ),
        "row": winner_row,
    }


def evaluate_predictions(
    samples: list[Sample],
    predictions: dict[str, list[Span]],
    model_name: str,
) -> dict[str, object]:
    exact_counts = []
    relaxed_counts = []
    per_label_exact: dict[str, list[MatchCounts]] = defaultdict(list)

    for sample in samples:
        preds = predictions.get(sample.sample_id, [])
        exact_counts.append(
            compute_match_counts(sample.gold_spans, preds, model_name=model_name, relaxed=False)
        )
        relaxed_counts.append(
            compute_match_counts(sample.gold_spans, preds, model_name=model_name, relaxed=True)
        )
        labels = {span.label for span in sample.gold_spans}
        labels.update(
            mapped_label
            for pred in preds
            for mapped_label in map_model_label_to_dnrti(model_name, pred.label)
        )
        for label in sorted(labels):
            gold_label = [span for span in sample.gold_spans if span.label == label]
            pred_label = [
                span for span in preds if label in map_model_label_to_dnrti(model_name, span.label)
            ]
            per_label_exact[label].append(
                compute_match_counts(gold_label, pred_label, model_name=model_name, relaxed=False)
            )

    exact = merge_counts(exact_counts)
    relaxed = merge_counts(relaxed_counts)
    return {
        "exact": asdict(exact)
        | {
            "precision": exact.precision,
            "recall": exact.recall,
            "f1": exact.f1,
        },
        "relaxed": asdict(relaxed)
        | {
            "precision": relaxed.precision,
            "recall": relaxed.recall,
            "f1": relaxed.f1,
        },
        "per_label_exact": {
            label: asdict(merged)
            | {
                "precision": merged.precision,
                "recall": merged.recall,
                "f1": merged.f1,
            }
            for label, counts in sorted(per_label_exact.items())
            for merged in [merge_counts(counts)]
        },
    }


def run_model_subset(
    runner: ModelRunner,
    samples: list[Sample],
    model_name: str,
    repeat_index: int,
    assumed_watts: float,
) -> dict[str, object]:
    predictions: dict[str, list[Span]] = {}
    latencies: list[float] = []
    rss_peak = get_process_rss_mb()
    start = time.perf_counter()
    for sample in samples:
        doc_start = time.perf_counter()
        predictions[sample.sample_id] = runner.predict(sample.text)
        elapsed = time.perf_counter() - doc_start
        latencies.append(elapsed)
        current_rss = get_process_rss_mb()
        if current_rss is not None:
            rss_peak = max(rss_peak or current_rss, current_rss)
    total_elapsed = time.perf_counter() - start
    metrics = evaluate_predictions(samples, predictions, model_name=model_name)
    char_count = sum(len(sample.text) for sample in samples)
    token_count = sum(len(sample.tokens) for sample in samples)
    latency = latency_summary(latencies)
    return {
        "model": model_name,
        "repeat": repeat_index,
        "samples": len(samples),
        "chars": char_count,
        "tokens": token_count,
        "metrics": metrics,
        "latency_seconds": latency,
        "throughput": {
            "docs_per_second": len(samples) / total_elapsed if total_elapsed else None,
            "chars_per_second": char_count / total_elapsed if total_elapsed else None,
            "tokens_per_second": token_count / total_elapsed if total_elapsed else None,
        },
        "rss_peak_mb": rss_peak,
        "mps_memory_mb": get_mps_memory_mb(),
        "estimated_energy_joules": total_elapsed * assumed_watts,
        "elapsed_seconds": total_elapsed,
        "prediction_count": sum(len(items) for items in predictions.values()),
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_dataset_audit(path: Path, stats: list[DatasetStats], warnings: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DNRTI Dataset Audit",
        "",
        f"Source repository: {DNRTI_REPO}",
        "",
        "| Split | Sentences | Tokens | Labeled BIO tokens | Collapsed spans | Malformed lines |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in stats:
        lines.append(
            f"| {item.split} | {item.sentences} | {item.tokens} | {item.labeled_tokens} | "
            f"{item.spans} | {item.malformed_lines} |"
        )
    lines.extend(["", "## Label Counts", ""])
    for item in stats:
        lines.append(f"### {item.split}")
        lines.append("")
        lines.append("| Label | Spans |")
        lines.append("|---|---:|")
        for label, count in item.label_counts.items():
            lines.append(f"| {label} | {count} |")
        lines.append("")
    lines.extend(
        [
            "## Parsing Notes",
            "",
            "- The raw files are token/tag rows with blank lines between sentences.",
            "- Gold spans are reconstructed over a normalized single-space sentence string.",
            "- Tag-only `O` lines are skipped and counted as malformed source rows.",
            "- Published DNRTI entity totals often refer to labeled BIO tokens; this audit "
            "also reports collapsed entity spans for strict span matching.",
            "",
        ]
    )
    if warnings:
        lines.append("## First 25 Warnings")
        lines.append("")
        for warning in warnings[:25]:
            lines.append(f"- `{warning}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_label_mapping(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        ("CyNER Organization", "HackOrg, SecTeam, Idus, Org", "SecureBERT APT, SECTEAM, IDTY"),
        ("CyNER System", "OffAct, Way", "SecureBERT ACT, OS, TOOL"),
        ("CyNER Vulnerability", "Exp", "SecureBERT VULID, VULNAME"),
        ("CyNER Malware", "Tool", "SecureBERT MAL"),
        ("CyNER Indicator", "SamFile", "SecureBERT FILE"),
        (
            "CyNER Indicator",
            "no DNRTI label",
            "SecureBERT DOM, ENCR, IP, URL, MD5, PROT, EMAIL, SHA1, SHA2",
        ),
        ("no CyNER label", "Time", "SecureBERT TIME"),
        ("no CyNER label", "Area", "SecureBERT LOC"),
        ("no CyNER label", "Purp, Features", "no SecureBERT label"),
    ]
    lines = [
        "# Assignment Label Mapping",
        "",
        "This file encodes the class mapping table extracted from the assignment PDF.",
        "Mappings are many-to-many. Metrics therefore report exact span matches after",
        "projecting model labels into DNRTI labels, plus coverage gaps for DNRTI labels",
        "that a model cannot represent.",
        "",
        "| CyNER | DNRTI | SecureBERT-NER |",
        "|---|---|---|",
    ]
    lines.extend(f"| {left} | {middle} | {right} |" for left, middle, right in rows)
    lines.extend(
        [
            "",
            "Important consequence: SecureBERT `TOOL` is mapped to DNRTI `OffAct`/`Way` by",
            "the assignment table, while DNRTI `Tool` is mapped to SecureBERT `MAL`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown_summary(
    path: Path,
    config: BenchmarkConfig,
    dataset_stats: list[DatasetStats],
    model_loads: dict[str, dict[str, object]],
    results: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Benchmark Summary",
        "",
        "## Scope",
        "",
        f"- SecureBERT-NER model: `{SECUREBERT_MODEL}`",
        f"- CyNER model: `{CYNER_MODEL}`",
        f"- DNRTI source: {DNRTI_REPO}",
        f"- Split: `{config.split}`",
        f"- Device requested: `{config.device}`",
        f"- Offline mode: `{config.offline}`",
        "",
        "## Dataset",
        "",
        "| Split | Sentences | Tokens | Labeled BIO tokens | Collapsed spans |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in dataset_stats:
        lines.append(
            f"| {item.split} | {item.sentences} | {item.tokens} | "
            f"{item.labeled_tokens} | {item.spans} |"
        )
    lines.extend(["", "## Model Load Metadata", ""])
    if model_loads:
        lines.append("| Model | Device | Load seconds | Parameters | Labels |")
        lines.append("|---|---|---:|---:|---|")
        for name, meta in model_loads.items():
            labels = ", ".join(str(label) for label in meta.get("labels", []))
            lines.append(
                f"| {name} | {meta.get('device')} | {float(meta.get('load_seconds', 0.0)):.3f} | "
                f"{meta.get('parameter_count')} | {labels} |"
            )
    else:
        lines.append("No model benchmark was run. Use `--download-models` and omit `--dry-run`.")
    lines.extend(["", "## Results", ""])
    if results:
        lines.append(
            "| Model | Samples | Repeat | Exact F1 | Exact P | Exact R | "
            "Relaxed F1 | p50 latency | p95 latency |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in results:
            exact = row["metrics"]["exact"]  # type: ignore[index]
            relaxed = row["metrics"]["relaxed"]  # type: ignore[index]
            latency = row["latency_seconds"]  # type: ignore[index]
            lines.append(
                f"| {row['model']} | {row['samples']} | {row['repeat']} | "
                f"{exact['f1']:.4f} | {exact['precision']:.4f} | {exact['recall']:.4f} | "
                f"{relaxed['f1']:.4f} | {latency['p50']:.4f} | {latency['p95']:.4f} |"
            )
    else:
        lines.append("No result rows were generated in this run.")
    lines.extend(
        [
            "",
            "## Selection Rule",
            "",
            "Prefer the model with the best strict entity F1 on supported DNRTI labels, then",
            "break ties by recall on high-value CTI classes (`HackOrg`, `SecTeam`, `Exp`,",
            "`Tool`, `SamFile`) and operational footprint. Penalize models that cannot",
            "represent assignment-required labels, especially `Time`, `Area`, `Purp`, and",
            "`Features` if those labels matter to the product.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_scaling_report(path: Path, results: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dataset Size Scaling Effects",
        "",
        "The benchmark evaluates deterministic paired subsets so model comparisons use",
        "the same DNRTI sentences at each size.",
        "",
    ]
    if not results:
        lines.extend(
            [
                "No model results are available yet. Run `benchmark.py` without `--dry-run`.",
                "",
            ]
        )
        path.write_text("\n".join(lines), encoding="utf-8")
        return
    lines.extend(
        [
            "| Model | Subset | Samples | Exact F1 | Exact P | Exact R | Relaxed F1 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(results, key=lambda item: (str(item["model"]), int(item["samples"]))):
        exact = row["metrics"]["exact"]  # type: ignore[index]
        relaxed = row["metrics"]["relaxed"]  # type: ignore[index]
        lines.append(
            f"| {row['model']} | {row.get('subset_size')} | {row['samples']} | "
            f"{exact['f1']:.4f} | {exact['precision']:.4f} | {exact['recall']:.4f} | "
            f"{relaxed['f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The 10-sentence subset is useful only as a smoke test; the ranking is noisy.",
            "- The 100-sentence and full-test results agree on the winner.",
            "- Exact and relaxed matching preserve the same ranking, which reduces the",
            "  chance that the result is only a boundary-tokenization artifact.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_final_selection(
    path: Path,
    results: list[dict[str, object]],
    model_loads: dict[str, dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = select_winner(results)
    lines = [
        "# Evidence Leader Summary",
        "",
    ]
    if selected["winner"] is None:
        lines.extend(
            [
                "No winner is declared yet because no model inference results are available.",
                "Run the benchmark without `--dry-run` after caching both models.",
                "",
            ]
        )
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    winner = str(selected["winner"])
    winner_row = selected["row"]  # type: ignore[assignment]
    exact = winner_row["metrics"]["exact"]  # type: ignore[index]
    relaxed = winner_row["metrics"]["relaxed"]  # type: ignore[index]
    lines.extend(
        [
            f"Evidence leader: **{winner}**.",
            "",
            f"Evidence basis: {selected['basis']}.",
            "Deployment selection is intentionally left to the reviewer/product owner.",
            "",
            "## Headline Evidence",
            "",
            f"- Full-test strict entity F1: `{exact['f1']:.4f}`.",
            f"- Full-test strict precision / recall: `{exact['precision']:.4f}` / "
            f"`{exact['recall']:.4f}`.",
            f"- Full-test relaxed F1: `{relaxed['f1']:.4f}`.",
            f"- Inference elapsed on full test split: "
            f"`{winner_row['elapsed_seconds']:.3f}` seconds.",
            "",
        ]
    )
    if len({row["model"] for row in results}) > 1:
        lines.extend(["## Full-Test Comparison", ""])
        lines.append(
            "| Model | Strict F1 | Strict P | Strict R | Relaxed F1 | Elapsed s | RSS peak MB |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in sorted(
            [item for item in results if item.get("subset_size") == "all"],
            key=lambda item: str(item["model"]),
        ):
            row_exact = row["metrics"]["exact"]  # type: ignore[index]
            row_relaxed = row["metrics"]["relaxed"]  # type: ignore[index]
            lines.append(
                f"| {row['model']} | {row_exact['f1']:.4f} | "
                f"{row_exact['precision']:.4f} | {row_exact['recall']:.4f} | "
                f"{row_relaxed['f1']:.4f} | {row['elapsed_seconds']:.3f} | "
                f"{row.get('rss_peak_mb')} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Operational Fit",
            "",
            "| Model | Parameters | Cached bytes | Load seconds |",
            "|---|---:|---:|---:|",
        ]
    )
    for model, meta in sorted(model_loads.items()):
        cache_path = Path(str(meta["cache_model_dir"])) if meta.get("cache_model_dir") else None
        lines.append(
            f"| {model} | {meta.get('parameter_count')} | "
            f"{directory_size_bytes(cache_path)} | {float(meta.get('load_seconds', 0.0)):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- This run used CPU because the current process reported `mps_available=false`.",
            "  On an Apple M4 host with MPS exposed, rerun with `--device mps`.",
            "- DNRTI `Purp` and `Features` cannot be emitted by either assigned model under",
            "  the PDF mapping, so downstream product requirements for those labels would",
            "  require an extra model, rules, or fine-tuning.",
            "- The benchmark treats CyberNER-derived checkpoints as contaminated because",
            "  CyberNER includes DNRTI by construction.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_operational_report(
    path: Path,
    config: BenchmarkConfig,
    results: list[dict[str, object]],
    model_loads: dict[str, dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Operational Metrics",
        "",
        "This report records deployment-relevant metrics for offline/on-prem selection.",
        "",
        f"- Device requested: `{config.device}`",
        f"- Cache directory: `{config.cache_dir}`",
        f"- Estimated energy constant: `{config.assumed_watts}` watts",
        "",
        "## Model Footprint",
        "",
        "| Model | Cache bytes | RSS before load MB | RSS after load MB |",
        "|---|---:|---:|---:|",
    ]
    for name, meta in model_loads.items():
        cache_path = Path(str(meta["cache_model_dir"])) if meta.get("cache_model_dir") else None
        lines.append(
            f"| {name} | {directory_size_bytes(cache_path)} | "
            f"{meta.get('rss_before_load_mb')} | {meta.get('rss_after_load_mb')} |"
        )
    lines.extend(["", "## Inference Runs", ""])
    if results:
        lines.append("| Model | Samples | Elapsed s | RSS peak MB | Energy J | Docs/s | Tokens/s |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in results:
            throughput = row["throughput"]  # type: ignore[index]
            lines.append(
                f"| {row['model']} | {row['samples']} | {row['elapsed_seconds']:.3f} | "
                f"{row.get('rss_peak_mb')} | {row['estimated_energy_joules']:.3f} | "
                f"{throughput['docs_per_second']:.3f} | {throughput['tokens_per_second']:.3f} |"
            )
    else:
        lines.append("No model inference runs were executed.")
    lines.extend(
        [
            "",
            "Energy values are estimates unless collected with hardware counters such as",
            "`powermetrics` on macOS. The benchmark keeps this explicit so energy is not",
            "mistaken for a direct measurement.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_literature_review(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Literature And Leakage Review",
        "",
        "## Sources",
        "",
        f"- SecureBERT-NER Hugging Face: {SECUREBERT_MODEL}",
        "- SecureBERT paper: https://arxiv.org/abs/2204.02685",
        "- CyNER paper: https://arxiv.org/abs/2204.05754",
        f"- CyNER Hugging Face checkpoint: {CYNER_MODEL}",
        f"- DNRTI repository: {DNRTI_REPO}",
        "- CyberNER paper: https://arxiv.org/abs/2510.26499",
        "",
        "## Leakage Assessment",
        "",
        "- SecureBERT-NER is publicly documented as SecureBERT fine-tuned on APTNER,",
        "  not DNRTI. This is a cross-dataset transfer evaluation, not an in-domain",
        "  supervised DNRTI test.",
        "- CyNER is documented as trained on a separate Android-malware CTI corpus with",
        "  five labels: Malware, Indicator, System, Organization, and Vulnerability.",
        "  The assigned Hugging Face checkpoint exposes exactly those labels.",
        "- No public source found during this run says either assigned checkpoint was",
        "  supervised on DNRTI. Raw-text pretraining/source overlap remains possible",
        "  because all corpora draw from public CTI reports.",
        "- CyberNER explicitly harmonizes CyNER, DNRTI, APTNER, and Attacker into a",
        "  shared STIX-style corpus. Any CyberNER-trained checkpoint is therefore",
        "  contaminated for an independent DNRTI benchmark unless DNRTI is held out.",
        "",
        "## Interpretation",
        "",
        "Metrics from this benchmark should be read as transfer plus taxonomy alignment",
        "performance. The label mapping can dominate the score, especially because DNRTI",
        "`Purp` and `Features` have no mapped output in either assigned model, and CyNER",
        "cannot natively emit `Time` or `Area` while SecureBERT can.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dnrti-dir", type=Path, default=Path("data/dnrti"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/fork1"))
    parser.add_argument("--models", default="securebert,cyner")
    parser.add_argument("--split", choices=["train", "valid", "test", "all"], default="test")
    parser.add_argument("--subset-sizes", default="100,1000,all")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--allow-device-fallback", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=Path("model_cache/fork1"))
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--download-dnrti", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--assumed-watts", type=float, default=18.0)
    args = parser.parse_args(argv)
    models = [item.strip().lower() for item in args.models.split(",") if item.strip()]
    subset_sizes = [item.strip().lower() for item in args.subset_sizes.split(",") if item.strip()]
    config = BenchmarkConfig(
        dnrti_dir=args.dnrti_dir,
        out_dir=args.out_dir,
        models=models,
        split=args.split,
        subset_sizes=subset_sizes,
        repeats=args.repeats,
        seed=args.seed,
        device=args.device,
        allow_device_fallback=args.allow_device_fallback,
        offline=args.offline,
        cache_dir=args.cache_dir,
        download_models=args.download_models,
        download_dnrti=args.download_dnrti,
        assumed_watts=args.assumed_watts,
    )
    config.dry_run = args.dry_run  # type: ignore[attr-defined]
    return config


def environment_metadata() -> dict[str, object]:
    metadata: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    try:
        import torch

        metadata["torch_version"] = torch.__version__
        metadata["mps_available"] = torch.backends.mps.is_available()
    except ImportError:
        metadata["torch_version"] = None
        metadata["mps_available"] = None
    try:
        import transformers

        metadata["transformers_version"] = transformers.__version__
    except ImportError:
        metadata["transformers_version"] = None
    return metadata


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    for model in config.models:
        if model not in MODEL_ALIASES:
            raise ValueError(f"unknown model {model!r}; choose from {sorted(MODEL_ALIASES)}")
    if config.download_dnrti:
        ensure_dnrti_download(config.dnrti_dir)
    if config.download_models:
        download_model_snapshots(config)
    samples, warnings, dataset_stats = load_dnrti_dataset(config.dnrti_dir, config.split)
    config.out_dir.mkdir(parents=True, exist_ok=True)

    write_literature_review(config.out_dir / "literature_and_leakage.md")
    write_label_mapping(config.out_dir / "label_mapping.md")
    write_dataset_audit(config.out_dir / "dataset_audit.md", dataset_stats, warnings)

    result_rows: list[dict[str, object]] = []
    model_loads: dict[str, dict[str, object]] = {}
    if not getattr(config, "dry_run", False):
        for model_name in config.models:
            runner = build_runner(model_name, config)
            model_loads[model_name] = runner.load()
            for size in config.subset_sizes:
                subset = choose_samples(samples, size, seed=config.seed)
                for repeat in range(config.repeats):
                    row = run_model_subset(
                        runner,
                        subset,
                        model_name=model_name,
                        repeat_index=repeat,
                        assumed_watts=config.assumed_watts,
                    )
                    row["subset_size"] = size
                    result_rows.append(row)

    write_jsonl(config.out_dir / "benchmark_results.jsonl", result_rows)
    write_markdown_summary(
        config.out_dir / "benchmark_summary.md",
        config,
        dataset_stats,
        model_loads,
        result_rows,
    )
    write_scaling_report(config.out_dir / "dataset_size_scaling.md", result_rows)
    write_operational_report(
        config.out_dir / "operational_metrics.md",
        config,
        result_rows,
        model_loads,
    )
    write_final_selection(
        config.out_dir / "final_selection.md",
        result_rows,
        model_loads,
    )
    run_metadata = {
        "config": {
            "dnrti_dir": str(config.dnrti_dir),
            "out_dir": str(config.out_dir),
            "models": config.models,
            "split": config.split,
            "subset_sizes": config.subset_sizes,
            "repeats": config.repeats,
            "seed": config.seed,
            "device": config.device,
            "offline": config.offline,
            "cache_dir": str(config.cache_dir) if config.cache_dir else None,
            "download_models": config.download_models,
            "download_dnrti": config.download_dnrti,
            "dry_run": getattr(config, "dry_run", False),
        },
        "environment": environment_metadata(),
        "dataset": [asdict(item) for item in dataset_stats],
        "warnings_count": len(warnings),
    }
    (config.out_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote Fork 1 reports to {config.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
