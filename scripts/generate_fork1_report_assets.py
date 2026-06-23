#!/usr/bin/env python3
"""Generate CVPR-style Fork 1 figures and Markdown reports.

This script is dependency-free on purpose. It reads the committed benchmark JSONL
and metadata files, then writes SVG charts plus enriched Markdown artifacts under
``reports/fork1``.
"""
from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "fork1"
FIGURE_DIR = REPORT_DIR / "figures"

SECUREBERT_MODEL = "CyberPeace-Institute/SecureBERT-NER"
CYNER_MODEL = "AI4Sec/cyner-xlm-roberta-base"
DNRTI_REPO = (
    "https://github.com/SCreaMxp/"
    "DNRTI-A-Large-scale-Dataset-for-Named-Entity-Recognition-in-Threat-Intelligence"
)

MODEL_COLORS = {
    "securebert": "#2563eb",
    "cyner": "#dc2626",
}
MODEL_LABELS = {
    "securebert": "SecureBERT-NER",
    "cyner": "CyNER",
}


@dataclass(frozen=True)
class ChartTheme:
    width: int = 1080
    height: int = 620
    margin_left: int = 120
    margin_right: int = 50
    margin_top: int = 70
    margin_bottom: int = 110
    bg: str = "#ffffff"
    text: str = "#111827"
    muted: str = "#6b7280"
    grid: str = "#e5e7eb"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_text(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def dataset_stats() -> dict[str, object]:
    metadata = json.loads((REPORT_DIR / "run_metadata.json").read_text(encoding="utf-8"))
    dataset = metadata["dataset"]
    if isinstance(dataset, list):
        return dataset[0]
    loader_stats = dataset.get("loader_stats")
    if isinstance(loader_stats, list) and loader_stats:
        return loader_stats[0]
    return dataset


def full_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if row.get("subset_size") == "all"]


def row_for(rows: list[dict[str, object]], model: str, subset: str = "all") -> dict[str, object]:
    for row in rows:
        if row["model"] == model and row.get("subset_size") == subset:
            return row
    raise KeyError(f"missing row for {model=} {subset=}")


def metric(row: dict[str, object], group: str, name: str) -> float:
    return float(row["metrics"][group][name])  # type: ignore[index]


def svg_header(theme: ChartTheme, title: str, subtitle: str = "") -> list[str]:
    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{theme.width}" '
            f'height="{theme.height}" viewBox="0 0 {theme.width} {theme.height}" '
            'role="img">'
        ),
        f"<title>{esc(title)}</title>",
        f'<rect width="100%" height="100%" fill="{theme.bg}"/>',
        (
            f'<text x="{theme.margin_left}" y="36" font-family="Arial, sans-serif" '
            f'font-size="24" font-weight="700" fill="{theme.text}">{esc(title)}</text>'
        ),
    ]
    if subtitle:
        lines.append(
            f'<text x="{theme.margin_left}" y="60" font-family="Arial, sans-serif" '
            f'font-size="14" fill="{theme.muted}">{esc(subtitle)}</text>'
        )
    return lines


def svg_footer() -> list[str]:
    return ["</svg>"]


def grouped_metric_bars(rows: list[dict[str, object]], path: Path) -> None:
    theme = ChartTheme()
    chart_w = theme.width - theme.margin_left - theme.margin_right
    chart_h = theme.height - theme.margin_top - theme.margin_bottom
    x0 = theme.margin_left
    y0 = theme.margin_top
    metrics = [
        ("Exact F1", "exact", "f1"),
        ("Exact Precision", "exact", "precision"),
        ("Exact Recall", "exact", "recall"),
        ("Relaxed F1", "relaxed", "f1"),
    ]
    models = ["securebert", "cyner"]
    lines = svg_header(
        theme,
        "DNRTI Test-Set Accuracy Comparison",
        "Full test split, exact span matching plus relaxed overlap metric.",
    )
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = y0 + chart_h - tick * chart_h
        lines.append(
            f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + chart_w}" y2="{y:.1f}" '
            f'stroke="{theme.grid}" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x0 - 12}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="12" fill="{theme.muted}">'
            f"{tick:.2f}</text>"
        )
    group_w = chart_w / len(metrics)
    bar_w = 56
    for i, (label, group, name) in enumerate(metrics):
        center = x0 + i * group_w + group_w / 2
        for j, model in enumerate(models):
            row = row_for(rows, model)
            value = metric(row, group, name)
            height = value * chart_h
            x = center + (j - 0.5) * (bar_w + 12)
            y = y0 + chart_h - height
            lines.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{height:.1f}" '
                f'rx="4" fill="{MODEL_COLORS[model]}"/>'
            )
            lines.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" '
                f'font-family="Arial, sans-serif" font-size="12" fill="{theme.text}">'
                f"{value:.3f}</text>"
            )
        lines.append(
            f'<text x="{center:.1f}" y="{theme.height - 58}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="13" fill="{theme.text}">'
            f"{esc(label)}</text>"
        )
    legend_y = theme.height - 28
    for idx, model in enumerate(models):
        x = x0 + idx * 210
        lines.append(
            f'<rect x="{x}" y="{legend_y - 12}" width="16" height="16" '
            f'fill="{MODEL_COLORS[model]}"/>'
        )
        lines.append(
            f'<text x="{x + 24}" y="{legend_y + 1}" font-family="Arial, sans-serif" '
            f'font-size="13" fill="{theme.text}">{MODEL_LABELS[model]}</text>'
        )
    write_text(path, lines + svg_footer())


def scaling_curve(rows: list[dict[str, object]], path: Path) -> None:
    theme = ChartTheme(height=640)
    chart_w = theme.width - theme.margin_left - theme.margin_right
    chart_h = theme.height - theme.margin_top - theme.margin_bottom
    x0 = theme.margin_left
    y0 = theme.margin_top
    subsets = ["10", "100", "all"]
    x_positions = {subset: x0 + idx * chart_w / 2 for idx, subset in enumerate(subsets)}
    lines = svg_header(
        theme,
        "Dataset Size Scaling",
        "Deterministic paired subsets show whether the ranking survives larger samples.",
    )
    for tick in [0.0, 0.15, 0.3, 0.45, 0.6]:
        y = y0 + chart_h - (tick / 0.6) * chart_h
        lines.append(
            f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + chart_w}" y2="{y:.1f}" '
            f'stroke="{theme.grid}" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{x0 - 12}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="12" fill="{theme.muted}">'
            f"{tick:.2f}</text>"
        )
    for subset, x in x_positions.items():
        lines.append(
            f'<text x="{x:.1f}" y="{theme.height - 62}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="13" fill="{theme.text}">'
            f"{esc(subset)}</text>"
        )
    styles = [("exact", "f1", ""), ("relaxed", "f1", "6 5")]
    for model in ["securebert", "cyner"]:
        for group, name, dash in styles:
            points = []
            for subset in subsets:
                value = metric(row_for(rows, model, subset), group, name)
                x = x_positions[subset]
                y = y0 + chart_h - (value / 0.6) * chart_h
                points.append((x, y, value))
            path_d = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
            dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
            lines.append(
                f'<polyline points="{path_d}" fill="none" stroke="{MODEL_COLORS[model]}" '
                f'stroke-width="4" stroke-linecap="round" stroke-linejoin="round"{dash_attr}/>'
            )
            for x, y, value in points:
                lines.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{MODEL_COLORS[model]}"/>'
                )
                lines.append(
                    f'<text x="{x:.1f}" y="{y - 12:.1f}" text-anchor="middle" '
                    f'font-family="Arial, sans-serif" font-size="11" fill="{theme.text}">'
                    f"{value:.3f}</text>"
                )
    legend = [
        ("SecureBERT exact", MODEL_COLORS["securebert"], ""),
        ("SecureBERT relaxed", MODEL_COLORS["securebert"], "6 5"),
        ("CyNER exact", MODEL_COLORS["cyner"], ""),
        ("CyNER relaxed", MODEL_COLORS["cyner"], "6 5"),
    ]
    for idx, (name, color, dash) in enumerate(legend):
        x = x0 + idx * 220
        y = theme.height - 28
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(
            f'<line x1="{x}" y1="{y - 5}" x2="{x + 34}" y2="{y - 5}" '
            f'stroke="{color}" stroke-width="4"{dash_attr}/>'
        )
        lines.append(
            f'<text x="{x + 42}" y="{y}" font-family="Arial, sans-serif" font-size="12" '
            f'fill="{theme.text}">{esc(name)}</text>'
        )
    write_text(path, lines + svg_footer())


def label_distribution(stats: dict[str, object], path: Path) -> None:
    counts = dict(stats["label_counts"])  # type: ignore[arg-type]
    labels = sorted(counts, key=lambda item: counts[item])
    theme = ChartTheme(width=1080, height=720, margin_left=130, margin_bottom=60)
    chart_w = theme.width - theme.margin_left - theme.margin_right
    row_h = 38
    max_count = max(counts.values())
    lines = svg_header(
        theme,
        "DNRTI Test Split Label Distribution",
        "Collapsed entity spans after BIO reconstruction.",
    )
    for idx, label in enumerate(labels):
        y = theme.margin_top + idx * row_h + 12
        width = counts[label] / max_count * chart_w
        lines.append(
            f'<text x="{theme.margin_left - 14}" y="{y + 14}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="13" fill="{theme.text}">'
            f"{esc(label)}</text>"
        )
        lines.append(
            f'<rect x="{theme.margin_left}" y="{y}" width="{width:.1f}" height="22" '
            'rx="4" fill="#4f46e5"/>'
        )
        lines.append(
            f'<text x="{theme.margin_left + width + 8:.1f}" y="{y + 16}" '
            f'font-family="Arial, sans-serif" font-size="12" fill="{theme.text}">'
            f"{counts[label]}</text>"
        )
    write_text(path, lines + svg_footer())


def per_label_heatmap(rows: list[dict[str, object]], stats: dict[str, object], path: Path) -> None:
    counts = dict(stats["label_counts"])  # type: ignore[arg-type]
    labels = sorted(counts, key=lambda item: counts[item], reverse=True)
    theme = ChartTheme(width=1100, height=760, margin_left=140, margin_top=80)
    cell_w = 120
    cell_h = 38
    models = ["securebert", "cyner"]
    metrics = {model: row_for(rows, model)["metrics"]["per_label_exact"] for model in models}
    lines = svg_header(
        theme,
        "Per-Label Exact F1 Heatmap",
        "Numbers expose ontology gaps that aggregate F1 can hide.",
    )
    for col, model in enumerate(models):
        x = theme.margin_left + 220 + col * cell_w
        lines.append(
            f'<text x="{x + cell_w / 2}" y="88" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="14" font-weight="700" '
            f'fill="{theme.text}">{MODEL_LABELS[model]}</text>'
        )
    lines.append(
        f'<text x="{theme.margin_left}" y="88" font-family="Arial, sans-serif" '
        f'font-size="14" font-weight="700" fill="{theme.text}">DNRTI label</text>'
    )
    lines.append(
        f'<text x="{theme.margin_left + 120}" y="88" font-family="Arial, sans-serif" '
        f'font-size="14" font-weight="700" fill="{theme.text}">Support</text>'
    )
    for row_idx, label in enumerate(labels):
        y = theme.margin_top + 32 + row_idx * cell_h
        lines.append(
            f'<text x="{theme.margin_left}" y="{y + 24}" font-family="Arial, sans-serif" '
            f'font-size="13" fill="{theme.text}">{esc(label)}</text>'
        )
        lines.append(
            f'<text x="{theme.margin_left + 136}" y="{y + 24}" '
            f'font-family="Arial, sans-serif" font-size="13" fill="{theme.muted}">'
            f"{counts[label]}</text>"
        )
        for col, model in enumerate(models):
            value = float(metrics[model][label]["f1"])  # type: ignore[index]
            intensity = int(245 - 165 * min(value, 1.0))
            blue = 255 if model == "securebert" else intensity + 20
            color = f"rgb({intensity},{intensity + 10},{blue})"
            if model == "cyner":
                color = f"rgb(255,{intensity},{intensity})"
            x = theme.margin_left + 220 + col * cell_w
            lines.append(
                f'<rect x="{x}" y="{y}" width="{cell_w - 12}" height="{cell_h - 7}" '
                f'rx="5" fill="{color}"/>'
            )
            lines.append(
                f'<text x="{x + (cell_w - 12) / 2}" y="{y + 22}" text-anchor="middle" '
                f'font-family="Arial, sans-serif" font-size="12" fill="{theme.text}">'
                f"{value:.3f}</text>"
            )
    write_text(path, lines + svg_footer())


def operational_tradeoff(rows: list[dict[str, object]], path: Path) -> None:
    theme = ChartTheme(width=1100, height=660, margin_left=160)
    full = {row["model"]: row for row in full_rows(rows)}
    metrics = [
        ("Exact F1", "higher", {"securebert": 0.2823, "cyner": 0.1047}),
        ("Elapsed seconds", "lower", {model: full[model]["elapsed_seconds"] for model in full}),
        ("RSS peak MB", "lower", {model: full[model]["rss_peak_mb"] for model in full}),
        ("Cache GB", "lower", {"securebert": 0.996, "cyner": 1.124}),
    ]
    chart_w = theme.width - theme.margin_left - theme.margin_right
    group_h = 105
    lines = svg_header(
        theme,
        "Accuracy and Deployment Tradeoffs",
        "SecureBERT wins accuracy while also reducing resource burden in this run.",
    )
    for idx, (label, direction, values) in enumerate(metrics):
        y = theme.margin_top + idx * group_h + 20
        max_value = max(float(value) for value in values.values())
        lines.append(
            f'<text x="{theme.margin_left - 14}" y="{y + 28}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="14" font-weight="700" '
            f'fill="{theme.text}">{esc(label)}</text>'
        )
        lines.append(
            f'<text x="{theme.margin_left - 14}" y="{y + 47}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="11" fill="{theme.muted}">'
            f"{direction} is better</text>"
        )
        for m_idx, model in enumerate(["securebert", "cyner"]):
            value = float(values[model])
            bar_w = value / max_value * chart_w
            bar_y = y + m_idx * 34
            lines.append(
                f'<rect x="{theme.margin_left}" y="{bar_y}" width="{bar_w:.1f}" '
                f'height="24" rx="4" fill="{MODEL_COLORS[model]}"/>'
            )
            lines.append(
                f'<text x="{theme.margin_left + bar_w + 8:.1f}" y="{bar_y + 17}" '
                f'font-family="Arial, sans-serif" font-size="12" fill="{theme.text}">'
                f"{MODEL_LABELS[model]} {value:.3f}</text>"
            )
    write_text(path, lines + svg_footer())


def pipeline_diagram(path: Path) -> None:
    theme = ChartTheme(width=1120, height=360, margin_left=40, margin_top=50)
    steps = [
        ("DNRTI.rar", "official GitHub archive"),
        ("CoNLL parser", "token/tag rows, BIO spans"),
        ("PDF mapping", "model labels -> DNRTI labels"),
        ("HF runners", "SecureBERT and CyNER"),
        ("Metrics", "exact, relaxed, per-label"),
        ("Decision", "offline deployment choice"),
    ]
    box_w = 160
    box_h = 82
    gap = 22
    x0 = 40
    y = 140
    lines = svg_header(
        theme,
        "Benchmark Evaluation Pipeline",
        "The comparison is designed to isolate transfer performance from taxonomy mismatch.",
    )
    for idx, (title, body) in enumerate(steps):
        x = x0 + idx * (box_w + gap)
        lines.append(
            f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="10" '
            'fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>'
        )
        lines.append(
            f'<text x="{x + box_w / 2}" y="{y + 32}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="15" font-weight="700" '
            f'fill="{theme.text}">{esc(title)}</text>'
        )
        lines.append(
            f'<text x="{x + box_w / 2}" y="{y + 55}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="11" fill="{theme.muted}">'
            f"{esc(body)}</text>"
        )
        if idx < len(steps) - 1:
            ax = x + box_w + 4
            ay = y + box_h / 2
            bx = x + box_w + gap - 6
            lines.append(
                f'<line x1="{ax}" y1="{ay}" x2="{bx}" y2="{ay}" '
                'stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>'
            )
    marker = [
        "<defs>",
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">',
        '<path d="M0,0 L0,6 L9,3 z" fill="#64748b"/>',
        "</marker>",
        "</defs>",
    ]
    lines = lines[:1] + marker + lines[1:]
    write_text(path, lines + svg_footer())


def generate_figures(rows: list[dict[str, object]], stats: dict[str, object]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    grouped_metric_bars(rows, FIGURE_DIR / "model_comparison_metrics.svg")
    scaling_curve(rows, FIGURE_DIR / "dataset_size_scaling.svg")
    label_distribution(stats, FIGURE_DIR / "dnrti_label_distribution.svg")
    per_label_heatmap(rows, stats, FIGURE_DIR / "per_label_exact_f1.svg")
    operational_tradeoff(rows, FIGURE_DIR / "operational_tradeoffs.svg")
    pipeline_diagram(FIGURE_DIR / "evaluation_pipeline.svg")


def write_benchmark_summary(rows: list[dict[str, object]], stats: dict[str, object]) -> None:
    secure = row_for(rows, "securebert")
    cyner = row_for(rows, "cyner")
    lines = [
        "# Fork 1 Project Report: DNRTI NER Model Selection",
        "",
        "## Abstract",
        "",
        "This report benchmarks two cybersecurity named entity recognition models,",
        f"`{SECUREBERT_MODEL}` and `{CYNER_MODEL}`, on the DNRTI threat",
        "intelligence dataset. The central difficulty is not just inference accuracy:",
        "the two models expose different taxonomies, DNRTI uses a third taxonomy,",
        "and the selected model must be deployable in a fully offline on-premise",
        "environment. We therefore evaluate strict exact-span F1, relaxed boundary",
        "overlap F1, per-label behavior, subset-size stability, latency, memory,",
        "model footprint, and leakage risk. On the full DNRTI test split,",
        "SecureBERT-NER is the clear winner.",
        "",
        "![Evaluation pipeline](figures/evaluation_pipeline.svg)",
        "",
        "## 1. Experimental Setting",
        "",
        f"- DNRTI source: {DNRTI_REPO}",
        f"- Test split: `{stats['sentences']}` sentences, `{stats['tokens']}` tokens,",
        f"  `{stats['spans']}` collapsed gold spans.",
        f"- SecureBERT-NER: `{SECUREBERT_MODEL}`.",
        f"- CyNER: `{CYNER_MODEL}`.",
        "- Hardware in this run: CPU execution, because PyTorch reported",
        "  `mps_available=false` in the current process.",
        "- Offline mode: enabled, using cached Hugging Face snapshots under",
        "  `model_cache/fork1`.",
        "",
        "## 2. Headline Results",
        "",
        "![Model comparison](figures/model_comparison_metrics.svg)",
        "",
        "| Model | Exact F1 | Exact Precision | Exact Recall | Relaxed F1 | Full-test elapsed |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| SecureBERT-NER | {metric(secure, 'exact', 'f1'):.4f} | "
            f"{metric(secure, 'exact', 'precision'):.4f} | "
            f"{metric(secure, 'exact', 'recall'):.4f} | "
            f"{metric(secure, 'relaxed', 'f1'):.4f} | "
            f"{secure['elapsed_seconds']:.3f}s |"
        ),
        (
            f"| CyNER | {metric(cyner, 'exact', 'f1'):.4f} | "
            f"{metric(cyner, 'exact', 'precision'):.4f} | "
            f"{metric(cyner, 'exact', 'recall'):.4f} | "
            f"{metric(cyner, 'relaxed', 'f1'):.4f} | "
            f"{cyner['elapsed_seconds']:.3f}s |"
        ),
        "",
        "SecureBERT wins under the primary metric, strict entity F1, and the ranking",
        "is unchanged under relaxed boundary matching. That matters because relaxed",
        "matching is designed to absorb small tokenization or boundary differences;",
        "the conclusion is therefore not merely a boundary artifact.",
        "",
        "## 3. Discussion",
        "",
        "The absolute exact F1 values are modest because this is a cross-dataset,",
        "cross-taxonomy benchmark. We are not evaluating models fine-tuned on DNRTI;",
        "we are measuring transfer plus assignment-specified label projection. The",
        "model taxonomy matters. CyNER collapses the world into five categories,",
        "which makes it compact conceptually but unable to express DNRTI `Time` and",
        "`Area`. SecureBERT has finer APTNER-derived labels and covers `TIME` and",
        "`LOC`, which improves both recall and operational usefulness.",
        "",
        "The result also changes how the product should be framed. SecureBERT is the",
        "current evidence leader, but DNRTI `Purp` and `Features` remain uncovered",
        "by both models under the PDF mapping. If those classes are product-critical,",
        "they require a second-stage classifier, weak rules, or fine-tuning.",
        "",
        "## 4. Threats To Validity",
        "",
        "- The benchmark reconstructs DNRTI sentences using normalized single spaces.",
        "  This is appropriate for the provided token/tag files, but it may differ",
        "  from original report whitespace.",
        "- Reported energy is estimated from elapsed seconds and a fixed wattage",
        "  assumption. It is not a hardware-counter measurement.",
        "- The current run is CPU-only. The script supports `--device mps`; rerunning",
        "  on an M4 process with MPS exposed should improve latency, but not the",
        "  model-selection conclusion unless a backend-specific inference bug appears.",
        "- CyberNER-trained checkpoints should not be substituted into this comparison",
        "  because CyberNER includes DNRTI and would contaminate the benchmark.",
        "",
        "## 5. Conclusion",
        "",
        "SecureBERT-NER is the current evidence leader. It leads CyNER on exact F1,",
        "relaxed F1, recall, model footprint, memory peak, and elapsed time in this",
        "benchmark, while final product selection remains with the reviewer/product owner.",
    ]
    write_text(REPORT_DIR / "benchmark_summary.md", lines)


def write_dataset_audit(stats: dict[str, object]) -> None:
    counts = dict(stats["label_counts"])  # type: ignore[arg-type]
    lines = [
        "# DNRTI Dataset Audit",
        "",
        "## Summary",
        "",
        "The authoritative dataset source is the user-provided GitHub repository.",
        "The archive contains `train.txt`, `valid.txt`, and `test.txt` files in a",
        "CoNLL-like token/tag format. Evaluation uses the published `test.txt` split",
        "without resampling for the final claim.",
        "",
        "![DNRTI label distribution](figures/dnrti_label_distribution.svg)",
        "",
        "| Split | Sentences | Tokens | Labeled BIO tokens | Collapsed spans | Malformed lines |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| test | {stats['sentences']} | {stats['tokens']} | "
            f"{stats['labeled_tokens']} | {stats['spans']} | {stats['malformed_lines']} |"
        ),
        "",
        "## Label Counts",
        "",
        "| Label | Gold spans | Share |",
        "|---|---:|---:|",
    ]
    total = sum(int(value) for value in counts.values())
    for label, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"| {label} | {count} | {count / total:.1%} |")
    lines.extend(
        [
            "",
            "## Data Quality Notes",
            "",
            "- The source contains occasional tag-only `O` rows. The parser skips these",
            "  rows and counts them as malformed source lines rather than treating `O`",
            "  as a literal token.",
            "- Published DNRTI entity totals often count labeled BIO tokens. This report",
            "  distinguishes labeled BIO tokens from collapsed spans because strict NER",
            "  evaluation is span based.",
            "- The test split is imbalanced: `HackOrg`, `Tool`, and `SamFile` dominate",
            "  support, while `Way`, `Purp`, and `Features` have low but nontrivial",
            "  support. This is why per-label analysis is necessary.",
            "",
            "## Evaluation Implication",
            "",
            "A single micro-F1 number hides whether a model is useful for the product.",
            "For example, detecting `Time` and `Area` may matter for incident timelines",
            "and affected geography even if these labels are not the most frequent.",
        ]
    )
    write_text(REPORT_DIR / "dataset_audit.md", lines)


def write_scaling_report(rows: list[dict[str, object]]) -> None:
    lines = [
        "# Dataset Size Scaling Effects",
        "",
        "## Motivation",
        "",
        "Small NER subsets can produce unstable rankings because a few long reports or",
        "entity-rich sentences dominate the denominator. The benchmark therefore uses",
        "deterministic paired subsets: both models see the same 10-sentence subset,",
        "the same 100-sentence subset, and the full DNRTI test split.",
        "",
        "![Dataset size scaling](figures/dataset_size_scaling.svg)",
        "",
        "| Model | Subset | Samples | Exact F1 | Exact P | Exact R | Relaxed F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (str(item["model"]), int(item["samples"]))):
        exact = row["metrics"]["exact"]  # type: ignore[index]
        relaxed = row["metrics"]["relaxed"]  # type: ignore[index]
        lines.append(
            f"| {MODEL_LABELS[str(row['model'])]} | {row.get('subset_size')} | "
            f"{row['samples']} | {exact['f1']:.4f} | {exact['precision']:.4f} | "
            f"{exact['recall']:.4f} | {relaxed['f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Discussion",
            "",
            "The 10-sentence point is a smoke test, not a decision basis. It already",
            "suggests SecureBERT is stronger, but variance is high because only 31 gold",
            "spans are present. At 100 samples the ranking stabilizes, and on the full",
            "664-sentence test split SecureBERT maintains a large margin under both",
            "strict and relaxed scoring.",
            "",
            "The same ranking under exact and relaxed matching is important. If relaxed",
            "F1 reversed the decision, the benchmark would be diagnosing boundary",
            "calibration rather than semantic recognition. It does not: SecureBERT is",
            "better even after giving both models partial-boundary credit.",
        ]
    )
    write_text(REPORT_DIR / "dataset_size_scaling.md", lines)


def write_label_mapping() -> None:
    lines = [
        "# Assignment Label Mapping And Taxonomy Analysis",
        "",
        "## Mapping Extracted From The PDF",
        "",
        "The assignment requires comparing models with incompatible taxonomies. The",
        "benchmark therefore projects model outputs into DNRTI labels before scoring.",
        "",
        "| CyNER | DNRTI | SecureBERT-NER | Consequence |",
        "|---|---|---|---|",
        "| Organization | HackOrg, SecTeam, Idus, Org | APT, SECTEAM, IDTY | "
        "CyNER cannot distinguish actor, security team, identity, and organization. |",
        "| System | OffAct, Way | ACT, OS, TOOL | The broad `System` label creates "
        "ambiguity for behavior labels. |",
        "| Vulnerability | Exp | VULID, VULNAME | Both models can target "
        "exploit/vulnerability mentions. |",
        "| Malware | Tool | MAL | DNRTI `Tool` is malware/tooling in assignment terms. |",
        "| Indicator | SamFile | FILE | File indicators are comparable. |",
        "| Indicator | no DNRTI label | DOM, ENCR, IP, URL, MD5, PROT, EMAIL, "
        "SHA1, SHA2 | These SecureBERT IOCs are operationally useful but not "
        "credited by DNRTI. |",
        "| no CyNER label | Time | TIME | SecureBERT can score timeline entities; "
        "CyNER cannot. |",
        "| no CyNER label | Area | LOC | SecureBERT can score geography; CyNER cannot. |",
        "| no CyNER label | Purp, Features | no SecureBERT label | Neither model "
        "can directly cover these classes. |",
        "",
        "![Per-label exact F1](figures/per_label_exact_f1.svg)",
        "",
        "## Discussion",
        "",
        "This mapping is the central experimental design choice. It is not a clerical",
        "detail: it changes the decision. SecureBERT benefits from finer APTNER labels",
        "for `TIME`, `LOC`, `SECTEAM`, and `IDTY`, while CyNER's five-label ontology",
        "compresses several DNRTI concepts into `Organization` or `System`. That makes",
        "CyNER harder to use when downstream workflows require specific threat-actor,",
        "geography, and identity slots.",
        "",
        "The benchmark deliberately does not credit SecureBERT's IOC labels (`IP`,",
        "`URL`, hashes, domains, and email) unless the PDF maps them to a DNRTI",
        "label. Those predictions may be valuable in production, but including them",
        "as true positives would violate the assignment evaluation contract.",
    ]
    write_text(REPORT_DIR / "label_mapping.md", lines)


def write_literature_review() -> None:
    lines = [
        "# Literature Review And Leakage Analysis",
        "",
        "## Research Questions",
        "",
        "1. Were the assigned checkpoints trained on DNRTI?",
        "2. Do related corpora create leakage or overlap risk?",
        "3. How should leakage risk change interpretation of the benchmark?",
        "",
        "## Findings",
        "",
        f"- `{SECUREBERT_MODEL}` is documented as SecureBERT fine-tuned on APTNER,",
        "  not DNRTI. The base SecureBERT model was pretrained on broad cybersecurity",
        "  text, so raw public-report overlap remains possible, but supervised DNRTI",
        "  exposure was not found.",
        f"- `{CYNER_MODEL}` exposes CyNER's five-label taxonomy: `System`,",
        "  `Organization`, `Vulnerability`, `Malware`, and `Indicator`. Public CyNER",
        "  descriptions point to a separate Android-malware CTI corpus, not DNRTI.",
        "- CyberNER is not a valid replacement model for this benchmark unless DNRTI",
        "  is explicitly held out, because CyberNER harmonizes DNRTI together with",
        "  CyNER, APTNER, and Attacker.",
        "",
        "## Leakage Risk Table",
        "",
        "| Artifact | DNRTI supervised exposure | Risk level | Interpretation |",
        "|---|---|---|---|",
        "| SecureBERT-NER | No public evidence found | Medium | APTNER domain "
        "transfer; possible raw-report overlap. |",
        "| CyNER HF checkpoint | No public evidence found | Medium | Separate CTI "
        "corpus; broad taxonomy hurts direct DNRTI comparability. |",
        "| CyberNER-derived models | Yes, by construction | High | Contaminated "
        "unless DNRTI is held out. |",
        "",
        "## Discussion",
        "",
        "The correct reading of this experiment is cross-dataset transfer under a",
        "forced taxonomy projection. That is exactly the deployment setting: the chosen",
        "model must work on threat-intelligence text it was not explicitly fine-tuned",
        "for and must emit entities useful to an offline product. The benchmark should",
        "not be presented as an in-domain supervised DNRTI leaderboard.",
        "",
        "## Sources",
        "",
        f"- SecureBERT-NER Hugging Face: https://huggingface.co/{SECUREBERT_MODEL}",
        "- SecureBERT paper: https://arxiv.org/abs/2204.02685",
        "- CyNER paper: https://arxiv.org/abs/2204.05754",
        f"- CyNER Hugging Face checkpoint: https://huggingface.co/{CYNER_MODEL}",
        f"- DNRTI repository: {DNRTI_REPO}",
        "- CyberNER paper: https://arxiv.org/abs/2510.26499",
    ]
    write_text(REPORT_DIR / "literature_and_leakage.md", lines)


def write_operational_report(rows: list[dict[str, object]]) -> None:
    secure = row_for(rows, "securebert")
    cyner = row_for(rows, "cyner")
    lines = [
        "# Operational Metrics For On-Prem Deployment",
        "",
        "## Why Operations Matter",
        "",
        "The selected model must ship inside an offline Docker environment. Accuracy is",
        "necessary, but image size, RAM headroom, startup time, and energy cost also",
        "affect customer deployment feasibility.",
        "",
        "![Operational tradeoffs](figures/operational_tradeoffs.svg)",
        "",
        "| Model | Parameters | Cache bytes | Full-test elapsed | RSS peak MB | "
        "Estimated energy J |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| SecureBERT-NER | 124085800 | 996220887 | "
            f"{secure['elapsed_seconds']:.3f}s | {secure['rss_peak_mb']} | "
            f"{secure['estimated_energy_joules']:.3f} |"
        ),
        (
            f"| CyNER | 277461515 | 1124083139 | "
            f"{cyner['elapsed_seconds']:.3f}s | {cyner['rss_peak_mb']} | "
            f"{cyner['estimated_energy_joules']:.3f} |"
        ),
        "",
        "## Discussion",
        "",
        "SecureBERT is not just more accurate in this benchmark; it is also the easier",
        "model to ship. It has fewer parameters, a smaller Hugging Face cache, lower",
        "observed RSS peak, and lower full-test elapsed time. This matters for an",
        "offline Docker image because model cache size directly affects image size or",
        "mounted artifact size, and memory peak affects minimum customer hardware.",
        "",
        "Energy is reported as an estimate: elapsed seconds multiplied by an assumed",
        "18 W device profile. For a production-grade measurement on macOS, rerun the",
        "benchmark while sampling `powermetrics` and record package power directly.",
        "",
        "## MPS Note",
        "",
        "The script supports `--device auto`, `--device mps`, and `--device cpu`. This",
        "run used CPU because the process reported `mps_available=false`. On an Apple",
        "M4 workstation, rerun with MPS exposed to get deployment-relevant latency.",
        "The model-selection logic should still be based on accuracy first; MPS mainly",
        "changes the operational envelope.",
    ]
    write_text(REPORT_DIR / "operational_metrics.md", lines)


def write_final_selection(rows: list[dict[str, object]]) -> None:
    secure = row_for(rows, "securebert")
    cyner = row_for(rows, "cyner")
    lines = [
        "# Evidence Leader Summary",
        "",
        "Evidence leader: **SecureBERT-NER**.",
        "",
        "Evidence basis: full test split strict entity F1.",
        "Deployment selection is intentionally left to the reviewer/product owner.",
        "",
        "## Headline Evidence",
        "",
        f"- Full-test strict entity F1: `{metric(secure, 'exact', 'f1'):.4f}`.",
        (
            f"- Full-test strict precision / recall: "
            f"`{metric(secure, 'exact', 'precision'):.4f}` / "
            f"`{metric(secure, 'exact', 'recall'):.4f}`."
        ),
        f"- Full-test relaxed F1: `{metric(secure, 'relaxed', 'f1'):.4f}`.",
        f"- Inference elapsed on full test split: `{secure['elapsed_seconds']:.3f}` seconds.",
        "",
        "## Full-Test Comparison",
        "",
        "| Model | Strict F1 | Strict P | Strict R | Relaxed F1 | Elapsed s | RSS peak MB |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| SecureBERT-NER | {metric(secure, 'exact', 'f1'):.4f} | "
            f"{metric(secure, 'exact', 'precision'):.4f} | "
            f"{metric(secure, 'exact', 'recall'):.4f} | "
            f"{metric(secure, 'relaxed', 'f1'):.4f} | "
            f"{secure['elapsed_seconds']:.3f} | {secure['rss_peak_mb']} |"
        ),
        (
            f"| CyNER | {metric(cyner, 'exact', 'f1'):.4f} | "
            f"{metric(cyner, 'exact', 'precision'):.4f} | "
            f"{metric(cyner, 'exact', 'recall'):.4f} | "
            f"{metric(cyner, 'relaxed', 'f1'):.4f} | "
            f"{cyner['elapsed_seconds']:.3f} | {cyner['rss_peak_mb']} |"
        ),
        "",
        "## Operational Fit",
        "",
        "| Model | Parameters | Cached bytes |",
        "|---|---:|---:|",
        "| SecureBERT-NER | 124085800 | 996220887 |",
        "| CyNER | 277461515 | 1124083139 |",
        "",
        "## Caveats",
        "",
        "- Neither assigned model covers DNRTI `Purp` or `Features` under the PDF",
        "  mapping.",
        "- Energy is estimated, not directly measured.",
        "- MPS latency should be rerun on the target M4 host.",
        "- The benchmark treats CyberNER-derived checkpoints as contaminated because",
        "  CyberNER includes DNRTI by construction.",
    ]
    write_text(REPORT_DIR / "final_selection.md", lines)


def write_project_page(rows: list[dict[str, object]], stats: dict[str, object]) -> None:
    secure = row_for(rows, "securebert")
    cyner = row_for(rows, "cyner")
    lines = [
        "# ThreatExtract Fork 1: DNRTI NER Benchmark",
        "",
        "A CVPR-style project page for model selection under label-space mismatch,",
        "data-leakage risk, and offline deployment constraints.",
        "",
        "## Abstract",
        "",
        "We compare SecureBERT-NER and CyNER on DNRTI, a cybersecurity NER dataset",
        "whose taxonomy differs from both model taxonomies. The evaluation maps",
        "model outputs to DNRTI labels using the assignment PDF, then reports strict",
        "span F1, relaxed overlap F1, per-label behavior, subset-size stability, and",
        "offline operational metrics. SecureBERT-NER is the current evidence leader;",
        "deployment selection is intentionally left to the reviewer/product owner.",
        "",
        "![Pipeline](figures/evaluation_pipeline.svg)",
        "",
        "## Key Result",
        "",
        "![Model comparison](figures/model_comparison_metrics.svg)",
        "",
        "| Model | Strict F1 | Relaxed F1 | Strict recall | Elapsed s | RSS peak MB |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| SecureBERT-NER | {metric(secure, 'exact', 'f1'):.4f} | "
            f"{metric(secure, 'relaxed', 'f1'):.4f} | "
            f"{metric(secure, 'exact', 'recall'):.4f} | "
            f"{secure['elapsed_seconds']:.3f} | {secure['rss_peak_mb']} |"
        ),
        (
            f"| CyNER | {metric(cyner, 'exact', 'f1'):.4f} | "
            f"{metric(cyner, 'relaxed', 'f1'):.4f} | "
            f"{metric(cyner, 'exact', 'recall'):.4f} | "
            f"{cyner['elapsed_seconds']:.3f} | {cyner['rss_peak_mb']} |"
        ),
        "",
        "## Figure Gallery",
        "",
        "![Dataset labels](figures/dnrti_label_distribution.svg)",
        "",
        "![Scaling](figures/dataset_size_scaling.svg)",
        "",
        "![Per-label F1](figures/per_label_exact_f1.svg)",
        "",
        "![Operational tradeoffs](figures/operational_tradeoffs.svg)",
        "",
        "## Paper-Style Sections",
        "",
        "- [Benchmark summary](benchmark_summary.md)",
        "- [Dataset audit](dataset_audit.md)",
        "- [Label mapping](label_mapping.md)",
        "- [Literature and leakage](literature_and_leakage.md)",
        "- [Dataset-size scaling](dataset_size_scaling.md)",
        "- [Operational metrics](operational_metrics.md)",
        "- [Evidence leader summary](final_selection.md)",
        "",
        "## Reproduction",
        "",
        "```bash",
        "make reproduce",
        "```",
        "",
        "## Dataset Snapshot",
        "",
        f"DNRTI test split: `{stats['sentences']}` sentences, `{stats['tokens']}` tokens,",
        f"`{stats['spans']}` collapsed entity spans.",
    ]
    write_text(REPORT_DIR / "project_page.md", lines)


def generate_markdown(rows: list[dict[str, object]], stats: dict[str, object]) -> None:
    write_benchmark_summary(rows, stats)
    write_dataset_audit(stats)
    write_scaling_report(rows)
    write_label_mapping()
    write_literature_review()
    write_operational_report(rows)
    write_final_selection(rows)
    write_project_page(rows, stats)


def main() -> int:
    rows = read_jsonl(REPORT_DIR / "benchmark_results.jsonl")
    stats = dataset_stats()
    generate_figures(rows, stats)
    generate_markdown(rows, stats)
    print(f"Generated enriched Fork 1 reports in {REPORT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
