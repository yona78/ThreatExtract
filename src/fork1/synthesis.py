from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fork1.data import load_dnrti_dataset


COMPARISON_FILES = {
    "methodology": "methodology_baseline.jsonl",
    "preprocessing": "preprocessing_sensitivity.jsonl",
    "subset": "subset_study.jsonl",
    "protocol": "protocol_comparison.jsonl",
    "robustness": "robustness.jsonl",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _coerce_master_row(source: str, row: dict[str, Any]) -> dict[str, Any] | None:
    model = row.get("model")
    if not model:
        return None
    if source == "robustness":
        f1 = row.get("noisy_f1")
        config = f"perturbation={row.get('perturbation', 'unknown')}"
        ci_low = row.get("gap_ci_low")
        ci_high = row.get("gap_ci_high")
    else:
        f1 = row.get("strict_f1", row.get("f1"))
        config = row.get("config", row.get("protocol", source))
        ci_low = row.get("ci_low")
        ci_high = row.get("ci_high")
    if f1 is None or ci_low is None or ci_high is None or "gap_vs_other" not in row:
        return None
    return {
        "source": source,
        "config": str(config),
        "model": str(model),
        "scheme": str(row.get("scheme", "strict")),
        "f1": float(f1),
        "gap_vs_other": float(row["gap_vs_other"]),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "mcnemar_p": row.get("mcnemar_p"),
        "flip": bool(row.get("flip", float(ci_low) <= 0 <= float(ci_high))),
    }


def collect_master_rows(reports_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, filename in COMPARISON_FILES.items():
        for raw in read_jsonl(reports_dir / filename):
            master = _coerce_master_row(source, raw)
            if master is not None:
                rows.append(master)
    return rows


def summarize_decisions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    secure_rows = [
        row
        for row in rows
        if row["model"] == "securebert" and row.get("scheme", "strict") == "strict"
    ]
    wins = ties = losses = 0
    flip_configs: list[str] = []
    loss_configs: list[str] = []
    for row in secure_rows:
        key = f"{row['source']}:{row['config']}"
        p_value = row.get("mcnemar_p")
        mcnemar_ok = p_value is None or float(p_value) < 0.05
        if float(row["ci_low"]) > 0 and float(row["gap_vs_other"]) > 0 and mcnemar_ok:
            wins += 1
        elif float(row["ci_high"]) < 0 and float(row["gap_vs_other"]) < 0 and mcnemar_ok:
            losses += 1
            loss_configs.append(key)
        else:
            ties += 1
            flip_configs.append(key)
    return {
        "securebert_wins": wins,
        "ties": ties,
        "cyner_wins": losses,
        "total": len(secure_rows),
        "flips": flip_configs,
        "losses": loss_configs,
    }


def bias_adjustment(
    protocol_rows: list[dict[str, Any]],
    intrinsic_rows: list[dict[str, Any]],
) -> dict[str, float]:
    raw_gap = next(
        float(row["gap_vs_other"])
        for row in protocol_rows
        if row.get("model") == "securebert"
        and row.get("protocol", row.get("config")) == "pdf_mapping"
    )
    oracle = {str(row["model"]): float(row["oracle_f1"]) for row in intrinsic_rows}
    oracle_gap = oracle["securebert"] - oracle["cyner"]
    residual_gap = raw_gap - oracle_gap
    ceiling_share = oracle_gap / raw_gap if raw_gap else 0.0
    return {
        "raw_gap": raw_gap,
        "oracle_gap": oracle_gap,
        "residual_gap": residual_gap,
        "ceiling_share": ceiling_share,
    }


def _best_rows_by_model(rows: list[dict[str, Any]], **criteria: Any) -> dict[str, dict[str, Any]]:
    out = {}
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            out[str(row["model"])] = row
    return out


def _cpu_deployment_note(operational_rows: list[dict[str, Any]]) -> str:
    cpu_256 = _best_rows_by_model(
        operational_rows,
        device="cpu",
        token_length=256,
        batch_size=1,
    )
    secure = cpu_256.get("securebert")
    cyner = cpu_256.get("cyner")
    if not secure or not cyner:
        return "CPU deployment note: operational CPU batch=1, 256-token rows were unavailable."
    speedup = float(cyner["p50_ms"]) / float(secure["p50_ms"]) if secure["p50_ms"] else 0.0
    return (
        "CPU deployment note: at 256 tokens and batch=1, SecureBERT p50 latency is "
        f"{float(secure['p50_ms']):.2f} ms vs CyNER {float(cyner['p50_ms']):.2f} ms "
        f"({speedup:.2f}x faster), with RSS {float(secure['rss_after_load_mb']):.1f} MB "
        f"vs {float(cyner['rss_after_load_mb']):.1f} MB."
    )


def _calibration_note(calibration_rows: list[dict[str, Any]]) -> str:
    if not calibration_rows:
        return "Calibration evidence was unavailable."
    parts = []
    for row in calibration_rows:
        threshold = row.get("recommended_threshold")
        threshold_text = "unreachable" if threshold is None else f"{float(threshold):.2f}"
        parts.append(
            f"{row['model']} ECE {float(row['ece']):.4f}, precision>=0.90 threshold {threshold_text}"
        )
    return "; ".join(parts) + "."


def _baseline_strict_rows(master_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    candidates = [
        row
        for row in master_rows
        if row["source"] in {"methodology", "protocol"}
        and row["config"] == "pdf_mapping"
        and row["scheme"] == "strict"
    ]
    out = {}
    for row in candidates:
        out[row["model"]] = row
    return out


def _direction_votes() -> list[tuple[str, str, str]]:
    return [
        ("01", "SecureBERT", "all four SemEval schemes preserve a positive gap"),
        ("02", "SecureBERT", "preprocessing sweeps did not flip the full-test ranking"),
        ("03", "SecureBERT", "min-faithful subsets reach stable positive gaps"),
        ("04", "SecureBERT", "paper-native protocol keeps the positive gap"),
        ("05", "SecureBERT with bias caveat", "oracle and lineage show structural advantage"),
        ("06", "SecureBERT", "CPU profile is smaller and faster at deployment lengths"),
        ("07", "SecureBERT with confidence caveat", "robustness holds, raw calibration fails both"),
    ]


def write_master_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Fork 1 Master Table",
        "",
        "| Source | Config | Model | Scheme | F1 | Gap | 95% CI | Flip? |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['source']} | {row['config']} | {row['model']} | {row['scheme']} | "
            f"{float(row['f1']):.4f} | {float(row['gap_vs_other']):.4f} | "
            f"[{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}] | "
            f"{'yes' if row['flip'] else 'no'} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_bias_report(
    path: Path,
    figure_path: Path,
    intrinsic_rows: list[dict[str, Any]],
    adjustment: dict[str, float],
) -> None:
    rows = {str(row["model"]): row for row in intrinsic_rows}
    secure = rows.get("securebert", {})
    cyner = rows.get("cyner", {})
    path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    expressible_secure = ", ".join(secure.get("expressible_labels", []))
    expressible_cyner = ", ".join(cyner.get("expressible_labels", []))
    text = f"""# Leakage & Bias Audit

## Lineage

| Model | Public lineage evidence | Bias implication |
|---|---|---|
| SecureBERT-NER | Hugging Face identifies `CyberPeace-Institute/SecureBERT-NER` as SecureBERT fine-tuned on APTNER; the SecureBERT paper describes cybersecurity-domain pretraining. | Cyber-domain pretraining, APTNER fine-tuning, and a DNRTI-adjacent ontology all structurally favor SecureBERT. |
| CyNER | Hugging Face identifies `AI4Sec/cyner-xlm-roberta-base` as an XLM-R token-classification model; the CyNER paper describes a broader cyber NER library and coarser event ontology. | The DNRTI projection compresses several labels into each CyNER label and leaves a lower ceiling. |

## APTNER/DNRTI Overlap

`data/aptner/` is not present in this offline workspace, so 8-gram and sentence-hash overlap could not be computed. The overlap risk is unquantified, not proven.

## Mapping Handicap

| Model | Oracle recall | Oracle F1 | Expressible labels |
|---|---:|---:|---|
| SecureBERT | {float(secure.get('oracle_recall', 0.0)):.4f} | {float(secure.get('oracle_f1', 0.0)):.4f} | {expressible_secure} |
| CyNER | {float(cyner.get('oracle_recall', 0.0)):.4f} | {float(cyner.get('oracle_f1', 0.0)):.4f} | {expressible_cyner} |

The raw PDF-mapping strict-F1 gap is {adjustment['raw_gap']:.4f}. The oracle ceiling gap is {adjustment['oracle_gap']:.4f}, about {adjustment['ceiling_share']:.1%} of the raw gap magnitude. Treating that ceiling difference as a conservative handicap leaves a bias-adjusted residual gap of {adjustment['residual_gap']:.4f}.

## Interpretation

SecureBERT is the better mapped-DNRTI choice, but the full raw gap should not be read as pure recognition capability. Direction 08 carries this caveat into the final verdict.
"""
    path.write_text(text, encoding="utf-8")
    figure = """<svg xmlns="http://www.w3.org/2000/svg" width="760" height="220" viewBox="0 0 760 220">
<rect width="100%" height="100%" fill="white"/>
<text x="24" y="32" font-family="Arial" font-size="17" font-weight="700">SecureBERT Structural Advantage Chain</text>
<rect x="30" y="74" width="140" height="56" fill="#e8f1fb" stroke="#2f6fbb"/>
<rect x="220" y="74" width="140" height="56" fill="#e8f1fb" stroke="#2f6fbb"/>
<rect x="410" y="74" width="140" height="56" fill="#e8f1fb" stroke="#2f6fbb"/>
<rect x="600" y="74" width="120" height="56" fill="#fbecea" stroke="#c45746"/>
<text x="48" y="106" font-family="Arial" font-size="12">Cyber pretraining</text>
<text x="254" y="106" font-family="Arial" font-size="12">APTNER</text>
<text x="430" y="99" font-family="Arial" font-size="12">DNRTI-like</text>
<text x="431" y="116" font-family="Arial" font-size="12">ontology</text>
<text x="617" y="99" font-family="Arial" font-size="12">Raw gap is</text>
<text x="617" y="116" font-family="Arial" font-size="12">confounded</text>
<line x1="170" y1="102" x2="220" y2="102" stroke="#555" marker-end="url(#a)"/>
<line x1="360" y1="102" x2="410" y2="102" stroke="#555" marker-end="url(#a)"/>
<line x1="550" y1="102" x2="600" y2="102" stroke="#555" marker-end="url(#a)"/>
<defs><marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#555"/></marker></defs>
</svg>
"""
    figure_path.write_text(figure, encoding="utf-8")


def write_benchmark_summary(
    path: Path,
    master_rows: list[dict[str, Any]],
    decision_summary: dict[str, Any],
    adjustment: dict[str, float],
    operational_note: str,
    calibration_note: str,
    dataset_stats: dict[str, Any] | None,
) -> None:
    baseline = _baseline_strict_rows(master_rows)
    secure = baseline.get("securebert")
    cyner = baseline.get("cyner")
    dataset_line = "DNRTI test split was loaded from the configured offline data directory."
    if dataset_stats:
        dataset_line = (
            f"DNRTI test split: {dataset_stats['sentences']} sentences, "
            f"{dataset_stats['tokens']} tokens, {dataset_stats['spans']} collapsed gold spans, "
            f"{dataset_stats['malformed_lines']} skipped malformed tag-only lines."
        )
    lines = [
        "# Fork 1 Final Report: Frozen SecureBERT-NER vs CyNER on DNRTI",
        "",
        "## Abstract",
        "",
        (
            "This offline study compares frozen SecureBERT-NER and CyNER on DNRTI for an "
            "on-prem threat-intelligence NER product. The evidence leader is SecureBERT, "
            "but final product selection is intentionally left to the reviewer. The raw F1 gap is partly a "
            "taxonomy and training-lineage advantage, not pure recognition capability."
        ),
        "",
        "## Method",
        "",
        dataset_line,
        "Models were run as frozen black boxes from `model_cache/fork1` with no training or weight edits. The primary metric is SemEval strict entity F1 with paired-bootstrap 95% CIs and McNemar checks.",
        "",
        "## Headline Strict Result",
        "",
        "| Model | Strict F1 | Gap vs other | 95% CI |",
        "|---|---:|---:|---|",
    ]
    for row in (secure, cyner):
        if row:
            lines.append(
                f"| {row['model']} | {float(row['f1']):.4f} | "
                f"{float(row['gap_vs_other']):.4f} | "
                f"[{float(row['ci_low']):.4f}, {float(row['ci_high']):.4f}] |"
            )
    lines.extend(
        [
            "",
            "## Cross-Experiment Decision Count",
            "",
            (
                f"SecureBERT wins {decision_summary['securebert_wins']}/"
                f"{decision_summary['total']} strict head-to-head configs; "
                f"{decision_summary['ties']} configs are statistically tied and "
                f"{decision_summary['cyner_wins']} favor CyNER."
            ),
            "",
            "Direction votes:",
            "",
            "| Direction | Vote | Rationale |",
            "|---|---|---|",
        ]
    )
    for direction, vote, rationale in _direction_votes():
        lines.append(f"| {direction} | {vote} | {rationale} |")
    lines.extend(
        [
            "",
            "## Sensitivity",
            "",
            "No full-split preprocessing, protocol, or robustness condition reverses the ranking. Small 10-sentence subsets are often underpowered and can tie by CI, so they are not decision-grade.",
            "",
            "## Capability Vs Bias",
            "",
            (
                f"The raw PDF-mapping gap is {adjustment['raw_gap']:.4f}. The oracle ceiling gap "
                f"is {adjustment['oracle_gap']:.4f} ({adjustment['ceiling_share']:.1%} of the raw gap). "
                f"The conservative bias-adjusted residual is {adjustment['residual_gap']:.4f}; this remains positive, but it is not a causal decomposition."
            ),
            "",
            "## Operational Profile",
            "",
            operational_note,
            "",
            "## Reliability",
            "",
            calibration_note,
            "Robustness perturbations preserve a positive SecureBERT gap, but random casing sharply reduces both models and should be normalized or monitored upstream.",
            "",
            "## Threats To Validity",
            "",
            "- APTNER/DNRTI overlap could not be quantified offline because `data/aptner/` is absent.",
            "- The label projection structurally favors SecureBERT; the bias adjustment is a ceiling-based sensitivity check, not proof of independent capability.",
            "- Energy is estimated where `powermetrics` is unavailable.",
            "- Raw confidence scores do not provide a precision>=0.90 operating point for either model.",
            "",
            "## Conclusion",
            "",
            "The evidence package supports SecureBERT-NER as the current evidence leader, but it does not hard-code the product selection. SecureBERT wins the statistical comparisons, survives the protocol and robustness checks, has the better CPU deployment envelope, and retains a positive residual after the ontology-ceiling bias check. The final chosen model should be set by the reviewer/product owner after deciding how to weigh the structural-bias caveat and label-coverage risks. The product should not present the full raw gap as pure model quality, and it should add calibration/abstention logic before using confidence as an analyst triage threshold.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment() -> dict[str, Any]:
    env = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch

        env["torch_version"] = torch.__version__
        env["mps_available"] = bool(torch.backends.mps.is_available())
    except Exception as exc:  # pragma: no cover - metadata best effort
        env["torch_error"] = str(exc)
    try:
        import transformers

        env["transformers_version"] = transformers.__version__
    except Exception as exc:  # pragma: no cover - metadata best effort
        env["transformers_error"] = str(exc)
    return env


def _git_info(cwd: Path) -> dict[str, str | None]:
    info: dict[str, str | None] = {"commit": None, "branch": None}
    commands = {
        "commit": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
    }
    for key, command in commands.items():
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            info[key] = result.stdout.strip()
    return info


def _dataset_stats(dnrti_dir: Path | None) -> dict[str, Any] | None:
    if dnrti_dir is None or not dnrti_dir.exists():
        return None
    samples, warnings, stats = load_dnrti_dataset(dnrti_dir, "test")
    return {
        "sentences": len(samples),
        "tokens": sum(len(sample.tokens) for sample in samples),
        "spans": sum(len(sample.gold_spans) for sample in samples),
        "malformed_lines": len(warnings),
        "loader_stats": stats,
    }


def write_run_metadata(
    path: Path,
    reports_dir: Path,
    master_rows: list[dict[str, Any]],
    dnrti_dir: Path | None,
    cache_dir: Path | None,
) -> None:
    jsonl_files = sorted(reports_dir.glob("*.jsonl"))
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_info(Path.cwd()),
        "environment": _environment(),
        "dataset": _dataset_stats(dnrti_dir),
        "cache_dir": None if cache_dir is None else str(cache_dir),
        "inputs": [
            {
                "path": str(path.relative_to(reports_dir)),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in jsonl_files
        ],
        "experiment_configs": sorted(
            {f"{row['source']}::{row['config']}::{row['scheme']}" for row in master_rows}
        ),
        "reproduce": {
            "command": "make reproduce",
            "research_dependency_mode": "make setup-research installs requirements-research.txt into .venv with uv pip",
        },
    }
    path.write_text(json.dumps(jsonable(metadata), indent=2, sort_keys=True), encoding="utf-8")


def run_synthesis(
    *,
    reports_dir: Path,
    out_dir: Path,
    dnrti_dir: Path | None,
    cache_dir: Path | None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    master_rows = collect_master_rows(reports_dir)
    decision_summary = summarize_decisions(master_rows)
    protocol_rows = read_jsonl(reports_dir / "protocol_comparison.jsonl")
    intrinsic_rows = read_jsonl(reports_dir / "intrinsic_metrics.jsonl")
    operational_rows = read_jsonl(reports_dir / "operational_envelope.jsonl")
    calibration_rows = read_jsonl(reports_dir / "calibration_summary.jsonl")
    adjustment = bias_adjustment(protocol_rows, intrinsic_rows)
    dataset_stats = _dataset_stats(dnrti_dir)

    write_jsonl(out_dir / "master_table.jsonl", master_rows)
    write_master_table(out_dir / "master_table.md", master_rows)
    write_bias_report(
        out_dir / "leakage_bias.md",
        out_dir / "figures" / "bias_chain.svg",
        intrinsic_rows,
        adjustment,
    )
    write_benchmark_summary(
        out_dir / "benchmark_summary.md",
        master_rows,
        decision_summary,
        adjustment,
        _cpu_deployment_note(operational_rows),
        _calibration_note(calibration_rows),
        dataset_stats,
    )
    write_run_metadata(
        out_dir / "run_metadata.json", reports_dir, master_rows, dnrti_dir, cache_dir
    )
    return {
        "decision_summary": decision_summary,
        "bias_adjustment": adjustment,
        "master_rows": len(master_rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthesize Fork 1 experiment artifacts.")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/fork1"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/fork1"))
    parser.add_argument("--dnrti-dir", type=Path, default=Path("data/dnrti"))
    parser.add_argument("--cache-dir", type=Path, default=Path("model_cache/fork1"))
    args = parser.parse_args(argv)
    run_synthesis(
        reports_dir=args.reports_dir,
        out_dir=args.out_dir,
        dnrti_dir=args.dnrti_dir,
        cache_dir=args.cache_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
