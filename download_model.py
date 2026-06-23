#!/usr/bin/env python3
"""Pre-build helper: download a Hugging Face NER model into a local cache.

Run this ONCE, online, *before* building the Docker image. The Dockerfile then
COPYs the resulting ``./model_cache/`` directory into the image, so the
container runs fully offline — it never downloads anything at build or runtime.

Usage::

    python download_model.py                        # default model
    python download_model.py --model <hf_repo_id>   # any token-classification model
    MODEL_NAME=<hf_repo_id> python download_model.py

The script verifies the snapshot and prints the ``MODEL_PATH`` to use at runtime.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# NOTE: ``huggingface_hub`` is imported lazily inside ``download()`` so the pure
# helper ``build_ignore_patterns`` can be imported and unit-tested without it —
# which keeps CI's test collection free of the runtime-only dependency.

DEFAULT_MODEL = "CyberPeace-Institute/SecureBERT-NER"

# Files we never need to *run inference*, so skipping them keeps the download
# (and the Docker image that bakes it in) lean while staying model-agnostic for
# any PyTorch token-classification model:
#   * non-PyTorch weight formats (TF / Flax / ONNX / TFLite); and
#   * training-checkpoint artifacts (optimizer / scheduler / RNG / trainer state)
#     that some repos — including SecureBERT-NER — ship next to the weights.
_BASE_IGNORE_PATTERNS = [
    # Non-PyTorch weight formats.
    "*.h5",
    "*.msgpack",
    "*.onnx",
    "tf_model.*",
    "flax_model.*",
    "*.tflite",
    # Training-checkpoint artifacts (never used at inference time).
    "optimizer.pt",
    "scheduler.pt",
    "rng_state*.pth",
    "trainer_state.json",
    "training_args.bin",
]


def build_ignore_patterns(repo_files: list[str]) -> list[str]:
    """Return ignore patterns for ``snapshot_download`` based on what the repo contains.

    Always excludes non-PyTorch weight formats (tf, flax, onnx, …).  When the
    repo ships at least one ``*.safetensors`` file, also excludes the redundant
    ``pytorch_model.bin`` / ``pytorch_model-*.bin`` shards so the download is
    roughly halved without breaking models that ship *only* ``.bin`` weights.
    """
    patterns = list(_BASE_IGNORE_PATTERNS)
    has_safetensors = any(f.endswith(".safetensors") for f in repo_files)
    if has_safetensors:
        patterns.append("pytorch_model.bin")
        patterns.append("pytorch_model-*.bin")
    return patterns


def sanitize(repo_id: str) -> str:
    """Turn a repo id like ``org/name`` into a filesystem-safe folder name."""
    return repo_id.replace("/", "__")


def verify_snapshot(path: Path) -> dict:
    """Ensure the download looks like a usable token-classification model.

    Returns the model's ``id2label`` map so the caller can report the classes.
    """
    config_file = path / "config.json"
    if not config_file.is_file():
        raise FileNotFoundError(f"config.json not found in {path}")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    id2label = config.get("id2label")
    if not id2label:
        raise ValueError(
            f"{config_file} has no 'id2label' map — " "is this really a token-classification model?"
        )
    return id2label


def download(model: str, out_dir: Path) -> Path:
    """Download ``model`` into ``out_dir/<sanitized_id>`` and verify it."""
    from huggingface_hub import list_repo_files, snapshot_download

    target = out_dir / sanitize(model)
    target.mkdir(parents=True, exist_ok=True)
    print(f"Downloading '{model}' -> {target} ...")
    repo_files = list(list_repo_files(model))
    ignore_patterns = build_ignore_patterns(repo_files)
    snapshot_download(
        repo_id=model,
        local_dir=str(target),
        ignore_patterns=ignore_patterns,
    )
    id2label = verify_snapshot(target)
    labels = ", ".join(sorted({str(v) for v in id2label.values()}))
    print(f"Verified OK. {len(id2label)} labels: {labels}")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face NER model for offline use."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_NAME", DEFAULT_MODEL),
        help=f"Hugging Face repo id (default: {DEFAULT_MODEL}, or $MODEL_NAME).",
    )
    parser.add_argument(
        "--out",
        default="./model_cache",
        help="Output cache directory (default: ./model_cache).",
    )
    args = parser.parse_args(argv)

    try:
        target = download(args.model, Path(args.out))
    except Exception as exc:  # surface a clean message instead of a traceback
        print(f"ERROR: failed to download '{args.model}': {exc}", file=sys.stderr)
        return 1

    rel = target.as_posix()
    print("\nNext steps:")
    print(f"  Local run:  MODEL_PATH=./{rel} streamlit run app.py")
    print(f"  Docker:     set MODEL_PATH=/app/{rel}  (see docker-compose.yml)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
