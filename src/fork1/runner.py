from __future__ import annotations

import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Protocol

from fork1.data import Span
from fork1.mapping import strip_bio

SECUREBERT_MODEL = "CyberPeace-Institute/SecureBERT-NER"
CYNER_MODEL = "AI4Sec/cyner-xlm-roberta-base"
MODEL_ALIASES = {"securebert": SECUREBERT_MODEL, "cyner": CYNER_MODEL}


class ModelRunner(Protocol):
    name: str

    def load(self) -> dict[str, object]: ...

    def predict(self, text: str) -> list[Span]: ...


def resolve_device(requested: str, allow_fallback: bool = False):
    try:
        import torch
    except ImportError as exc:
        if requested in {"auto", "cpu"}:
            return "cpu"
        raise RuntimeError("PyTorch is required for non-CPU device resolution") from exc

    if requested == "cpu":
        return torch.device("cpu")
    if requested in {"auto", "mps"}:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if requested == "mps" and not allow_fallback:
            raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is false")
    return torch.device("cpu")


def get_process_rss_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        try:
            output = subprocess.check_output(
                ["ps", "-o", "rss=", "-p", str(os.getpid())],
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                return peak_rss / (1024 * 1024)
            return peak_rss / 1024
        return int(output) / 1024 if output else None
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_mps_memory_mb() -> dict[str, float] | None:
    try:
        import torch
    except ImportError:
        return None
    if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
        return None
    return {
        "mps_current_allocated_mb": torch.mps.current_allocated_memory() / (1024 * 1024),
        "mps_driver_allocated_mb": torch.mps.driver_allocated_memory() / (1024 * 1024),
    }


def hf_cache_model_dir(cache_dir: Path | None, model_id: str) -> Path | None:
    if cache_dir is None:
        return None
    return cache_dir / f"models--{model_id.replace('/', '--')}"


def directory_size_bytes(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    if path.is_file():
        return path.lstat().st_size
    return sum(child.lstat().st_size for child in path.rglob("*") if child.is_file())


class HfTokenClassificationRunner:
    def __init__(
        self,
        name: str,
        model_id: str,
        device_request: str,
        allow_device_fallback: bool,
        offline: bool,
        cache_dir: Path | None,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self.device_request = device_request
        self.allow_device_fallback = allow_device_fallback
        self.offline = offline
        self.cache_dir = cache_dir
        self.device = None
        self.pipe = None
        self.model = None
        self.tokenizer = None

    def load(self) -> dict[str, object]:
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

        self.device = resolve_device(self.device_request, self.allow_device_fallback)
        before_rss = get_process_rss_mb()
        start = time.perf_counter()
        cache = str(self.cache_dir) if self.cache_dir else None
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=cache,
            local_files_only=self.offline,
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_id,
            cache_dir=cache,
            local_files_only=self.offline,
        )
        self.model.to(self.device)
        self.model.eval()
        pipeline_device = self.device if str(self.device) != "cpu" else -1
        # Word-level aggregation. "first" assigns each whole word the label of its
        # first sub-token, so multi-subword entities (e.g. "StoneDrill", "CrowdStrike")
        # are emitted as a single span instead of being fragmented into pieces like
        # "Stone"/"Crow". "simple" only merges *consecutive same-label* sub-tokens and
        # therefore splits words whenever sub-token predictions disagree, which crushed
        # strict precision/recall and inflated spurious-fragment false positives.
        self.pipe = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            aggregation_strategy="first",
            device=pipeline_device,
        )
        elapsed = time.perf_counter() - start
        after_rss = get_process_rss_mb()
        labels = sorted(
            {strip_bio(label) for label in self.model.config.id2label.values() if label != "O"}
        )
        return {
            "model_id": self.model_id,
            "device": str(self.device),
            "load_seconds": elapsed,
            "rss_before_load_mb": before_rss,
            "rss_after_load_mb": after_rss,
            "labels": labels,
            "parameter_count": sum(param.numel() for param in self.model.parameters()),
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "cache_model_dir": str(hf_cache_model_dir(self.cache_dir, self.model_id)),
        }

    def predict(self, text: str, max_length: int | None = None) -> list[Span]:
        if self.pipe is None:
            raise RuntimeError(f"{self.name} runner is not loaded")
        if max_length and self.tokenizer is not None:
            previous_max_length = self.tokenizer.model_max_length
            self.tokenizer.model_max_length = max_length
            try:
                raw_entities = self.pipe(text)
            finally:
                self.tokenizer.model_max_length = previous_max_length
        else:
            raw_entities = self.pipe(text)
        spans: list[Span] = []
        for entity in raw_entities:
            label = entity.get("entity_group") or entity.get("entity") or ""
            start = entity.get("start")
            end = entity.get("end")
            if start is None or end is None:
                continue
            clean = strip_bio(str(label))
            spans.append(
                Span(
                    label=clean,
                    start=int(start),
                    end=int(end),
                    text=text[int(start) : int(end)],
                    score=float(entity["score"]) if "score" in entity else None,
                    source=self.name,
                )
            )
        return spans


def build_runner(name: str, config) -> HfTokenClassificationRunner:
    if name not in MODEL_ALIASES:
        raise ValueError(f"unknown model alias {name!r}; choose from {sorted(MODEL_ALIASES)}")
    return HfTokenClassificationRunner(
        name=name,
        model_id=MODEL_ALIASES[name],
        device_request=config.device,
        allow_device_fallback=config.allow_device_fallback,
        offline=config.offline,
        cache_dir=config.cache_dir,
    )
