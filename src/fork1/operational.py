from __future__ import annotations

import random
import subprocess
import time
from statistics import mean


DEFAULT_TOKEN_POOL = (
    "threat",
    "actor",
    "uses",
    "powershell",
    "mimikatz",
    "ransomware",
    "CVE-2021-44228",
    "command",
    "control",
    "hxxp",
    "payload",
    "persistence",
)


def make_workload(
    token_lengths: list[int],
    n_per_length: int,
    seed: int,
    token_pool: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    rng = random.Random(seed)
    pool = tuple(token_pool or DEFAULT_TOKEN_POOL)
    workload = []
    for length in token_lengths:
        for _ in range(n_per_length):
            workload.append(" ".join(rng.choice(pool) for _ in range(length)))
    return workload


def benchmark_latency(
    runner, workload: list[str], warmup: int, repeats: int
) -> dict[str, float | int]:
    if not workload:
        return {
            "sentences": 0,
            "tokens": 0,
            "total_seconds": 0.0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "sent_per_s": 0.0,
            "tok_per_s": 0.0,
        }

    for index in range(warmup):
        runner.predict(workload[index % len(workload)])

    latencies = []
    total_tokens = 0
    start_total = time.perf_counter()
    for _ in range(repeats):
        for text in workload:
            start = time.perf_counter()
            runner.predict(text)
            latencies.append(time.perf_counter() - start)
            total_tokens += len(text.split())
    total_seconds = time.perf_counter() - start_total
    latency_ms = [value * 1000 for value in latencies]
    return {
        "sentences": len(latencies),
        "tokens": total_tokens,
        "total_seconds": total_seconds,
        "mean_ms": mean(latency_ms),
        "p50_ms": _percentile(latency_ms, 50),
        "p95_ms": _percentile(latency_ms, 95),
        "p99_ms": _percentile(latency_ms, 99),
        "sent_per_s": len(latencies) / total_seconds if total_seconds else 0.0,
        "tok_per_s": total_tokens / total_seconds if total_seconds else 0.0,
    }


def sample_power(duration_s: float) -> float | None:
    try:
        result = subprocess.run(
            [
                "powermetrics",
                "--samplers",
                "cpu_power,gpu_power",
                "-n",
                "1",
                "-i",
                str(int(duration_s * 1000)),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((percentile / 100) * (len(ordered) - 1))
    return ordered[index]
