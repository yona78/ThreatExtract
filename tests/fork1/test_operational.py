from fork1.operational import benchmark_latency, make_workload, sample_power


class FakeRunner:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, text: str):
        self.calls += 1
        return []


def test_make_workload_returns_requested_lengths_and_count() -> None:
    workload = make_workload([16, 64], 2, seed=1)

    assert len(workload) == 4
    assert sorted(len(item.split()) for item in workload) == [16, 16, 64, 64]


def test_benchmark_latency_reports_percentiles_and_throughput() -> None:
    runner = FakeRunner()

    result = benchmark_latency(runner, ["one two", "three four"], warmup=1, repeats=2)

    assert {"p50_ms", "p95_ms", "p99_ms", "sent_per_s", "tok_per_s"} <= set(result)
    assert result["sentences"] == 4
    assert runner.calls == 5


def test_sample_power_returns_none_when_powermetrics_fails(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("powermetrics")

    monkeypatch.setattr("subprocess.run", fake_run)

    assert sample_power(1.0) is None
