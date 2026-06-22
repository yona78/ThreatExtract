from fork1.runner import HfTokenClassificationRunner


class FakePipeline:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, text: str, **kwargs):
        self.calls.append((text, kwargs))
        return []


def test_predict_passes_optional_max_length_to_pipeline() -> None:
    pipeline = FakePipeline()
    runner = HfTokenClassificationRunner(
        name="securebert",
        model_id="model",
        device_request="cpu",
        allow_device_fallback=False,
        offline=True,
        cache_dir=None,
    )
    runner.pipe = pipeline

    assert runner.predict("APT text", max_length=128) == []

    assert pipeline.calls == [("APT text", {"truncation": True, "max_length": 128})]
