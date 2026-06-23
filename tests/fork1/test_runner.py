from fork1.runner import HfTokenClassificationRunner


class FakePipeline:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.calls = []

    def __call__(self, text: str, **kwargs):
        self.calls.append((text, kwargs, self.tokenizer.model_max_length))
        return []


class FakeTokenizer:
    model_max_length = 512


def test_predict_temporarily_sets_tokenizer_max_length() -> None:
    tokenizer = FakeTokenizer()
    pipeline = FakePipeline(tokenizer)
    runner = HfTokenClassificationRunner(
        name="securebert",
        model_id="model",
        device_request="cpu",
        allow_device_fallback=False,
        offline=True,
        cache_dir=None,
    )
    runner.pipe = pipeline
    runner.tokenizer = tokenizer

    assert runner.predict("APT text", max_length=128) == []

    assert pipeline.calls == [("APT text", {}, 128)]
    assert tokenizer.model_max_length == 512
