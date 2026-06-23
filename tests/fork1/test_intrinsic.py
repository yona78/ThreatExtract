from fork1.data import Sample, Span
from fork1.intrinsic import (
    domain_coverage,
    expressible_dnrti_labels,
    oracle_upper_bound,
    tokenizer_fertility,
)


class FakeTokenizer:
    def __init__(self, pieces_by_word: dict[str, list[str]]) -> None:
        self.pieces_by_word = pieces_by_word

    def tokenize(self, word: str) -> list[str]:
        return self.pieces_by_word[word]


def _sample(label: str, index: int) -> Sample:
    return Sample(
        sample_id=f"test-{index}",
        split="test",
        index=index,
        text=label,
        tokens=(label,),
        tags=(f"B-{label}",),
        gold_spans=[
            Span(label=label, start=0, end=len(label), text=label, score=None, source="gold")
        ],
    )


def test_tokenizer_fertility_is_mean_subwords_per_word() -> None:
    tokenizer = FakeTokenizer({"APT": ["APT"], "T1059.001": ["T", "1059", ".", "001"]})

    assert tokenizer_fertility(tokenizer, ["APT", "T1059.001"]) == 2.5


def test_domain_coverage_is_fraction_of_single_piece_words() -> None:
    tokenizer = FakeTokenizer({"APT": ["APT"], "T1059.001": ["T", "1059", ".", "001"]})

    assert domain_coverage(tokenizer, ["APT", "T1059.001"]) == 0.5


def test_expressible_labels_cyner_cannot_express_time_area() -> None:
    labels = expressible_dnrti_labels("cyner")

    assert "Time" not in labels
    assert "Area" not in labels
    assert {"Exp", "Tool", "SamFile"} <= labels


def test_oracle_recall_equals_expressible_support_fraction() -> None:
    samples = [_sample("Exp", 0), _sample("Time", 1), _sample("Tool", 2)]

    out = oracle_upper_bound(samples, "cyner")

    assert out["true_positive"] == 2
    assert out["false_negative"] == 1
    assert out["precision"] == 1.0
    assert out["recall"] == 2 / 3
    assert out["f1"] == 0.8
