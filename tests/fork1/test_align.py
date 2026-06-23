import pytest

from fork1.align import align_pred_to_tokens
from fork1.data import Span


def _pred(start: int, end: int, label: str = "APT") -> Span:
    return Span(
        label=label,
        start=start,
        end=end,
        text="",
        score=0.9,
        source="securebert",
    )


def test_overlap_policy_marks_any_token_with_char_overlap() -> None:
    token_offsets = [(0, 3), (4, 7), (8, 11)]

    aligned = align_pred_to_tokens(token_offsets, [_pred(4, 11)], "securebert", "overlap")

    assert aligned == [set(), {"HackOrg"}, {"HackOrg"}]


def test_majority_policy_requires_more_than_half_token_covered() -> None:
    token_offsets = [(0, 4), (5, 9)]

    aligned = align_pred_to_tokens(token_offsets, [_pred(2, 7)], "securebert", "majority")

    assert aligned == [set(), set()]


def test_contained_policy_excludes_partially_covered_tokens() -> None:
    token_offsets = [(0, 3), (4, 7), (8, 11)]

    aligned = align_pred_to_tokens(token_offsets, [_pred(5, 10)], "securebert", "contained")

    assert aligned == [set(), set(), set()]


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown alignment policy"):
        align_pred_to_tokens([(0, 3)], [_pred(0, 3)], "securebert", "nearby")
