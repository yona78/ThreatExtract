from fork1.preprocess import detok_punct_aware, detok_single_space


def test_single_space_matches_legacy() -> None:
    text, offsets = detok_single_space(["APT", "hit", "."])

    assert text == "APT hit ."
    assert offsets[0] == (0, 3)


def test_punct_aware_no_space_before_period() -> None:
    text, offsets = detok_punct_aware(["APT", "hit", "."])

    assert text == "APT hit."
    assert text[offsets[2][0] : offsets[2][1]] == "."
