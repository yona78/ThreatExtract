from fork1.preprocess import defang, detok_punct_aware, detok_single_space, normalize_text, refang


def test_single_space_matches_legacy() -> None:
    text, offsets = detok_single_space(["APT", "hit", "."])

    assert text == "APT hit ."
    assert offsets[0] == (0, 3)


def test_punct_aware_no_space_before_period() -> None:
    text, offsets = detok_punct_aware(["APT", "hit", "."])

    assert text == "APT hit."
    assert text[offsets[2][0] : offsets[2][1]] == "."


def test_refang_restores_common_ioc_forms() -> None:
    assert refang("hxxp://1.1.1[.]1") == "http://1.1.1.1"


def test_normalize_nfkc_is_idempotent() -> None:
    normalized = normalize_text("ＣＶＥ-２０２１-４４２２８", "nfkc")

    assert normalize_text(normalized, "nfkc") == normalized


def test_normalize_lower_lowercases() -> None:
    assert normalize_text("APT HIT", "lower") == "apt hit"


def test_defang_refang_round_trips_ioc() -> None:
    original = "http://1.1.1.1"

    assert refang(defang(original)) == original
