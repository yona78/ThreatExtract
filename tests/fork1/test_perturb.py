from fork1.perturb import PERTURBATIONS, defang, keyboard_typo, random_case, refang


def test_random_case_is_deterministic_for_seed() -> None:
    perturb = random_case(seed=7)

    assert perturb("Threat Intel") == random_case(seed=7)("Threat Intel")
    assert perturb("Threat Intel") != "Threat Intel"


def test_random_case_keeps_rng_state_across_token_calls() -> None:
    perturb = random_case(seed=7)

    assert perturb("abcdefghijklmnopqrstuvwxyz") != perturb("abcdefghijklmnopqrstuvwxyz")


def test_keyboard_typo_is_deterministic_for_seed() -> None:
    perturb = keyboard_typo(rate=0.5, seed=3)

    assert perturb("powershell") == keyboard_typo(rate=0.5, seed=3)("powershell")


def test_keyboard_typo_keeps_rng_state_across_token_calls() -> None:
    perturb = keyboard_typo(rate=1.0, seed=3)

    assert perturb("ssssssssss") != perturb("ssssssssss")


def test_defang_refang_round_trips_ioc() -> None:
    value = "http://1.1.1.1"

    assert refang(defang(value)) == value


def test_perturbations_registry_contains_expected_keys() -> None:
    assert {"defang", "refang", "random_case", "keyboard_typo"} <= set(PERTURBATIONS)
