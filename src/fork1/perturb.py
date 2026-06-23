from __future__ import annotations

import random
from collections.abc import Callable

from fork1.preprocess import defang, refang


KEYBOARD_NEIGHBORS = {
    "a": "qswsz",
    "c": "xdfv",
    "d": "ersfcx",
    "e": "wsdr",
    "h": "yugjbn",
    "l": "opk",
    "o": "iklp",
    "p": "ol",
    "r": "edft",
    "s": "awedxz",
    "t": "rfgy",
    "w": "qase",
}


def random_case(seed: int) -> Callable[[str], str]:
    rng = random.Random(seed)

    def perturb(text: str) -> str:
        return "".join(
            (
                char.upper()
                if char.isalpha() and rng.random() < 0.5
                else char.lower() if char.isalpha() else char
            )
            for char in text
        )

    return perturb


def keyboard_typo(rate: float, seed: int) -> Callable[[str], str]:
    rng = random.Random(seed)

    def perturb(text: str) -> str:
        chars: list[str] = []
        for char in text:
            lower = char.lower()
            if lower in KEYBOARD_NEIGHBORS and rng.random() < rate:
                replacement = rng.choice(KEYBOARD_NEIGHBORS[lower])
                chars.append(replacement.upper() if char.isupper() else replacement)
            else:
                chars.append(char)
        return "".join(chars)

    return perturb


PERTURBATIONS: dict[str, Callable] = {
    "defang": defang,
    "refang": refang,
    "random_case": random_case,
    "keyboard_typo": keyboard_typo,
}
