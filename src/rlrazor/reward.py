"""Answer parsing and rewards shared by GRPO training and evaluation."""

import random
import re
from collections.abc import Callable

_BOXED = re.compile(r"\\boxed\s*\{")
_NUMBER = re.compile(r"([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)?(?:\.\d+)?)(?:\s+[A-Za-z][A-Za-z ]*)?")
_LATEX_TEXT = re.compile(r"\\(?:text|mbox)\{[^{}]*\}")
_LATEX_NOISE = re.compile(r"\\[$%!,]|[$%]")


def extract_boxed(text: str) -> str | None:
    """Contents of the last \\boxed{...} (nested braces balanced), else None."""
    matches = list(_BOXED.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    i = start
    depth = 1
    while i < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    # an unclosed last box (e.g. truncated output) is no answer
    return text[start : i - 1] if depth == 0 else None


def normalize_number(s: str) -> float | None:
    """Parse a plain number after stripping $, %, LaTeX spacing/\\text, commas and units."""
    s = _LATEX_NOISE.sub("", _LATEX_TEXT.sub("", s)).strip().rstrip(".").strip()
    m = _NUMBER.fullmatch(s)
    if m is None:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def is_correct(completion: str, gold: str) -> bool:
    boxed = extract_boxed(completion)
    if boxed is None:
        return False
    a, b = normalize_number(boxed), normalize_number(gold)
    return a is not None and b is not None and abs(a - b) < 1e-6


def completion_text(c: str | list[dict]) -> str:
    if isinstance(c, str):
        return c
    return "".join(m.get("content") or "" for m in c)


def gsm8k_reward(completions: list, answer: list[str], **kwargs) -> list[float]:
    return [
        1.0 if is_correct(completion_text(c), a) else 0.0
        for c, a in zip(completions, answer)
    ]


def make_flipped_reward(reward_fn: Callable, p: float, seed: int) -> Callable:
    """Wrap reward_fn so each reward is flipped (1 - r) independently with prob p."""
    rng = random.Random(seed)

    def flipped(completions, **kwargs) -> list[float]:
        rewards = reward_fn(completions, **kwargs)
        return [1.0 - r if rng.random() < p else r for r in rewards]

    return flipped
