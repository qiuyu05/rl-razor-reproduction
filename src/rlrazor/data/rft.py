"""Rejection-sampling fine-tuning (RFT) sets: sample k per prompt, keep 1 random correct."""

import random
from typing import Literal

from datasets import Dataset

from rlrazor.gen import generate
from rlrazor.reward import is_correct


def _correct_samples(samples: list[str], gold: str) -> list[str]:
    return [s for s in samples if is_correct(s, gold)]


def select_correct(ds: Dataset, samples: list[list[str]], seed: int) -> Dataset:
    """Per row, keep one uniformly random correct sample as `completion`; drop rows with none.

    `prompt`, `answer`, `question` (and any other columns) are carried over unchanged.
    """
    if len(samples) != len(ds):
        raise ValueError(f"{len(samples)} sample lists for {len(ds)} rows")
    rng = random.Random(seed)
    keep, chosen = [], []
    for i, (row_samples, gold) in enumerate(zip(samples, ds["answer"])):
        correct = _correct_samples(row_samples, gold)
        if correct:
            keep.append(i)
            chosen.append([{"role": "assistant", "content": rng.choice(correct)}])
    return ds.select(keep).remove_columns("completion").add_column("completion", chosen)


def build_rft_set(
    model_path: str,
    ds: Dataset,
    k: int,
    seed: int,
    backend: Literal["vllm", "hf"] = "vllm",
) -> tuple[Dataset, dict]:
    """Sample k completions per prompt (T=1.0, 512 tokens) from model_path and run select_correct.

    Used for offline RFT (base model), iterative RFT (current checkpoint) and RL-distilled SFT
    (final GRPO model). Stats: kept_frac = kept/input rows; mean_correct_per_prompt over all rows.
    """
    samples = generate(
        model_path, ds["prompt"], n=k, temperature=1.0, max_tokens=512, seed=seed, backend=backend
    )
    out = select_correct(ds, samples, seed)
    n = len(ds)
    n_correct = sum(len(_correct_samples(s, g)) for s, g in zip(samples, ds["answer"]))
    stats = {
        "kept_frac": len(out) / n if n else 0.0,
        "mean_correct_per_prompt": n_correct / n if n else 0.0,
    }
    return out, stats
