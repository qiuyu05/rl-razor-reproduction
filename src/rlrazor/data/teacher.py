"""SFT data from a stronger teacher (paper App. B.1; rulings R42/R43).

The paper annotates each training prompt with responses from a stronger model, samples up to 16 per prompt, keeps
one whose final answer is correct, and drops prompts with none. Here the teacher is Qwen2.5-7B-Instruct, prompted
exactly like the student (Qwen chat template, the GSM8K system prompt asking for a \\boxed{} answer), sampled at
T=1.0 / top-p 1 / 512 new tokens: the same settings as every other sampled training set in this study (spec §3);
the paper does not state a teacher temperature.

Sampling is adaptive: BATCH_K samples per prompt per round; only prompts with no correct sample yet are sampled
again, until MAX_SAMPLES per prompt. Round r uses seed `seed * 1000 + r`, so the set is a pure function of
(teacher, prompts, seed).
"""
from __future__ import annotations

from typing import Callable

from datasets import Dataset

from rlrazor.data.rft import select_correct
from rlrazor.reward import extract_boxed, is_correct

TEACHER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BATCH_K = 4
MAX_SAMPLES = 16
TEMPERATURE = 1.0
MAX_TOKENS = 512
SEED = 0

SampleFn = Callable[[list, int, int], list[list[str]]]  # (prompts, n, seed) -> n completions per prompt


def adaptive_sample(sample: SampleFn, ds: Dataset, seed: int, batch_k: int = BATCH_K,
                    cap: int = MAX_SAMPLES) -> tuple[list[list[str]], list[dict]]:
    """Samples per row: batch_k per round for rows with no correct sample yet, up to `cap` per row.

    Returns (samples per row, per-round log [{round, n_prompts, n_per_prompt, solved_after}])."""
    if batch_k < 1 or cap < batch_k:
        raise ValueError(f"need 1 <= batch_k <= cap, got batch_k={batch_k}, cap={cap}")
    prompts, golds = ds["prompt"], ds["answer"]
    samples: list[list[str]] = [[] for _ in range(len(ds))]
    solved = [False] * len(ds)
    todo = list(range(len(ds)))
    log, r = [], 0
    while todo:
        n = min(batch_k, cap - len(samples[todo[0]]))
        out = sample([prompts[i] for i in todo], n, seed * 1000 + r)
        if len(out) != len(todo) or any(len(o) != n for o in out):
            raise RuntimeError(f"round {r}: sampler returned the wrong shape")
        for i, o in zip(todo, out):
            samples[i].extend(o)
            solved[i] = solved[i] or any(is_correct(s, golds[i]) for s in o)
        log.append({"round": r, "n_prompts": len(todo), "n_per_prompt": n, "solved_after": sum(solved)})
        todo = [i for i in todo if not solved[i] and len(samples[i]) < cap]
        r += 1
    return samples, log


def build_teacher_set(ds: Dataset, sample: SampleFn, seed: int = SEED, batch_k: int = BATCH_K,
                      cap: int = MAX_SAMPLES) -> tuple[Dataset, dict]:
    """Adaptive sampling, then one random correct sample per prompt (prompts with none are dropped).

    The returned rows keep every input column (prompt, question, answer = gold) with `completion` = the kept teacher
    answer. Stats include the keep rate (the paper's "prompts with no correct sample are dropped")."""
    samples, log = adaptive_sample(sample, ds, seed, batch_k, cap)
    out = select_correct(ds, samples, seed)
    n = len(ds)
    flat = [s for row in samples for s in row]
    kept = [c[0]["content"] for c in out["completion"]]
    stats = {
        "teacher": None,  # filled in by the caller
        "n_prompts": n,
        "n_rows": len(out),
        "keep_rate": len(out) / n if n else 0.0,
        "samples_total": len(flat),
        "samples_per_prompt": len(flat) / n if n else 0.0,
        "first_round_solved_frac": (log[0]["solved_after"] / n) if (n and log) else 0.0,
        "boxed_rate": sum(extract_boxed(s) is not None for s in flat) / len(flat) if flat else 0.0,
        "correct_rate": sum(is_correct(s, g) for row, g in zip(samples, ds["answer"]) for s in row) / len(flat)
        if flat else 0.0,
        "distinct_frac_first_round": (sum(len(set(row[:batch_k])) for row in samples) / (batch_k * n)) if n else 0.0,
        "kept_mean_chars": sum(map(len, kept)) / len(kept) if kept else 0.0,
        "rounds": log,
        "batch_k": batch_k,
        "cap": cap,
        "seed": seed,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    return out, stats
