"""Self-rewrite set: the base model restates each gold solution in its own words."""

from typing import Literal

from datasets import Dataset

from rlrazor.data.gsm8k import SYSTEM_PROMPT
from rlrazor.gen import generate
from rlrazor.reward import completion_text, is_correct

REWRITE_TEMPLATE = (
    "Problem:\n{question}\n\nReference solution:\n{solution}\n\n"
    "Rewrite the reference solution in your own words, step by step, ending with \\boxed{{answer}}."
)


def rewrite_prompt(question: str, solution: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": REWRITE_TEMPLATE.format(question=question, solution=solution)},
    ]


def apply_rewrites(ds: Dataset, rewrites: list[str]) -> tuple[Dataset, float]:
    """Use each rewrite as `completion` if it still reaches the gold answer, else keep the gold.

    The original `prompt` is kept. Adds bool column `rewritten`; returns (dataset, fallback_rate).
    """
    if len(rewrites) != len(ds):
        raise ValueError(f"{len(rewrites)} rewrites for {len(ds)} rows")
    completions, flags = [], []
    for rw, gold_c, ans in zip(rewrites, ds["completion"], ds["answer"]):
        ok = is_correct(rw, ans)
        completions.append([{"role": "assistant", "content": rw}] if ok else gold_c)
        flags.append(ok)
    out = (
        ds.remove_columns("completion")
        .add_column("completion", completions)
        .add_column("rewritten", flags)
    )
    n = len(ds)
    fallback_rate = (n - sum(flags)) / n if n else 0.0
    return out, fallback_rate


def build_rewrite_set(
    model_path: str, ds: Dataset, backend: Literal["vllm", "hf"] = "vllm"
) -> tuple[Dataset, float]:
    """Greedy (T=0, n=1) rewrite of each gold solution by model_path, then apply_rewrites."""
    prompts = [
        rewrite_prompt(q, completion_text(c)) for q, c in zip(ds["question"], ds["completion"])
    ]
    outs = generate(model_path, prompts, n=1, temperature=0, max_tokens=512, backend=backend)
    return apply_rewrites(ds, [o[0] for o in outs])
