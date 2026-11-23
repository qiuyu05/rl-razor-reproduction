from datasets import Dataset, concatenate_datasets, load_dataset

from rlrazor.data.gsm8k import build_prompt

TOKENIZER_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def n_general_for_fraction(n_task: int, f: float) -> int:
    if f == 0:
        return 0
    return round(n_task * f / (1 - f))


def _to_general_row(messages: list[dict], tokenizer, max_tokens: int) -> dict | None:
    """Single-turn (one user then one assistant) rows within the token budget, else None."""
    if len(messages) != 2:
        return None
    user, assistant = messages
    if user["role"] != "user" or assistant["role"] != "assistant":
        return None
    if len(tokenizer.encode(assistant["content"])) > max_tokens:
        return None
    return {
        "prompt": build_prompt(user["content"]),
        "completion": [{"role": "assistant", "content": assistant["content"]}],
        "answer": "",
        "question": user["content"],
    }


def load_general_pool(seed: int, max_tokens: int = 1024, pool_size: int = 20000) -> Dataset:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    stream = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=10000)
    rows = []
    for ex in stream:
        row = _to_general_row(ex["messages"], tok, max_tokens)
        if row is not None:
            rows.append(row)
            if len(rows) >= pool_size:
                break
    return Dataset.from_list(rows)


def build_mixture(task_ds: Dataset, general_pool: Dataset, f: float, seed: int) -> Dataset:
    k = n_general_for_fraction(len(task_ds), f)
    if k > len(general_pool):
        raise ValueError(f"need {k} general examples for f={f}, pool has {len(general_pool)}")
    cols = ["prompt", "completion", "answer", "question"]
    task = task_ds.select_columns(cols).add_column("is_general", [False] * len(task_ds))
    gen = (
        general_pool.shuffle(seed=seed)
        .select(range(k))
        .select_columns(cols)
        .add_column("is_general", [True] * k)
    )
    return concatenate_datasets([task, gen]).shuffle(seed=seed)
