import re

from datasets import Dataset, load_dataset

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

_ANNOTATION = re.compile(r"<<.*?>>")


def gold_answer(answer_field: str) -> str:
    return answer_field.split("####")[-1].strip()


def format_target(answer_field: str) -> str:
    body = answer_field.rsplit("####", 1)[0]
    body = _ANNOTATION.sub("", body).rstrip()
    return f"{body}\nThe answer is \\boxed{{{gold_answer(answer_field)}}}."


def build_prompt(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def load_gsm8k(split: str, limit: int | None = None) -> Dataset:
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    def _fmt(row):
        return {
            "prompt": build_prompt(row["question"]),
            "completion": [{"role": "assistant", "content": format_target(row["answer"])}],
            "answer": gold_answer(row["answer"]),
        }

    return ds.map(_fmt)
