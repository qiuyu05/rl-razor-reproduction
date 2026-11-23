import pytest
from datasets import Dataset

from rlrazor.data.gsm8k import (
    SYSTEM_PROMPT,
    build_prompt,
    format_target,
    gold_answer,
)
from rlrazor.data.mixing import (
    _to_general_row,
    build_mixture,
    n_general_for_fraction,
)
from rlrazor.data.noise import corrupt_labels
from rlrazor.reward import extract_boxed


def _task_ds(n=100):
    rows = []
    for i in range(n):
        ans = f"step {i}\n#### {i + 1}"
        rows.append(
            {
                "question": f"q{i}",
                "answer": str(i + 1),
                "prompt": build_prompt(f"q{i}"),
                "completion": [{"role": "assistant", "content": format_target(ans)}],
            }
        )
    return Dataset.from_list(rows)


def test_format_target():
    assert (
        format_target("48/2 = <<48/2=24>>24 clips\n#### 72")
        == "48/2 = 24 clips\nThe answer is \\boxed{72}."
    )


def test_gold_answer_with_commas():
    assert gold_answer("blah\n#### 1,200") == "1,200"


def test_build_prompt_uses_exact_system_prompt():
    p = build_prompt("What is 1+1?")
    assert SYSTEM_PROMPT == "Please reason step by step, and put your final answer within \\boxed{}."
    assert p == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is 1+1?"},
    ]


def test_corrupt_labels_rate_and_all_wrong():
    ds = _task_ds(100)
    out = corrupt_labels(ds, p=0.2, seed=0)
    assert len(out) == 100
    assert sum(out["corrupted"]) == 20
    for orig, row in zip(ds, out):
        if row["corrupted"]:
            boxed = extract_boxed(row["completion"][0]["content"])
            assert boxed != row["answer"]
            assert row["answer"] == orig["answer"]
        else:
            assert row["completion"] == orig["completion"]
    assert corrupt_labels(ds, 0.2, 0).to_list() == out.to_list()
    assert corrupt_labels(ds, 0.2, 1).to_list() != out.to_list()


def test_corrupt_labels_needs_two_distinct_answers():
    ds = _task_ds(3).map(lambda r: {"answer": "5"})
    with pytest.raises(ValueError):
        corrupt_labels(ds, 0.5, 0)


def test_n_general_for_fraction():
    assert n_general_for_fraction(7473, 0.5) == 7473
    assert n_general_for_fraction(7473, 0.0) == 0
    assert n_general_for_fraction(7473, 0.02) == 153


def _pool(n):
    return Dataset.from_list(
        [
            {
                "question": f"g{i}",
                "answer": "",
                "prompt": build_prompt(f"g{i}"),
                "completion": [{"role": "assistant", "content": f"r{i}"}],
            }
            for i in range(n)
        ]
    )


def test_mixture_fraction():
    task = _task_ds(100)
    mix = build_mixture(task, _pool(500), f=0.2, seed=0)
    assert len(mix) == 125
    assert abs(sum(mix["is_general"]) / len(mix) - 0.2) < 0.005
    assert mix.to_list() == build_mixture(task, _pool(500), 0.2, 0).to_list()


def test_mixture_pool_too_small_raises():
    with pytest.raises(ValueError, match="need 100 general examples for f=0.5, pool has 10"):
        build_mixture(_task_ds(100), _pool(10), f=0.5, seed=0)


class _Tok:
    def encode(self, s):
        return s.split()


def test_to_general_row_filters():
    u = {"role": "user", "content": "hi"}
    a = {"role": "assistant", "content": "a b c"}
    row = _to_general_row([u, a], _Tok(), 5)
    assert row["answer"] == "" and row["question"] == "hi"
    assert row["prompt"] == build_prompt("hi")
    assert row["completion"] == [a]
    assert _to_general_row([u, a], _Tok(), 2) is None
    assert _to_general_row([{"role": "system", "content": "s"}, u, a], _Tok(), 5) is None
    assert _to_general_row([u, a, u, a], _Tok(), 5) is None
    assert _to_general_row([a, u], _Tok(), 5) is None


def _teacher_ds():
    """Teacher-set-shaped rows (R42/R43, R54): free-form teacher CoT ending in \\boxed{}, gold kept in `answer`.
    Answers repeat across rows and are written in different formats (1,200 vs 1200), and one row's teacher text
    has no \\boxed{}, so a donor check on the gold string or on row identity alone would be fooled."""
    golds = ["1,200", "7", "7", "3.5", "12", "12", "40", "1200", "9", "7"] * 6
    rows = []
    for i, g in enumerate(golds):
        boxed = g.replace(",", "")
        text = f"Let me think about problem {i}.\nSo the total is {boxed}.\n\\boxed{{{boxed}}}"
        if i == 13:
            text = "I computed 12 but forgot the box."  # unparseable donor: never used
        rows.append({"question": f"q{i}", "answer": g, "prompt": build_prompt(f"q{i}"),
                     "completion": [{"role": "assistant", "content": text}]})
    return Dataset.from_list(rows)


@pytest.mark.parametrize("p", [0.2, 0.4])
def test_corrupt_labels_teacher_set_donors_have_a_different_final_answer(p):
    from rlrazor.data.noise import final_number
    from rlrazor.reward import normalize_number

    ds = _teacher_ds()
    out = corrupt_labels(ds, p=p, seed=3)
    assert sum(out["corrupted"]) == round(p * len(ds))
    texts = {r["completion"][0]["content"] for r in ds}
    for orig, row in zip(ds, out):
        text = row["completion"][0]["content"]
        assert row["answer"] == orig["answer"]
        if row["corrupted"]:
            assert text in texts and text != orig["completion"][0]["content"]  # another problem's teacher answer
            donor = final_number(text)
            assert donor is not None and abs(donor - normalize_number(orig["answer"])) > 1e-6
        else:
            assert row["completion"] == orig["completion"]


def test_corrupt_labels_raises_on_unparseable_gold():
    """A chosen row whose gold answer does not parse cannot be shown to get a wrong label."""
    ds = _teacher_ds()
    rows = ds.to_list()
    rows[5]["answer"] = "unknown"
    with pytest.raises(ValueError, match="does not parse"):
        corrupt_labels(Dataset.from_list(rows), p=1.0, seed=0)


@pytest.mark.parametrize("f", [0.02, 0.05, 0.10, 0.25, 0.50])
def test_mixture_fraction_on_teacher_set(f):
    """Q3 (R54): the teacher set is held fixed and Tulu rows are added to reach general fraction f."""
    task = _teacher_ds().add_column("extra", list(range(60)))  # extra teacher-set columns are dropped
    n_gen = n_general_for_fraction(len(task), f)
    mix = build_mixture(task, _pool(200), f=f, seed=1)
    assert len(mix) == len(task) + n_gen
    assert sum(mix["is_general"]) == n_gen
    assert abs(n_gen / len(mix) - f) <= 0.5 / len(mix) + 1e-9
    kept = sorted(r["completion"][0]["content"] for r in mix if not r["is_general"])
    assert kept == sorted(r["completion"][0]["content"] for r in task)  # every teacher row exactly once
