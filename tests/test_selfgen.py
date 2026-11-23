from datasets import Dataset

from rlrazor.data import rewrite as rewrite_mod
from rlrazor.data import rft as rft_mod
from rlrazor.data.gsm8k import SYSTEM_PROMPT, build_prompt
from rlrazor.data.rewrite import REWRITE_TEMPLATE, apply_rewrites, build_rewrite_set
from rlrazor.data.rft import build_rft_set, select_correct
from rlrazor.reward import is_correct


def _ds(answers):
    return Dataset.from_list(
        [
            {
                "question": f"q{i}",
                "prompt": build_prompt(f"q{i}"),
                "completion": [
                    {"role": "assistant", "content": f"gold {i}\nThe answer is \\boxed{{{a}}}."}
                ],
                "answer": a,
            }
            for i, a in enumerate(answers)
        ]
    )


def _box(x):
    return f"work \\boxed{{{x}}}"


SAMPLES = [
    [_box(1), _box(9), _box(1), _box(1) + " alt"],  # 3 correct
    [_box(9), "no box", _box(8), _box(7)],  # none correct -> dropped
    [_box(5), _box(3), _box(3) + " b", _box(3) + " c"],  # 3 correct
]


def test_select_correct_keeps_only_correct():
    ds = _ds(["1", "2", "3"])
    out = select_correct(ds, SAMPLES, seed=0)
    assert len(out) == 2
    assert out["question"] == ["q0", "q2"]
    assert out["answer"] == ["1", "3"]
    assert out["prompt"] == [build_prompt("q0"), build_prompt("q2")]
    for c, a in zip(out["completion"], out["answer"]):
        assert c[0]["role"] == "assistant"
        assert is_correct(c[0]["content"], a)
    again = select_correct(ds, SAMPLES, seed=0)
    assert again["completion"] == out["completion"]
    picks = {
        tuple(c[0]["content"] for c in select_correct(ds, SAMPLES, seed=s)["completion"])
        for s in range(30)
    }
    assert len(picks) > 1  # pick is random over correct samples, keyed on seed


def test_build_rft_set_stats_and_generate_args(monkeypatch):
    ds = _ds(["1", "2", "3"])
    calls = {}

    def fake_generate(model_path, prompts, n, temperature, max_tokens=512, seed=0, backend="vllm"):
        calls.update(model_path=model_path, prompts=prompts, n=n, temperature=temperature,
                     max_tokens=max_tokens, seed=seed, backend=backend)
        return SAMPLES

    monkeypatch.setattr(rft_mod, "generate", fake_generate)
    out, stats = build_rft_set("m", ds, k=4, seed=7, backend="hf")
    assert calls == dict(model_path="m", prompts=ds["prompt"], n=4, temperature=1.0,
                         max_tokens=512, seed=7, backend="hf")
    assert len(out) == 2
    assert stats["kept_frac"] == 2 / 3
    assert stats["mean_correct_per_prompt"] == 2.0  # (3 + 0 + 3) / 3


def test_apply_rewrites_fallback():
    ds = _ds(["10", "20", "30"])
    rewrites = ["mine \\boxed{10}", "mine \\boxed{21}", "mine \\boxed{30}"]
    out, rate = apply_rewrites(ds, rewrites)
    assert rate == 1 / 3
    assert out["completion"][1] == ds["completion"][1]
    assert out["completion"][0] == [{"role": "assistant", "content": rewrites[0]}]
    assert out["completion"][2] == [{"role": "assistant", "content": rewrites[2]}]
    assert out["rewritten"] == [True, False, True]
    assert out["prompt"] == ds["prompt"]


def test_rewrite_prompt_contains_problem_and_solution():
    s = REWRITE_TEMPLATE.format(question="How many apples?", solution="2+3=5 \\boxed{5}")
    assert "How many apples?" in s
    assert "2+3=5 \\boxed{5}" in s
    assert s.endswith(
        "Rewrite the reference solution in your own words, step by step, ending with \\boxed{answer}."
    )


def test_build_rewrite_set_uses_rewrite_prompt_but_trains_on_original(monkeypatch):
    ds = _ds(["10", "20"])
    calls = {}

    def fake_generate(model_path, prompts, n, temperature, max_tokens=512, seed=0, backend="vllm"):
        calls.update(prompts=prompts, n=n, temperature=temperature, backend=backend)
        return [["r \\boxed{10}"], ["r \\boxed{0}"]]

    monkeypatch.setattr(rewrite_mod, "generate", fake_generate)
    out, rate = build_rewrite_set("m", ds, backend="hf")
    assert calls["n"] == 1 and calls["temperature"] == 0 and calls["backend"] == "hf"
    p0 = calls["prompts"][0]
    assert p0[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert p0[1]["role"] == "user"
    assert p0[1]["content"] == REWRITE_TEMPLATE.format(
        question="q0", solution=ds["completion"][0][0]["content"]
    )
    assert out["prompt"] == ds["prompt"]
    assert rate == 0.5
    assert out["rewritten"] == [True, False]


def test_build_rft_set_smoke_tiny_model(tiny_model_dir):
    from rlrazor.data.gsm8k import load_gsm8k

    out, stats = build_rft_set(tiny_model_dir, load_gsm8k("train", limit=4), k=2, seed=0, backend="hf")
    assert 0.0 <= stats["kept_frac"] <= 1.0
    assert len(out) == round(stats["kept_frac"] * 4)
