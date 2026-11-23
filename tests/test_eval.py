import json
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM

from rlrazor.config import RunConfig, run_id
from rlrazor.eval import forgetting, run_eval
from rlrazor.eval.forgetting import (
    BENCHMARKS,
    forgetting_mean,
    lm_eval_args,
    lm_eval_calls,
    slice_lm_eval,
    task_versions,
    parse_lm_eval,
)
from rlrazor.data.gsm8k import load_gsm8k
from rlrazor.eval import gsm8k as gsm8k_mod
from rlrazor.eval.gsm8k import gsm8k_accuracy
from rlrazor.eval.kl import N_KL_PROMPTS, forward_kl, kl_prompts, sequence_logprobs
from rlrazor.eval.run_eval import ensure_base_kl_samples, evaluate_checkpoint
from rlrazor.results import append_row, has_row

FIXTURE = Path(__file__).parent / "fixtures" / "lm_eval_results.json"
NAMES = ["hellaswag", "mmlu", "truthfulqa", "winogrande", "ifeval", "humaneval"]


@pytest.fixture(scope="module")
def lm_eval_json():
    return json.loads(FIXTURE.read_text())


def test_benchmark_table_matches_spec():
    assert list(BENCHMARKS) == NAMES
    assert [BENCHMARKS[k]["limit"] for k in NAMES] == [500, 10, 500, 500, None, None]
    assert {k for k, b in BENCHMARKS.items() if b["chat"]} == {"ifeval", "humaneval"}
    for b in BENCHMARKS.values():
        assert set(b) == {"task", "metric", "limit", "chat"}


def test_lm_eval_args_humaneval_unsafe_flag():
    args = lm_eval_args("/ckpt", "humaneval")
    assert args[args.index("--model") + 1] == "vllm"
    assert args[args.index("--tasks") + 1] == BENCHMARKS["humaneval"]["task"]
    assert "--confirm_run_unsafe_code" in args
    assert "--apply_chat_template" in args
    assert "--limit" not in args
    assert "pretrained=/ckpt" in args[args.index("--model_args") + 1]


def test_lm_eval_args_limits_and_smoke():
    args = lm_eval_args("/ckpt", "hellaswag")
    assert args[args.index("--limit") + 1] == "500"
    assert "--confirm_run_unsafe_code" not in args
    assert "--apply_chat_template" not in args
    smoke = lm_eval_args("/ckpt", "ifeval", smoke=True)
    assert smoke[smoke.index("--limit") + 1] == "5"
    assert smoke.count("--limit") == 1


def test_parse_lm_eval_fixture(lm_eval_json):
    got = {k: parse_lm_eval(lm_eval_json, k) for k in NAMES}
    # real base-model lm-eval 0.4.13 output (A100-40GB), trimmed
    assert got == pytest.approx({
        "hellaswag": 0.512,
        "mmlu": 0.4842105263157895,
        "truthfulqa": 0.4159712901472932,
        "winogrande": 0.566,
        "ifeval": 0.25508317929759705,
        "humaneval": 0.31097560975609756,
    })
    assert lm_eval_json["results"]["mmlu"]["sample_len"] == 570  # --limit 10 applies per subject (57 x 10)


def test_forgetting_mean():
    assert forgetting_mean({"a": 0.5, "b": 0.4}, {"a": 0.6, "b": 0.4}) == pytest.approx(-0.05)


def test_forgetting_mean_requires_same_benchmarks():
    with pytest.raises(ValueError):
        forgetting_mean({"a": 0.5}, {"a": 0.6, "b": 0.4})
    with pytest.raises(ValueError):
        forgetting_mean({"a": 0.5, "b": 0.4}, {"a": 0.6})


def test_parse_lm_eval_missing_metric_raises(lm_eval_json):
    broken = json.loads(json.dumps(lm_eval_json))
    del broken["results"]["ifeval"]["prompt_level_strict_acc,none"]
    with pytest.raises(KeyError):
        parse_lm_eval(broken, "ifeval")


@pytest.mark.parametrize("n_files", [0, 2])
def test_run_lm_eval_requires_exactly_one_results_file(monkeypatch, lm_eval_json, n_files):
    def fake_run(cmd, check, env):
        out = Path(cmd[cmd.index("--output_path") + 1]) / "model__name"
        out.mkdir(parents=True)
        for i in range(n_files):
            (out / f"results_2026-11-21T00-00-0{i}.json").write_text(json.dumps(lm_eval_json))

    monkeypatch.setattr(forgetting.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="one lm-eval results file"):
        forgetting.run_lm_eval("/ckpt", "hellaswag")


def test_run_forgetting_suite_shells_out(monkeypatch, lm_eval_json):
    calls = []

    def fake_run(cmd, check, env):
        calls.append(cmd)
        assert check and env["HF_ALLOW_CODE_EVAL"] == "1"
        out = Path(cmd[cmd.index("--output_path") + 1]) / "model__name"
        out.mkdir(parents=True)
        (out / "results_2026-11-21T00-00-00.json").write_text(json.dumps(lm_eval_json))

    monkeypatch.setattr(forgetting.subprocess, "run", fake_run)
    bench = forgetting.run_forgetting_suite("/ckpt", smoke=True)
    assert list(bench) == NAMES
    assert bench["hellaswag"] == 0.512
    # R34: three grouped calls (same limit + chat setting per call), not one per benchmark
    assert [c[c.index("--tasks") + 1] for c in calls] == [
        "hellaswag,truthfulqa_mc2,winogrande", "mmlu", "ifeval,humaneval_instruct"]
    assert all(c[c.index("--limit") + 1] == "5" for c in calls)


def test_lm_eval_calls_group_by_limit_and_chat():
    assert lm_eval_calls() == [("hellaswag", "truthfulqa", "winogrande"), ("mmlu",), ("ifeval", "humaneval")]
    full = [lm_eval_args("/ckpt", names) for names in lm_eval_calls()]
    assert [a[a.index("--limit") + 1] if "--limit" in a else None for a in full] == ["500", "10", None]
    assert ["--apply_chat_template" in a for a in full] == [False, False, True]
    assert ["--confirm_run_unsafe_code" in a for a in full] == [False, False, True]
    with pytest.raises(ValueError, match="same limit"):
        lm_eval_args("/ckpt", ["hellaswag", "mmlu"])


def test_slice_lm_eval_keeps_only_the_benchmark_tree(lm_eval_json):
    mmlu = slice_lm_eval(lm_eval_json, "mmlu")
    assert set(mmlu["results"]) == {"mmlu", "mmlu_abstract_algebra"}
    assert set(mmlu["versions"]) == {"mmlu", "mmlu_abstract_algebra"}
    assert mmlu["config"] == lm_eval_json["config"]
    hs = slice_lm_eval(lm_eval_json, "hellaswag")
    assert set(hs["results"]) == set(hs["versions"]) == {"hellaswag"}
    assert parse_lm_eval(hs, "hellaswag") == parse_lm_eval(lm_eval_json, "hellaswag")


def test_gsm8k_accuracy_hf_tiny(tiny_model_dir):
    acc = gsm8k_accuracy(tiny_model_dir, limit=2, backend="hf")
    assert 0.0 <= acc <= 1.0


def test_gsm8k_accuracy_greedy_and_scored(monkeypatch):
    calls = []
    golds = list(load_gsm8k("test", limit=2)["answer"])

    def gen(model_path, prompts, n, temperature, max_tokens=512, seed=0, backend="vllm"):
        calls.append(dict(n=n, temperature=temperature, max_tokens=max_tokens, k=len(prompts)))
        return [[f"so \\boxed{{{golds[0]}}}"], [f"so \\boxed{{{golds[1]}1}}"]]

    monkeypatch.setattr(gsm8k_mod, "generate", gen)
    assert gsm8k_accuracy("/ckpt", limit=2) == 0.5
    assert calls == [dict(n=1, temperature=0.0, max_tokens=512, k=2)]


def _fake_generate(calls):
    def gen(model_path, prompts, n, temperature, max_tokens=512, seed=0, backend="vllm"):
        calls.append(dict(n=n, temperature=temperature, max_tokens=max_tokens, seed=seed))
        return [[f"The answer is \\boxed{{{i}}}."] for i in range(len(prompts))]

    return gen


def test_ensure_base_kl_samples_generates_once(tiny_model_dir, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(run_eval, "generate", _fake_generate(calls))
    p1 = ensure_base_kl_samples(tmp_path, base_model=tiny_model_dir, backend="hf")
    p2 = ensure_base_kl_samples(tmp_path, base_model=tiny_model_dir, backend="hf")
    assert p1 == p2 and Path(p1).exists()
    assert calls == [dict(n=1, temperature=1.0, max_tokens=512, seed=0)]
    cache = json.loads(Path(p1).read_text())
    assert len(cache["responses"]) == len(cache["base_logps"]) == N_KL_PROMPTS
    assert not list(Path(tmp_path).glob(".tmp-*"))


def _cfg():
    return RunConfig(arm="sft", method="sft", lr=1e-5, smoke=True)


def _versions(lm_eval_json):
    return {k: task_versions(slice_lm_eval(lm_eval_json, k)) for k in NAMES}


def _base_row(lm_eval_json, smoke):
    return {
        "bench": {k: 0.5 for k in NAMES},
        "smoke": smoke,
        "lm_eval_versions": _versions(lm_eval_json),
        "lm_eval_package": "0.4.13",
        "eval_vllm_env": dict(run_eval.EVAL_VLLM_ENV),
        "langdetect_seed": forgetting.LANGDETECT_SEED,
    }


def _build_kl_cache(cache_dir, base_model, monkeypatch):
    """The base KL cache must exist before a checkpoint is scored (base_eval builds it on Modal)."""
    monkeypatch.setattr(run_eval, "generate", _fake_generate([]))
    ensure_base_kl_samples(cache_dir, base_model=base_model, backend="hf")


@pytest.fixture(scope="module")
def other_model_dir(tiny_model_dir, tmp_path_factory):
    """Same architecture/tokenizer as the tiny base, different random init."""
    base = AutoModelForCausalLM.from_pretrained(tiny_model_dir)
    torch.manual_seed(1)
    model = Qwen2ForCausalLM(base.config)
    path = tmp_path_factory.mktemp("other_model")
    model.save_pretrained(path)
    AutoTokenizer.from_pretrained(tiny_model_dir).save_pretrained(path)
    return str(path)


def test_evaluate_checkpoint_builds_row(tiny_model_dir, tmp_path, monkeypatch, lm_eval_json):
    gen_calls = []
    _build_kl_cache(tmp_path / "kl", tiny_model_dir, monkeypatch)
    monkeypatch.setattr(run_eval, "generate", _fake_generate(gen_calls))
    monkeypatch.setattr(run_eval, "gsm8k_accuracy", lambda path, limit=None, backend="vllm": 0.25)
    monkeypatch.setattr(run_eval, "run_lm_eval", lambda path, name, smoke=False: lm_eval_json)
    results = tmp_path / "results"
    base_row = _base_row(lm_eval_json, smoke=True)

    row = evaluate_checkpoint(
        _cfg(), tiny_model_dir, 40, results, base_row, smoke=True,
        total_steps=80, cache_dir=tmp_path / "kl", base_model=tiny_model_dir, backend="hf",
    )

    assert has_row(results, run_id(_cfg()), 40)
    assert row["run_id"] == run_id(_cfg())
    assert (row["arm"], row["method"], row["lr"], row["seed"]) == ("sft", "sft", 1e-5, 0)
    assert (row["noise_p"], row["mix_f"], row["lora"]) == (0.0, 0.0, 0)
    assert (row["wise_alpha"], row["init_from"]) == (0.0, None)
    assert row["step"] == 40 and row["frac_trained"] == 0.5
    assert row["gsm8k_acc"] == 0.25
    # checkpoint == base model, so the forward KL must be exactly zero (spec sanity check)
    assert row["kl_fwd"] == pytest.approx(0.0, abs=1e-6)
    assert row["bench"]["hellaswag"] == 0.512
    expected = sum(row["bench"][k] - 0.5 for k in NAMES) / 6
    assert row["forgetting_mean"] == pytest.approx(expected)
    assert isinstance(row["gpu_seconds"], float) and row["gpu_seconds"] > 0
    assert row["lm_eval_versions"] == _versions(lm_eval_json)
    assert row["lm_eval_versions"]["hellaswag"] == {"hellaswag": 1.0}
    assert row["lm_eval_package"] == "0.4.13"
    assert row["smoke"] is True
    assert len(gen_calls) == 0  # the KL cache already existed


def test_evaluate_checkpoint_kl_scores_the_checkpoint(
    tiny_model_dir, other_model_dir, tmp_path, monkeypatch, lm_eval_json
):
    _build_kl_cache(tmp_path / "kl", tiny_model_dir, monkeypatch)
    monkeypatch.setattr(run_eval, "gsm8k_accuracy", lambda path, limit=None, backend="vllm": 0.25)
    monkeypatch.setattr(run_eval, "run_lm_eval", lambda path, name, smoke=False: lm_eval_json)
    row = evaluate_checkpoint(
        _cfg(), other_model_dir, 40, tmp_path / "results", _base_row(lm_eval_json, smoke=False),
        total_steps=80, cache_dir=tmp_path / "kl", base_model=tiny_model_dir, backend="hf",
    )
    cache = json.loads((tmp_path / "kl" / run_eval.KL_CACHE_FILE).read_text())
    other = AutoModelForCausalLM.from_pretrained(other_model_dir, dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(other_model_dir)
    other_logps = sequence_logprobs(other, tok, list(kl_prompts()["prompt"]), cache["responses"])
    expected = forward_kl(torch.tensor(cache["base_logps"]), other_logps)
    assert abs(expected) > 1e-2
    assert row["kl_fwd"] == pytest.approx(expected, rel=1e-5)
    # R46: per-sequence log-ratio sums and token counts are kept, so per-token KL can be reported later
    assert len(row["kl_logratio_sums"]) == len(row["kl_n_tokens"]) == len(cache["responses"])
    assert sum(row["kl_logratio_sums"]) / len(cache["responses"]) == pytest.approx(expected, rel=1e-5)
    assert row["kl_n_tokens"] == [len(tok(r, add_special_tokens=False)["input_ids"]) for r in cache["responses"]]
    assert row["kl_fwd_per_token"] == pytest.approx(sum(row["kl_logratio_sums"]) / sum(row["kl_n_tokens"]))


def test_evaluate_checkpoint_rejects_incomparable_base_row(tiny_model_dir, tmp_path, monkeypatch, lm_eval_json):
    monkeypatch.setattr(run_eval, "generate", _fake_generate([]))
    monkeypatch.setattr(run_eval, "gsm8k_accuracy", lambda path, limit=None, backend="vllm": 0.25)
    monkeypatch.setattr(run_eval, "run_lm_eval", lambda path, name, smoke=False: lm_eval_json)
    kw = dict(total_steps=80, cache_dir=tmp_path / "kl", base_model=tiny_model_dir, backend="hf")
    with pytest.raises(ValueError, match="smoke"):
        evaluate_checkpoint(_cfg(), tiny_model_dir, 40, tmp_path / "r", _base_row(lm_eval_json, smoke=False),
                            smoke=True, **kw)
    _build_kl_cache(tmp_path / "kl", tiny_model_dir, monkeypatch)
    stale = _base_row(lm_eval_json, smoke=True)
    stale["lm_eval_versions"]["mmlu"]["mmlu_abstract_algebra"] = 0.0
    with pytest.raises(ValueError, match="lm_eval_versions"):
        evaluate_checkpoint(_cfg(), tiny_model_dir, 40, tmp_path / "r", stale, smoke=True, **kw)
    assert not has_row(tmp_path / "r", run_id(_cfg()), 40)


def test_ensure_base_kl_samples_rejects_mismatched_cache(tiny_model_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "generate", _fake_generate([]))
    path = ensure_base_kl_samples(tmp_path, base_model=tiny_model_dir, backend="hf")
    cache = json.loads(path.read_text())
    assert {k: cache[k] for k in ("n_samples", "seed", "temperature", "max_tokens", "dtype", "device")} == {
        "n_samples": 256, "seed": 0, "temperature": 1.0, "max_tokens": 512, "dtype": "float32", "device": "cpu",
    }
    for key, bad in [("temperature", 0.7), ("dtype", "bfloat16"), ("n_samples", 128),
                     ("seed", 1), ("max_tokens", 256), ("base_model", "other"), ("device", "NVIDIA L4")]:
        path.write_text(json.dumps({**cache, key: bad}))
        with pytest.raises(ValueError, match=key):
            ensure_base_kl_samples(tmp_path, base_model=tiny_model_dir, backend="hf")


def test_evaluate_checkpoint_base_row_has_zero_forgetting(tiny_model_dir, tmp_path, monkeypatch, lm_eval_json):
    monkeypatch.setattr(run_eval, "generate", _fake_generate([]))
    monkeypatch.setattr(run_eval, "gsm8k_accuracy", lambda path, limit=None, backend="vllm": 0.3)
    monkeypatch.setattr(run_eval, "run_lm_eval", lambda path, name, smoke=False: lm_eval_json)
    row = evaluate_checkpoint(
        _cfg(), tiny_model_dir, 0, tmp_path / "results", None,
        total_steps=80, cache_dir=tmp_path / "kl", base_model=tiny_model_dir, backend="hf",
    )
    assert row["forgetting_mean"] == 0.0 and row["frac_trained"] == 0.0


def test_evaluate_checkpoint_skips_existing(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not be called")

    monkeypatch.setattr(run_eval, "generate", boom)
    monkeypatch.setattr(run_eval, "gsm8k_accuracy", boom)
    monkeypatch.setattr(run_eval, "run_lm_eval", boom)
    results = tmp_path / "results"
    append_row(results, {"run_id": run_id(_cfg()), "step": 40})
    out = evaluate_checkpoint(_cfg(), "/ckpt", 40, results, None, total_steps=80, cache_dir=tmp_path)
    assert out is None


def test_evaluate_checkpoint_saves_raw_lm_eval(tiny_model_dir, tmp_path, monkeypatch, lm_eval_json):
    monkeypatch.setattr(run_eval, "generate", _fake_generate([]))
    monkeypatch.setattr(run_eval, "gsm8k_accuracy", lambda path, limit=None, backend="vllm": 0.3)
    monkeypatch.setattr(run_eval, "run_lm_eval", lambda path, name, smoke=False: lm_eval_json)
    evaluate_checkpoint(
        _cfg(), tiny_model_dir, 0, tmp_path / "results", None, total_steps=80, cache_dir=tmp_path / "kl",
        base_model=tiny_model_dir, backend="hf", raw_dir=tmp_path / "raw",
    )
    assert sorted(p.stem for p in (tmp_path / "raw").glob("*.json")) == sorted(NAMES)
    assert json.loads((tmp_path / "raw" / "ifeval.json").read_text()) == slice_lm_eval(lm_eval_json, "ifeval")


@pytest.mark.parametrize("problem", ["device", "missing_cache", "package", "eval_env", "langdetect"])
def test_evaluate_checkpoint_guards_run_before_any_paid_work(tiny_model_dir, tmp_path, monkeypatch, lm_eval_json,
                                                             problem):
    """A KL-cache/GPU-class mismatch, a missing cache or an incomparable base row must fail before lm-eval/GSM8K."""
    def boom(*a, **k):
        raise AssertionError("paid eval work started before the guards")

    base_row = _base_row(lm_eval_json, smoke=True)
    if problem != "missing_cache":
        _build_kl_cache(tmp_path / "kl", tiny_model_dir, monkeypatch)
    if problem == "device":
        path = tmp_path / "kl" / run_eval.KL_CACHE_FILE
        path.write_text(json.dumps({**json.loads(path.read_text()), "device": "NVIDIA A100-SXM4-40GB"}))
    if problem == "package":
        base_row["lm_eval_package"] = "0.4.12"
    if problem == "eval_env":
        base_row.pop("eval_vllm_env")  # e.g. a base row scored before batch-invariant mode
    if problem == "langdetect":
        base_row["langdetect_seed"] = 1  # e.g. a changed lm-eval bootstrap
    for name in ("generate", "gsm8k_accuracy", "run_lm_eval"):
        monkeypatch.setattr(run_eval, name, boom)
    err = {"device": ValueError, "missing_cache": RuntimeError, "package": ValueError, "eval_env": ValueError,
           "langdetect": ValueError}[problem]
    match = {"device": "device", "missing_cache": "KL cache", "package": "package", "eval_env": "eval_vllm_env",
             "langdetect": "langdetect_seed"}
    with pytest.raises(err, match=match[problem]):
        evaluate_checkpoint(_cfg(), tiny_model_dir, 40, tmp_path / "r", base_row, smoke=True, total_steps=80,
                            cache_dir=tmp_path / "kl", base_model=tiny_model_dir, backend="hf")
    assert not has_row(tmp_path / "r", run_id(_cfg()), 40)


def test_evaluate_checkpoint_runs_vllm_in_batch_invariant_mode(tiny_model_dir, tmp_path, monkeypatch, lm_eval_json):
    """R37: lm-eval subprocesses (env inherited) and GSM8K greedy see VLLM_BATCH_INVARIANT=1; restored after."""
    import os

    seen = []
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    monkeypatch.setattr(run_eval, "generate", _fake_generate([]))

    def fake_lm_eval(path, names, smoke=False):
        seen.append(("lm_eval", os.environ.get("VLLM_BATCH_INVARIANT")))
        return lm_eval_json

    def fake_gsm8k(path, limit=None, backend="vllm"):
        seen.append(("gsm8k", os.environ.get("VLLM_BATCH_INVARIANT")))
        return 0.3

    monkeypatch.setattr(run_eval, "run_lm_eval", fake_lm_eval)
    monkeypatch.setattr(run_eval, "gsm8k_accuracy", fake_gsm8k)
    row = evaluate_checkpoint(_cfg(), tiny_model_dir, 0, tmp_path / "results", None, total_steps=80,
                              cache_dir=tmp_path / "kl", base_model=tiny_model_dir, backend="hf")
    assert seen and all(v == "1" for _, v in seen)
    assert row["eval_vllm_env"] == {"VLLM_BATCH_INVARIANT": "1"}
    assert row["langdetect_seed"] == forgetting.LANGDETECT_SEED
    assert "VLLM_BATCH_INVARIANT" not in os.environ


def test_run_lm_eval_seeds_langdetect(monkeypatch, lm_eval_json):
    """IFEval's langdetect checks must be seeded, or identical generations can score differently."""
    seen = {}

    def fake_run(cmd, check, env):
        seen["cmd"] = cmd
        out = Path(cmd[cmd.index("--output_path") + 1]) / "m"
        out.mkdir(parents=True)
        (out / "results_x.json").write_text(json.dumps(lm_eval_json))

    monkeypatch.setattr(forgetting.subprocess, "run", fake_run)
    forgetting.run_lm_eval("/ckpt", ("ifeval", "humaneval"))
    assert seen["cmd"][1:3] == ["-c", forgetting.LM_EVAL_BOOTSTRAP]
    assert "DetectorFactory.seed = 0" in forgetting.LM_EVAL_BOOTSTRAP
    assert seen["cmd"][3] == "run" and "ifeval,humaneval_instruct" in seen["cmd"]


def test_lm_eval_bootstrap_seeds_langdetect_before_cli(tmp_path):
    """Run the real bootstrap against stub `lm_eval` and `langdetect` packages: cli_evaluate must see the lm-eval
    arguments unchanged (argv[0] renamed to lm_eval) and DetectorFactory.seed already set (R38)."""
    import subprocess
    import sys

    (tmp_path / "langdetect").mkdir()
    (tmp_path / "langdetect" / "__init__.py").write_text("class DetectorFactory:\n    seed = None\n")
    (tmp_path / "lm_eval").mkdir()
    (tmp_path / "lm_eval" / "__init__.py").write_text("")
    (tmp_path / "lm_eval" / "__main__.py").write_text(
        "import json, os, sys\n"
        "def cli_evaluate():\n"
        "    from langdetect import DetectorFactory\n"
        "    with open(os.environ['STUB_OUT'], 'w') as f:\n"
        "        json.dump({'argv': sys.argv, 'seed': DetectorFactory.seed}, f)\n"
    )
    args = forgetting.lm_eval_args("/ckpt", ("ifeval", "humaneval"))
    out = tmp_path / "seen.json"
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(tmp_path), "STUB_OUT": str(out)}
    subprocess.run([sys.executable, "-c", forgetting.LM_EVAL_BOOTSTRAP, *args], check=True, env=env, cwd=tmp_path)
    seen = json.loads(out.read_text())
    assert seen["argv"] == ["lm_eval", *args]
    assert seen["seed"] == forgetting.LANGDETECT_SEED
