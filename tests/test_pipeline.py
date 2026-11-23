"""CPU tests for the Modal-side orchestration (rlrazor.pipeline) with trainers and evals mocked out."""
import json
import os
import shutil
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import rlrazor.data.gsm8k as gsm8k_mod
import rlrazor.data.rft as rft_mod
import rlrazor.eval.run_eval as run_eval
import rlrazor.train.sft as sft_mod
from rlrazor import pipeline
from rlrazor.config import RunConfig, run_id
from rlrazor.results import get_row


@pytest.fixture(scope="module")
def bf16_model_dir(tiny_model_dir, tmp_path_factory):
    path = tmp_path_factory.mktemp("bf16")
    AutoModelForCausalLM.from_pretrained(tiny_model_dir).to(torch.bfloat16).save_pretrained(path)
    AutoTokenizer.from_pretrained(tiny_model_dir).save_pretrained(path)
    return path


def _toy_ds(n=6):
    return Dataset.from_list([
        {"prompt": [{"role": "user", "content": f"q{i}"}], "question": f"q{i}", "answer": str(i),
         "completion": [{"role": "assistant", "content": f"\\boxed{{{i}}}"}]}
        for i in range(n)
    ])


@pytest.fixture
def fake_sft(monkeypatch, bf16_model_dir):
    calls = []

    def train_sft(cfg, ds, out_dir, model_path=pipeline.BASE_MODEL):
        calls.append({"cfg": cfg, "n": len(ds), "out": out_dir, "model_path": model_path})
        dst = Path(out_dir) / "step-2"
        shutil.copytree(bf16_model_dir, dst)
        return [(2, str(dst))]

    monkeypatch.setattr(sft_mod, "train_sft", train_sft)
    monkeypatch.setattr(gsm8k_mod, "load_gsm8k", lambda split, limit=None: _toy_ds())
    return calls


def test_train_run_sft_writes_index_and_is_idempotent(tmp_path, fake_sft):
    cfg = RunConfig(arm="q0", method="sft", lr=3e-5, smoke=True)
    saved = pipeline.train_run(cfg, tmp_path)
    out = tmp_path / "ckpts" / run_id(cfg)
    assert saved == [(2, str(out / "step-2"))]
    assert json.loads((out / "ckpts.json").read_text()) == {"total_steps": 2, "ckpts": [[2, str(out / "step-2")]]}
    assert (out / "config.yaml").exists()
    assert pipeline.train_run(cfg, tmp_path) == saved and len(fake_sft) == 1


def test_train_run_rejects_fp32_ckpt(tmp_path, monkeypatch, tiny_model_dir):
    monkeypatch.setattr(gsm8k_mod, "load_gsm8k", lambda split, limit=None: _toy_ds())

    def train_sft(cfg, ds, out_dir, model_path=None):
        dst = Path(out_dir) / "step-2"
        shutil.copytree(tiny_model_dir, dst)  # fp32 weights
        return [(2, str(dst))]

    monkeypatch.setattr(sft_mod, "train_sft", train_sft)
    with pytest.raises(RuntimeError, match="BF16"):
        pipeline.train_run(RunConfig(arm="q0", method="sft", lr=3e-5, smoke=True), tmp_path)


def test_rft_iter_chains_rounds_and_records_stats(tmp_path, fake_sft, monkeypatch):
    gen_from = []

    def build_rft_set(model_path, ds, k, seed, backend="vllm"):
        gen_from.append((model_path, k, seed))
        return ds.select(range(3)), {"kept_frac": 0.5, "mean_correct_per_prompt": 1.0}

    monkeypatch.setattr(rft_mod, "build_rft_set", build_rft_set)
    cfg = RunConfig(arm="q1", method="rft_iter", lr=3e-5, seed=1, rounds=3)
    saved = pipeline.train_run(cfg, tmp_path)
    out = tmp_path / "ckpts" / run_id(cfg)
    assert saved == [(r, str(out / f"step-{r}")) for r in (1, 2, 3)]
    # round r samples k=4 from the previous round's checkpoint (round 1: the base model)
    assert gen_from == [(pipeline.BASE_MODEL, 4, 1001), (str(out / "step-1"), 4, 1002), (str(out / "step-2"), 4, 1003)]
    assert [c["model_path"] for c in fake_sft] == [pipeline.BASE_MODEL, str(out / "step-1"), str(out / "step-2")]
    assert all(c["cfg"].epochs == 1 and c["cfg"].n_ckpts == 1 for c in fake_sft)
    assert json.loads((out / "ckpts.json").read_text())["total_steps"] == 3
    stats = json.loads((tmp_path / "cache" / run_id(cfg) / "dataset_stats.json").read_text())
    assert set(stats) == {"round_1", "round_2", "round_3"}
    assert not list(out.glob("round-*"))


def test_rl_distill_sft_samples_from_init_from(tmp_path, fake_sft, monkeypatch):
    seen = []
    monkeypatch.setattr(rft_mod, "build_rft_set",
                        lambda m, ds, k, seed, backend="vllm": (seen.append((m, k)) or ds, {"kept_frac": 1.0}))
    cfg = RunConfig(arm="q1", method="rl_distill_sft", lr=3e-5, init_from="grpo-x/step-400")
    pipeline.train_run(cfg, tmp_path)
    assert seen == [(str(tmp_path / "ckpts" / "grpo-x" / "step-400"), 8)]
    stats = json.loads((tmp_path / "cache" / run_id(cfg) / "dataset_stats.json").read_text())
    assert stats["kept_frac"] == 1.0
    with pytest.raises(ValueError, match="init_from"):
        pipeline.train_run(RunConfig(arm="q1", method="rl_distill_sft", lr=1e-5), tmp_path)


def _fake_eval(monkeypatch, calls):
    def evaluate_checkpoint(cfg, ckpt, step, results, base_row, smoke=False, *, total_steps, cache_dir,
                            base_model, backend, raw_dir, extra=None):
        calls.append({"cfg": cfg, "ckpt": ckpt, "base_row": base_row, "cache_dir": cache_dir, "smoke": smoke})
        row = {"run_id": run_id(cfg), "step": step, "arm": cfg.arm, "method": cfg.method, "smoke": smoke,
               "gsm8k_acc": 0.4, "kl_fwd": 0.0, "frac_trained": step / total_steps, **(extra or {})}
        from rlrazor.results import append_row
        append_row(results, row)
        return row

    monkeypatch.setattr(run_eval, "evaluate_checkpoint", evaluate_checkpoint)
    monkeypatch.setattr(run_eval, "ensure_base_kl_samples",
                        lambda cache_dir, base_model, backend: calls.append(
                            {"kl_cache": cache_dir, "env": os.environ.get("VLLM_BATCH_INVARIANT")}))


@pytest.mark.parametrize("smoke,rid,cache", [(False, "base", "kl"), (True, "base-smoke", "kl_smoke")])
def test_base_eval_builds_cache_first_and_writes_base_row(tmp_path, monkeypatch, smoke, rid, cache):
    calls = []
    _fake_eval(monkeypatch, calls)
    row = pipeline.base_eval(tmp_path, smoke=smoke)
    # R38: the base KL samples are generated under the same vLLM settings as every eval (reproducible rebuild)
    assert calls[0] == {"kl_cache": tmp_path / "cache" / cache, "env": "1"}
    assert calls[1]["base_row"] is None and calls[1]["ckpt"] == pipeline.BASE_MODEL
    assert (row["run_id"], row["arm"], row["method"], row["step"], row["smoke"]) == (rid, "base", "base", 0, smoke)
    assert get_row(tmp_path / "results", rid, 0) == row
    versions = json.loads((tmp_path / "cache" / "runtime_versions" / "cpu.json").read_text())
    assert {"torch", "transformers", "trl", "vllm", "lm-eval", "gpu"} <= set(versions)
    assert pipeline.base_eval(tmp_path, smoke=smoke) == row and len(calls) == 2  # cached


def test_eval_ckpt_uses_matching_base_row(tmp_path, monkeypatch):
    calls = []
    _fake_eval(monkeypatch, calls)
    cfg = RunConfig(arm="q0", method="sft", lr=3e-5, smoke=True)
    with pytest.raises(RuntimeError, match="base-smoke"):
        pipeline.eval_ckpt(cfg, "/ck", 2, 2, tmp_path)
    base = pipeline.base_eval(tmp_path, smoke=True)
    pipeline.eval_ckpt(cfg, "/ck", 2, 2, tmp_path)
    assert calls[-1]["base_row"] == base and calls[-1]["cache_dir"] == tmp_path / "cache" / "kl_smoke"
    assert pipeline.export(tmp_path) == 2
    lines = (tmp_path / "results.jsonl").read_text().splitlines()
    assert {json.loads(x)["run_id"] for x in lines} == {"base-smoke", run_id(cfg)}


def test_resolve_model():
    assert pipeline.resolve_model("/vol", "base") == pipeline.BASE_MODEL
    assert pipeline.resolve_model("/vol", None) == pipeline.BASE_MODEL
    assert pipeline.resolve_model("/vol", "sft-x/step-5") == "/vol/ckpts/sft-x/step-5"


@pytest.mark.parametrize("cfg", [RunConfig(arm="q3", method="sft", lr=3e-5, mix_f=0.1),
                                 RunConfig(arm="q4", method="distill", lr=1e-6, init_from="sft-x/step-5")])
def test_train_run_requires_prebuilt_general_pool(tmp_path, fake_sft, cfg):
    with pytest.raises(RuntimeError, match="build_general_pool"):
        pipeline.train_run(cfg, tmp_path)
    assert fake_sft == [] and not (tmp_path / "ckpts" / run_id(cfg)).exists()


def test_run_status_reports_what_exists(tmp_path, fake_sft, monkeypatch):
    calls = []
    _fake_eval(monkeypatch, calls)
    cfg = RunConfig(arm="q0", method="sft", lr=3e-5, smoke=True)
    assert pipeline.run_status(tmp_path, cfg) == {"index": None, "scored_steps": [], "base_row": None,
                                                  "general_pool": False, "teacher_data_done": False}
    pipeline.train_run(cfg, tmp_path)
    pipeline.base_eval(tmp_path, smoke=True)
    st = pipeline.run_status(tmp_path, cfg)
    assert st["index"]["total_steps"] == 2 and st["scored_steps"] == [] and st["base_row"]["run_id"] == "base-smoke"
    pipeline.eval_ckpt(cfg, st["index"]["ckpts"][0][1], 2, 2, tmp_path)
    assert pipeline.run_status(tmp_path, cfg)["scored_steps"] == [2]


def test_diverged_grpo_run_is_recorded_in_index_and_rows(tmp_path, monkeypatch, bf16_model_dir):
    """R50: a dead-run stop (train/grpo.py writes diverged.json) is carried into ckpts.json and every result row."""
    import rlrazor.train.grpo as grpo_mod

    monkeypatch.setattr(gsm8k_mod, "load_gsm8k", lambda split, limit=None: _toy_ds())

    def train_grpo(cfg, ds, out_dir, use_vllm=True):
        dst = Path(out_dir) / "step-97"
        shutil.copytree(bf16_model_dir, dst)
        (Path(out_dir) / "diverged.json").write_text(json.dumps({"step": 97, "rule": "r"}))
        return [(97, str(dst))]

    monkeypatch.setattr(grpo_mod, "train_grpo", train_grpo)
    cfg = RunConfig(arm="q0", method="grpo", lr=1e-5, grpo_steps=234, n_ckpts=3)
    pipeline.train_run(cfg, tmp_path)
    idx = pipeline.ckpt_index(tmp_path, cfg)
    assert idx["diverged"] == {"step": 97, "rule": "r"} and idx["total_steps"] == 234  # planned length
    calls = []
    _fake_eval(monkeypatch, calls)
    pipeline.base_eval(tmp_path)
    row = pipeline.eval_ckpt(cfg, idx["ckpts"][0][1], 97, idx["total_steps"], tmp_path)
    assert row["diverged"] is True and row["diverged_step"] == 97
    assert row["frac_trained"] == pytest.approx(97 / 234)


def test_diverged_smoke_grpo_run_keeps_smoke_planned_length(tmp_path, monkeypatch, bf16_model_dir):
    """A smoke GRPO run's planned length is SMOKE_STEPS, not cfg.grpo_steps."""
    import rlrazor.train.grpo as grpo_mod
    from rlrazor.train import SMOKE_STEPS

    monkeypatch.setattr(gsm8k_mod, "load_gsm8k", lambda split, limit=None: _toy_ds())

    def train_grpo(cfg, ds, out_dir, use_vllm=True):
        dst = Path(out_dir) / "step-1"
        shutil.copytree(bf16_model_dir, dst)
        (Path(out_dir) / "diverged.json").write_text(json.dumps({"step": 1, "rule": "r"}))
        return [(1, str(dst))]

    monkeypatch.setattr(grpo_mod, "train_grpo", train_grpo)
    cfg = RunConfig(arm="q0", method="grpo", lr=1e-5, grpo_steps=234, n_ckpts=3, smoke=True)
    pipeline.train_run(cfg, tmp_path)
    assert pipeline.ckpt_index(tmp_path, cfg)["total_steps"] == SMOKE_STEPS


def test_healthy_run_rows_say_not_diverged(tmp_path, fake_sft, monkeypatch):
    cfg = RunConfig(arm="q0", method="sft", lr=3e-5, smoke=True)
    pipeline.train_run(cfg, tmp_path)
    assert "diverged" not in pipeline.ckpt_index(tmp_path, cfg)
    calls = []
    _fake_eval(monkeypatch, calls)
    pipeline.base_eval(tmp_path, smoke=True)
    row = pipeline.eval_ckpt(cfg, pipeline.ckpt_index(tmp_path, cfg)["ckpts"][0][1], 2, 2, tmp_path)
    assert row["diverged"] is False


def test_wise_run_interpolates_and_indexes_without_training(tmp_path, monkeypatch, bf16_model_dir, fake_sft):
    """WiSE-FT configs (spec Q4) are eval-only: train_run builds the interpolation on CPU and indexes it as the
    run's one checkpoint at the source step; no trainer is called."""
    monkeypatch.setattr(pipeline, "BASE_MODEL", str(bf16_model_dir))
    src = tmp_path / "ckpts" / "teacher_sft-src"
    shutil.copytree(bf16_model_dir, src / "step-459")
    (src / "ckpts.json").write_text(json.dumps({"total_steps": 459, "ckpts": [[459, str(src / "step-459")]]}))
    cfg = RunConfig(arm="q4", method="teacher_sft", lr=3e-6, init_from="teacher_sft-src/step-459", wise_alpha=0.5)
    saved = pipeline.train_run(cfg, tmp_path)
    assert len(saved) == 1 and saved[0][0] == 459 and fake_sft == []
    path = Path(saved[0][1])
    assert path.parent.name == "wise" and (path / "config.json").exists()
    idx = json.loads((tmp_path / "ckpts" / run_id(cfg) / "ckpts.json").read_text())
    assert idx["total_steps"] == 459 and idx["ckpts"] == [[459, str(path)]]
    assert pipeline.wise_run(cfg, tmp_path) == saved  # idempotent


def test_wise_run_needs_trained_source(tmp_path):
    cfg = RunConfig(arm="q4", method="teacher_sft", lr=3e-6, init_from="missing-run/step-9", wise_alpha=0.25)
    with pytest.raises(RuntimeError, match="not trained"):
        pipeline.wise_run(cfg, tmp_path)


def test_short_prompt_rows_bounds_distill_prompt_length():
    """Q4 GKD: long Tulu prompts OOMed the full-vocab JSD loss; prompts are capped by token count."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    pool = Dataset.from_list([{"question": "word " * n, "completion": [], "answer": ""} for n in (5, 100, 300, 50)])
    out = pipeline.short_prompt_rows(pool, 256, tok)
    assert [len(tok.encode(q)) <= 256 for q in out["question"]] == [True] * 3 and len(out) == 3
    assert pipeline.DISTILL_MAX_PROMPT_TOKENS == 256
