"""Teacher data (paper App. B.1, R42/R43): adaptive sampling, keep-1-correct, pipeline job and teacher_sft."""
import json
import shutil
from pathlib import Path

import pytest
from datasets import Dataset

import rlrazor.data.gsm8k as gsm8k_mod
import rlrazor.gen as gen_mod
import rlrazor.train.sft as sft_mod
from rlrazor import pipeline
from rlrazor.config import RunConfig, run_id
from rlrazor.data.gsm8k import build_prompt
from rlrazor.data.teacher import BATCH_K, MAX_SAMPLES, TEACHER_MODEL, adaptive_sample, build_teacher_set
from rlrazor.reward import is_correct


def _ds(n=5):
    return Dataset.from_list([
        {"question": f"q{i}", "prompt": build_prompt(f"q{i}"), "answer": str(i),
         "completion": [{"role": "assistant", "content": f"gold {i} \\boxed{{{i}}}"}]}
        for i in range(n)
    ])


class FakeTeacher:
    """Row i becomes solvable at its `solve_at[i]`-th sample (0-based), never if None."""

    def __init__(self, solve_at):
        self.solve_at = solve_at
        self.drawn = {}
        self.calls = []

    def __call__(self, prompts, n, seed):
        self.calls.append((len(prompts), n, seed))
        out = []
        for p in prompts:
            i = int(p[-1]["content"][1:])
            k0 = self.drawn.get(i, 0)
            row = []
            for k in range(k0, k0 + n):
                ok = self.solve_at[i] is not None and k >= self.solve_at[i]
                row.append(f"teacher {i} sample {k} seed {seed} \\boxed{{{i if ok else i + 100}}}")
            self.drawn[i] = k0 + n
            out.append(row)
        return out


def test_adaptive_sampling_resamples_only_unsolved_up_to_cap():
    t = FakeTeacher([0, 5, None, 3, 13])
    samples, log = adaptive_sample(t, _ds(), seed=2)
    assert [len(s) for s in samples] == [4, 8, 16, 4, 16]  # solved in round 0/1/never/0/3
    assert t.calls == [(5, 4, 2000), (3, 4, 2001), (2, 4, 2002), (2, 4, 2003)]
    assert [x["solved_after"] for x in log] == [2, 3, 3, 4]
    assert BATCH_K == 4 and MAX_SAMPLES == 16


def test_adaptive_sampling_partial_last_round():
    t = FakeTeacher([None])
    samples, _ = adaptive_sample(t, _ds(1), seed=0, batch_k=3, cap=7)
    assert len(samples[0]) == 7 and [c[1] for c in t.calls] == [3, 3, 1]


def test_build_teacher_set_keeps_one_correct_and_drops_unsolved():
    ds = _ds()
    out, st = build_teacher_set(ds, FakeTeacher([0, 5, None, 3, 13]), seed=0)
    assert out["question"] == ["q0", "q1", "q3", "q4"]
    assert out["answer"] == ["0", "1", "3", "4"]  # gold kept
    for c, a, q in zip(out["completion"], out["answer"], out["question"]):
        assert c[0]["role"] == "assistant" and is_correct(c[0]["content"], a)
        assert c[0]["content"].startswith(f"teacher {a} ")  # the teacher's answer, not the gold solution
    assert st["keep_rate"] == pytest.approx(0.8) and st["n_rows"] == 4 and st["samples_total"] == 48
    assert st["boxed_rate"] == 1.0
    again, _ = build_teacher_set(ds, FakeTeacher([0, 5, None, 3, 13]), seed=0)
    assert again["completion"] == out["completion"]  # deterministic given the seed


@pytest.fixture
def fake_env(monkeypatch, tmp_path):
    made = []

    class FakeSampler:
        def __init__(self, model_path, temperature, max_tokens=512, backend="vllm", **kw):
            made.append({"model": model_path, "temperature": temperature, "max_tokens": max_tokens,
                         "backend": backend, **kw})
            self.t = FakeTeacher([0, 1, None, 0, 0, 0])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __call__(self, prompts, n, seed):
            return self.t(prompts, n, seed)

    monkeypatch.setattr(gen_mod, "Sampler", FakeSampler)
    monkeypatch.setattr(gsm8k_mod, "load_gsm8k", lambda split, limit=None: _ds(6))
    return made


def test_teacher_data_job_writes_dataset_stats_and_done_last(tmp_path, fake_env):
    cfg = RunConfig(arm="q0", method="teacher_data", lr=0.0, n_ckpts=3)
    st = pipeline.build_teacher_data(cfg, tmp_path, backend="hf")
    d = tmp_path / "cache" / "teacher_data"
    assert (d / "done.json").exists() and pipeline.teacher_data_done(tmp_path, smoke=False)
    assert pipeline.TEACHER_DATA_DONE_REL == "cache/teacher_data/done.json"
    assert fake_env == [{"model": TEACHER_MODEL, "temperature": 1.0, "max_tokens": 512, "backend": "hf",
                         "gpu_memory_utilization": pipeline.TEACHER_GPU_UTIL,
                         "max_model_len": pipeline.TEACHER_MAX_MODEL_LEN}]
    stats = json.loads((d / "dataset_stats.json").read_text())
    assert stats["keep_rate"] == pytest.approx(5 / 6) and stats["teacher"] == TEACHER_MODEL
    assert st["keep_rate"] == stats["keep_rate"]
    assert (tmp_path / "cache" / run_id(cfg) / "train_time.json").exists()  # launcher measures its cost
    assert (tmp_path / "ckpts" / run_id(cfg) / "config.yaml").exists()
    assert not (tmp_path / "ckpts" / run_id(cfg) / "ckpts.json").exists()  # no checkpoints, no evals
    ds = pipeline.load_teacher_set(tmp_path, smoke=False)
    assert len(ds) == 5 and set(ds.column_names) >= {"prompt", "completion", "answer", "question"}
    # idempotent: a second call does not sample again
    assert pipeline.build_teacher_data(cfg, tmp_path, backend="hf")["n_rows"] == 5 and len(fake_env) == 1
    assert pipeline.run_status(tmp_path, cfg)["teacher_data_done"] is True


def test_teacher_data_failure_leaves_no_done(tmp_path, fake_env, monkeypatch):
    import rlrazor.data.teacher as teacher_mod

    def boom(*a, **k):
        raise RuntimeError("engine died")

    monkeypatch.setattr(teacher_mod, "build_teacher_set", boom)
    with pytest.raises(RuntimeError):
        pipeline.build_teacher_data(RunConfig(arm="q0", method="teacher_data", lr=0.0), tmp_path, backend="hf")
    assert not pipeline.teacher_data_done(tmp_path, smoke=False)


def test_smoke_teacher_data_is_separate(tmp_path, fake_env):
    pipeline.build_teacher_data(RunConfig(arm="q0", method="teacher_data", lr=0.0, smoke=True), tmp_path,
                                backend="hf")
    assert pipeline.teacher_data_done(tmp_path, smoke=True) and not pipeline.teacher_data_done(tmp_path, smoke=False)


def test_train_run_rejects_teacher_data(tmp_path):
    with pytest.raises(ValueError, match="build_teacher_data"):
        pipeline.train_run(RunConfig(arm="q0", method="teacher_data", lr=0.0), tmp_path)


@pytest.fixture
def fake_sft(monkeypatch, tiny_model_dir, tmp_path_factory):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    bf16 = tmp_path_factory.mktemp("bf16")
    AutoModelForCausalLM.from_pretrained(tiny_model_dir).to(torch.bfloat16).save_pretrained(bf16)
    AutoTokenizer.from_pretrained(tiny_model_dir).save_pretrained(bf16)
    calls = []

    def train_sft(cfg, ds, out_dir, model_path=pipeline.BASE_MODEL):
        calls.append({"cfg": cfg, "ds": ds, "model_path": model_path})
        dst = Path(out_dir) / "step-2"
        shutil.copytree(bf16, dst)
        return [(2, str(dst))]

    monkeypatch.setattr(sft_mod, "train_sft", train_sft)
    return calls


def test_teacher_sft_trains_on_teacher_set(tmp_path, fake_env, fake_sft):
    pipeline.build_teacher_data(RunConfig(arm="q0", method="teacher_data", lr=0.0), tmp_path, backend="hf")
    cfg = RunConfig(arm="q0", method="teacher_sft", lr=3e-5, n_ckpts=3)
    saved = pipeline.train_run(cfg, tmp_path)
    assert len(saved) == 1 and len(fake_sft) == 1
    ds = fake_sft[0]["ds"]
    assert len(ds) == 5 and all(c[0]["content"].startswith("teacher ") for c in ds["completion"])
    assert fake_sft[0]["model_path"] == pipeline.BASE_MODEL


def test_teacher_sft_noise_variant_corrupts_teacher_labels(tmp_path, fake_env, fake_sft):
    pipeline.build_teacher_data(RunConfig(arm="q0", method="teacher_data", lr=0.0), tmp_path, backend="hf")
    pipeline.train_run(RunConfig(arm="q2", method="teacher_sft", lr=3e-5, noise_p=0.4, n_ckpts=3), tmp_path)
    ds = fake_sft[0]["ds"]
    assert sum(ds["corrupted"]) == 2
    for c, a, bad in zip(ds["completion"], ds["answer"], ds["corrupted"]):
        assert is_correct(c[0]["content"], a) != bad


def test_teacher_sft_without_teacher_data_raises(tmp_path, fake_sft):
    with pytest.raises(RuntimeError, match="teacher_data"):
        pipeline.train_run(RunConfig(arm="q0", method="teacher_sft", lr=3e-5), tmp_path)
    assert fake_sft == []
