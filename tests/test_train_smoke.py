import json
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM

from rlrazor.config import RunConfig
from rlrazor.data.gsm8k import load_gsm8k
from rlrazor.train.distill import train_distill
from rlrazor.train.grpo import train_grpo
from rlrazor.train.schedule import ckpt_steps
from rlrazor.train.sft import train_sft


@pytest.fixture(scope="module")
def train_ds():
    return load_gsm8k("train", limit=8)


def _assert_single_loadable_ckpt(ckpts):
    assert len(ckpts) == 1
    step, path = ckpts[0]
    assert step == 2
    model = AutoModelForCausalLM.from_pretrained(path, dtype="auto")
    assert {p.dtype for p in model.parameters()} == {torch.bfloat16}
    assert (Path(path) / "tokenizer_config.json").exists()
    return Path(path)


def test_ckpt_steps():
    assert ckpt_steps(400, 5) == [80, 160, 240, 320, 400]


def test_sft_smoke_saves_loadable_ckpt(tiny_model_dir, train_ds, tmp_path):
    cfg = RunConfig(arm="sft", method="sft", lr=1e-5, smoke=True)
    _assert_single_loadable_ckpt(train_sft(cfg, train_ds, str(tmp_path), model_path=tiny_model_dir))


def test_sft_full_path_saves_every_scheduled_ckpt(tiny_model_dir, train_ds, tmp_path):
    # non-smoke path: batch 8 x accum 4; 8 rows x 8 epochs / 32 = 2 steps, n_ckpts=2 -> steps [1, 2]
    cfg = RunConfig(arm="sft", method="sft", lr=1e-5, epochs=8, n_ckpts=2)
    ckpts = train_sft(cfg, train_ds, str(tmp_path), model_path=tiny_model_dir)
    assert [step for step, _ in ckpts] == [1, 2]
    assert all((Path(p) / "config.json").exists() for _, p in ckpts)


def test_lora_sft_smoke_saves_merged_full_model(tiny_model_dir, train_ds, tmp_path):
    cfg = RunConfig(arm="lora", method="lora_sft", lr=1e-4, lora_r=4, smoke=True)
    path = _assert_single_loadable_ckpt(
        train_sft(cfg, train_ds, str(tmp_path), model_path=tiny_model_dir)
    )
    assert not (path / "adapter_config.json").exists()
    assert "model_type" in json.loads((path / "config.json").read_text())


def test_grpo_smoke_runs_with_noisy_reward(tiny_model_dir, train_ds, tmp_path):
    cfg = RunConfig(arm="grpo", method="grpo", lr=1e-6, noise_p=0.2, smoke=True)
    _assert_single_loadable_ckpt(
        train_grpo(cfg, train_ds, str(tmp_path), model_path=tiny_model_dir, use_vllm=False)
    )


def test_distill_smoke(tiny_model_dir, train_ds, tmp_path):
    cfg = RunConfig(arm="distill", method="distill", lr=1e-5, smoke=True)
    _assert_single_loadable_ckpt(
        train_distill(cfg, tiny_model_dir, tiny_model_dir, train_ds, str(tmp_path))
    )
