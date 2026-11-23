"""Run configuration and deterministic run ids."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import yaml

Method = Literal[
    "sft", "grpo", "rft_offline", "rft_iter", "self_rewrite",
    "rl_distill_sft", "lora_sft", "distill",
    # R42/R43: one-off teacher data job (Qwen2.5-7B-Instruct) and SFT on its output. Named here so the
    # launcher can budget and order them; pipeline.train_run implements them.
    "teacher_data", "teacher_sft",
]


@dataclass(frozen=True)
class RunConfig:
    arm: str
    method: Method
    lr: float
    seed: int = 0
    noise_p: float = 0.0
    mix_f: float = 0.0
    lora_r: int = 0
    epochs: int = 2
    grpo_steps: int = 400
    rounds: int = 4
    n_ckpts: int = 5
    init_from: str | None = None
    smoke: bool = False
    wise_alpha: float = 0.0

    def __post_init__(self):
        if self.method not in get_args(Method):
            raise ValueError(f"unknown method: {self.method!r}")


PILOT_ARM = "pilot"  # spec §6 pilot runs: never grid data (analysis ignores unknown arms; R30)


def run_id(cfg: RunConfig) -> str:
    parts = [f"{cfg.method}-lr{cfg.lr:g}-s{cfg.seed}"]
    if cfg.arm == PILOT_ARM:
        parts.insert(0, PILOT_ARM)
    if cfg.noise_p != 0:
        parts.append(f"n{cfg.noise_p:g}")
    if cfg.mix_f != 0:
        parts.append(f"f{cfg.mix_f:g}")
    if cfg.lora_r != 0:
        parts.append(f"r{cfg.lora_r}")
    if cfg.wise_alpha != 0:
        parts.append(f"w{cfg.wise_alpha:g}")
    blob = json.dumps(dataclasses.asdict(cfg), sort_keys=True)
    parts.append(hashlib.sha1(blob.encode()).hexdigest()[:6])
    return "-".join(parts)


_FLOATS = ("lr", "noise_p", "mix_f", "wise_alpha")
_INTS = ("seed", "lora_r", "epochs", "grpo_steps", "rounds", "n_ckpts")


def config_to_yaml(cfg: RunConfig) -> str:
    return yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=True)


def config_from_yaml(s: str) -> RunConfig:
    d = yaml.safe_load(s) or {}
    names = {f.name for f in dataclasses.fields(RunConfig)}
    for k in d:
        if k not in names:
            raise ValueError(f"unknown config key: {k!r}")
    for k in _FLOATS:
        if k in d:
            d[k] = float(d[k])
    for k in _INTS:
        if k in d:
            d[k] = int(d[k])
    if "smoke" in d:
        d["smoke"] = bool(d["smoke"])
    if d.get("init_from") is not None:
        d["init_from"] = str(d["init_from"])
    return RunConfig(**d)


def load_config(path) -> RunConfig:
    return config_from_yaml(Path(path).read_text())


def dump_config(cfg: RunConfig, path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(config_to_yaml(cfg))
