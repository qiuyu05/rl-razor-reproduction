"""Write the grid configs of one stage to configs/grid/<stage>/ (spec §4, amended by rulings R30/R40/R42/R43).

The commands that regenerate the committed configs/grid/ YAMLs:

    uv run python scripts/make_configs.py --stage q0_sweep
    uv run python scripts/make_configs.py --stage q0_lowlr
    uv run python scripts/make_configs.py --stage q0_seeds --best-sft-lr 3e-6 --best-grpo-lr 1e-6 --best-rft-lr 3e-6
    uv run python scripts/make_configs.py --stage q1 --best-sft-lr 3e-6 --grpo-final grpo-lr1e-06-s0-6f9f01/step-234
    uv run python scripts/make_configs.py --stage q2 --best-sft-lr 3e-6 --best-grpo-lr 1e-6
    uv run python scripts/make_configs.py --stage q3 --best-sft-lr 3e-6
    uv run python scripts/make_configs.py --stage q3_seeds --best-sft-lr 3e-6 --crossover 0.02,0.05
    uv run python scripts/make_configs.py --stage q4 --best-sft-lr 3e-6 --best-rft-lr 3e-6 \
        --sft-final teacher_sft-lr3e-06-s0-72863e/step-459 --rft-final rft_offline-lr3e-06-s0-863d2a/step-384

Amendments to spec §4 (rulings win over the spec):
  * Every "SFT" arm is SFT on teacher answers, method `teacher_sft` (R42/R43, paper App. B.1); the teacher data set
    is the one-off `teacher_data` job, written first in q0_sweep so the launcher starts it first. No SFT(gold) arm.
  * LRs: teacher_sft and offline RFT {1e-5, 3e-5, 7e-5}; GRPO {1e-5, 3e-5, 5e-5} (R42, paper Table 2 ranges).
  * Q1 iterative RFT / self-rewrite / RL-distilled SFT use teacher_sft's best LR; Q2 label noise, Q3 mixing and
    Q4 LoRA are teacher_sft variants (noise_p / mix_f / lora_r).
  * Checkpoints (R40 as amended by R46, paper Table 2): SFT-like runs (teacher_sft and its variants, offline RFT,
    self-rewrite, RL-distilled SFT) train 2 epochs and evaluate 2 checkpoints, the end of epoch 1 and of epoch 2
    (n_ckpts=2); GRPO and GKD evaluate 3 evenly spaced checkpoints; iterative RFT evaluates its 4 rounds.
  * GRPO: one epoch over the 7,473 train prompts at 32 prompts/step = 234 steps (paper Table 2 RL "Epochs 1", R46);
    constant LR after 50 warmup steps lives in train/grpo.py.
  * arm is q0..q4 (R30): offline RFT is a Q1 arm even though its LR sweep and seeds run with Q0.
  * init_from is "<source run_id>/step-<n>" (R30): RL-distilled SFT <- final GRPO; GKD distillation <- final
    best teacher_sft; WiSE-FT (eval-only configs, wise_alpha > 0) <- final best teacher_sft / offline RFT.
  * Q4 GKD distillation LR 1e-5 (the spec gives none; Q4 is an extra per R45).
  * Stage q0_lowlr (FORCED deviation, R47/R50; not one of the spec's 46): the paper's GRPO LRs {1e-5..5e-5} (3B)
    all diverged at 0.5B in q0_sweep, so GRPO is re-swept at {1e-6, 3e-6}; for symmetry teacher_sft and offline RFT
    also get 3e-6. q0_sweep's committed YAMLs are left as they are.
  * Stage 2 (R54; teacher_sft 3e-6, GRPO 1e-6, offline RFT 3e-6): q2 (label noise on the teacher set:
    donors are other problems' teacher answers with a different final answer; GRPO reward flips), q3 (Tulu mixing
    of the teacher set), q3_seeds, q4 (WiSE-FT first, then LoRA and GKD; WiSE-FT configs are eval-only, run.py
    interpolates on CPU). Q1's remaining seeds use the committed q1 YAMLs.
Files are named NN_<label>.yaml; NN gives the launch order (launch.py starts configs in the order given).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rlrazor.config import RunConfig, dump_config  # noqa: E402

STAGES = ("q0_sweep", "q0_seeds", "q1", "q2", "q3", "q3_seeds", "q4")  # the spec's grid: 46 training configs
EXTRA_STAGES = ("q0_lowlr",)  # forced deviations (R47/R50)
ALL_STAGES = STAGES + EXTRA_STAGES
LOWLR_GRPO = (1e-6, 3e-6)  # R47
LOWLR_SFT = 3e-6  # R50: teacher_sft and offline RFT, for symmetry
SFT_LRS = (1e-5, 3e-5, 7e-5)  # teacher_sft and offline RFT (R42)
GRPO_LRS = (1e-5, 3e-5, 5e-5)  # R42
N_CKPTS = 3  # R40: GRPO, GKD distillation (and the eval-only/data configs' placeholder)
SFT_N_CKPTS = 2  # R46: SFT-like runs evaluate the end of epoch 1 and of epoch 2
SFT_LIKE = ("teacher_sft", "rft_offline", "self_rewrite", "rl_distill_sft", "sft", "lora_sft")
GRPO_STEPS = 234  # R46: ceil(7473 / 32), one epoch of prompts
EXTRA_SEEDS = (1, 2)
SEEDS3 = (0, 1, 2)
NOISE_P = (0.2, 0.4)
MIX_F = (0.02, 0.05, 0.10, 0.25, 0.50)
LORA_R, LORA_LRS = 64, (1e-4, 3e-4)
WISE_ALPHAS = (0.25, 0.5, 0.75)
DISTILL_LR = 1e-5
DEFAULT_OUT = ROOT / "configs" / "grid"
_INIT = re.compile(r"^[\w.+-]+/step-\d+$")


def is_training(cfg: RunConfig) -> bool:
    """A training run (the 46 of spec §6); not the teacher data job, not a WiSE-FT eval-only config."""
    return cfg.method != "teacher_data" and cfg.wise_alpha == 0


def _cfg(arm, method, lr, **kw) -> RunConfig:
    if method == "grpo":
        kw.setdefault("grpo_steps", GRPO_STEPS)
    n = SFT_N_CKPTS if method in SFT_LIKE and not kw.get("wise_alpha") else N_CKPTS
    return RunConfig(arm=arm, method=method, lr=lr, n_ckpts=n, **kw)


def _need(name, value, stage):
    if value is None:
        raise ValueError(f"stage {stage} needs {name}")
    return value


def _init(name, value, stage):
    v = _need(name, value, stage)
    if not _INIT.match(v):
        raise ValueError(f"{name}={v!r}: expected '<run_id>/step-<n>' relative to /vol/ckpts")
    return v


def _label(c: RunConfig) -> str:
    parts = [c.method, f"lr{c.lr:g}", f"s{c.seed}"]
    if c.noise_p:
        parts.append(f"n{c.noise_p:g}")
    if c.mix_f:
        parts.append(f"f{c.mix_f:g}")
    if c.lora_r:
        parts.append(f"r{c.lora_r}")
    if c.wise_alpha:
        parts.append(f"w{c.wise_alpha:g}-{c.init_from.split('/')[0].split('-')[0]}")
    return "_".join(parts)


def stage_configs(stage: str, *, best_sft_lr=None, best_grpo_lr=None, best_rft_lr=None, grpo_final=None,
                  sft_final=None, rft_final=None, crossover=None) -> list[tuple[str, RunConfig]]:
    """(file name, config) pairs of one stage, in launch order."""
    out: list[RunConfig] = []
    if stage == "q0_sweep":
        out.append(_cfg("q0", "teacher_data", 0.0))
        out += [_cfg("q0", "teacher_sft", lr) for lr in SFT_LRS]
        out += [_cfg("q0", "grpo", lr) for lr in GRPO_LRS]
        out += [_cfg("q1", "rft_offline", lr) for lr in SFT_LRS]
    elif stage == "q0_lowlr":
        out += [_cfg("q0", "grpo", lr) for lr in LOWLR_GRPO]
        out.append(_cfg("q0", "teacher_sft", LOWLR_SFT))
        out.append(_cfg("q1", "rft_offline", LOWLR_SFT))
    elif stage == "q0_seeds":
        sft, grpo, rft = (_need(n, v, stage) for n, v in (("best_sft_lr", best_sft_lr),
                                                          ("best_grpo_lr", best_grpo_lr),
                                                          ("best_rft_lr", best_rft_lr)))
        out += [_cfg("q0", "teacher_sft", sft, seed=s) for s in EXTRA_SEEDS]
        out += [_cfg("q0", "grpo", grpo, seed=s) for s in EXTRA_SEEDS]
        out += [_cfg("q1", "rft_offline", rft, seed=s) for s in EXTRA_SEEDS]
    elif stage == "q1":
        sft = _need("best_sft_lr", best_sft_lr, stage)
        src = _init("grpo_final", grpo_final, stage)
        out += [_cfg("q1", "rft_iter", sft, seed=s) for s in SEEDS3]
        out += [_cfg("q1", "self_rewrite", sft, seed=s) for s in SEEDS3]
        out.append(_cfg("q1", "rl_distill_sft", sft, init_from=src))
    elif stage == "q2":
        sft = _need("best_sft_lr", best_sft_lr, stage)
        grpo = _need("best_grpo_lr", best_grpo_lr, stage)
        out += [_cfg("q2", "teacher_sft", sft, seed=s, noise_p=p) for p in NOISE_P for s in SEEDS3]
        out += [_cfg("q2", "grpo", grpo, seed=s, noise_p=p) for p in NOISE_P for s in SEEDS3]
    elif stage == "q3":
        sft = _need("best_sft_lr", best_sft_lr, stage)
        out += [_cfg("q3", "teacher_sft", sft, mix_f=f) for f in MIX_F]
    elif stage == "q3_seeds":
        sft = _need("best_sft_lr", best_sft_lr, stage)
        fs = _need("crossover", crossover, stage)
        if len(fs) != 2:
            raise ValueError(f"crossover needs the two bracketing f values, got {fs!r}")
        out += [_cfg("q3", "teacher_sft", sft, mix_f=float(f), seed=s) for f in fs for s in EXTRA_SEEDS]
    elif stage == "q4":
        sft = _need("best_sft_lr", best_sft_lr, stage)
        sft_src = _init("sft_final", sft_final, stage)
        rft_src = _init("rft_final", rft_final, stage)
        rft = best_rft_lr if best_rft_lr is not None else sft
        # R54 launch order: WiSE-FT (cheap, eval-only) first, then LoRA and the GKD restore
        out += [_cfg("q4", "teacher_sft", sft, init_from=sft_src, wise_alpha=a) for a in WISE_ALPHAS]
        out += [_cfg("q4", "rft_offline", rft, init_from=rft_src, wise_alpha=a) for a in WISE_ALPHAS]
        out += [_cfg("q4", "teacher_sft", lr, lora_r=LORA_R) for lr in LORA_LRS]
        out.append(_cfg("q4", "distill", DISTILL_LR, init_from=sft_src))
    else:
        raise ValueError(f"unknown stage {stage!r}; one of {ALL_STAGES}")
    return [(f"{i:02d}_{_label(c)}.yaml", c) for i, c in enumerate(out)]


def write_stage(stage: str, out_root=DEFAULT_OUT, **kw) -> list[Path]:
    """Write the stage's YAML files to <out_root>/<stage>/, removing stale ones; returns the paths in order."""
    pairs = stage_configs(stage, **kw)
    d = Path(out_root) / stage
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.yaml"):
        old.unlink()
    paths = []
    for name, cfg in pairs:
        dump_config(cfg, d / name)
        paths.append(d / name)
    return paths


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=ALL_STAGES)
    ap.add_argument("--best-sft-lr", type=float, help="teacher_sft best LR (from q0_sweep)")
    ap.add_argument("--best-grpo-lr", type=float)
    ap.add_argument("--best-rft-lr", type=float, help="offline-RFT best LR")
    ap.add_argument("--grpo-final", help="final checkpoint of the best GRPO run: <run_id>/step-<n>")
    ap.add_argument("--sft-final", help="final checkpoint of the best teacher_sft run: <run_id>/step-<n>")
    ap.add_argument("--rft-final", help="final checkpoint of the best offline-RFT run: <run_id>/step-<n>")
    ap.add_argument("--crossover", help="the two f values bracketing the Q3 crossover, e.g. 0.05,0.1")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args(argv)
    cross = tuple(float(x) for x in a.crossover.split(",")) if a.crossover else None
    paths = write_stage(a.stage, a.out, best_sft_lr=a.best_sft_lr, best_grpo_lr=a.best_grpo_lr,
                        best_rft_lr=a.best_rft_lr, grpo_final=a.grpo_final, sft_final=a.sft_final,
                        rft_final=a.rft_final, crossover=cross)
    for p in paths:
        print(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
