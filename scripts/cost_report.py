"""Measured $ per training run / per checkpoint eval, and the projected cost of the spec §6 grid.

    uv run python scripts/cost_report.py [--fetch] [--results PATH] [--train-times PATH] [--spent USD] [--json PATH]

Inputs (both written by modal_app.export_results; --fetch exports and downloads them from the Volume first):
  --results      results.jsonl: one row per scored checkpoint; `gpu_seconds` = eval GPU time inside eval_ckpt
  --train-times  train_times.jsonl: per run, `wall_seconds` inside the train function plus the run's config
Smoke rows/runs are ignored. Pilot runs (arm "pilot") are the intended source of the unit costs.

Prices, $ per hour of GPU (Modal bills per second; set these from modal.com/pricing):
  RLRAZOR_GPU_PRICE_PER_HR       training GPU (RLRAZOR_GPU, default A100-40GB)    default 2.10
  RLRAZOR_EVAL_GPU_PRICE_PER_HR  eval GPU (RLRAZOR_EVAL_GPU, default L4)          default 0.80
  (Defaults: Modal's A100-40GB / L4 list prices; modal.com was unreachable from this sandbox on 2026-11-22,
  but `modal billing report` matched them: pilot SFT train 478 billed A100-s = $0.279 at $2.10/h.)
Overheads the rows cannot see (defaults calibrated on the pilot's `modal billing report`):
  RLRAZOR_OVERHEAD_FRAC     CPU + memory that Modal bills on top of the GPU, as a fraction of GPU $
                            (pilot SFT app: $0.040 CPU+memory on $0.654 GPU = 6.1%)                 default 0.06
  RLRAZOR_CONTAINER_S       per-container GPU time outside the timed code (start, image load, imports,
                            idle scale-down window), seconds; added to every train and eval
                            (pilot: train 478 billed s vs 380 timed; eval ~845 billed vs ~760 timed) default 90
ASSUMPTIONS for what the pilot does not measure (printed with every report; groups that use one are tagged
"est." in the by-group table):
  RLRAZOR_GEN_S_PER_1K      vLLM sampling of 1,000 GSM8K completions by the 0.5B model (T=1, <=512 tok)  default 10
  RLRAZOR_ENGINE_START_S    one vLLM engine start + trainer re-init per generation phase (RFT, every
                            iterative-RFT round, self-rewrite, teacher data)                       default 90
  RLRAZOR_DISTILL_STEP_S    one GKD step (32 prompts, HF generate of student rollouts)             default 120
  RLRAZOR_GENERAL_ROW_COST  SFT cost of one Tulu row relative to one GSM8K row (longer responses)  default 3
  RLRAZOR_RFT_KEEP          fraction of GSM8K prompts with >=1 correct sample (RFT set size)         default 0.9
CPU-only work (general-pool build, WiSE-FT interpolation; 4 cores, 16 GiB):
  RLRAZOR_CPU_CORE_PRICE_PER_HR / RLRAZOR_MEM_GIB_PRICE_PER_HR  defaults 0.047 / 0.008 (Modal list prices)
  RLRAZOR_POOL_CPU_S / RLRAZOR_WISE_CPU_S   wall seconds of one pool build / one interpolation   1200 / 300

Output: a per-row table (run_id, step, gpu_seconds, $), per-run training cost, unit costs, the assumptions, and
the projection of the grid with Q0 as amended (R39 + paper fidelity, App. B.1: SFT(teacher) LR {1e-5,3e-5,7e-5} +
2 seeds, GRPO LR {1e-5,3e-5,5e-5} + 2 seeds, no SFT(gold) arm, plus the one-off teacher data generation with
adaptive sampling up to 16 per prompt): full grid, stage 1 (Q0+Q1) at 5 and 3 ckpts/run for BOTH teacher options
(Qwen2.5-7B-Instruct short CoT; DeepSeek-R1-Distill-Qwen-7B long traces; estimates, see TEACHERS),
remainder (Q2-Q4), the optional async-staleness experiment, and the spec §6 cut order applied one cut at a time
with the running total. `run_cost(cfg)` prices one config (used by scripts/launch.py's --budget guard).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

N_TRAIN = 7473  # GSM8K train rows
SFT_BATCH = 32
SPEC_BUDGET = 75.0


def _env(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


_ASSUMED = {  # field -> (env var, default, description)
    "gen_s_per_1k": ("RLRAZOR_GEN_S_PER_1K", 10.0, "s per 1k 0.5B vLLM samples"),
    "engine_start_s": ("RLRAZOR_ENGINE_START_S", 90.0, "s per generation phase (engine start)"),
    "distill_step_s": ("RLRAZOR_DISTILL_STEP_S", 120.0, "s per GKD step"),  # stage 2 measured about 118
    "general_row_cost": ("RLRAZOR_GENERAL_ROW_COST", 3.0, "x cost of a Tulu row vs a GSM8K row"),
    "rft_keep": ("RLRAZOR_RFT_KEEP", 0.9, "RFT keep rate"),
    "pool_cpu_s": ("RLRAZOR_POOL_CPU_S", 1200.0, "s per general-pool build (CPU)"),
    "wise_cpu_s": ("RLRAZOR_WISE_CPU_S", 300.0, "s per WiSE interpolation (CPU)"),
}


# Teacher options for SFT(teacher) (paper App. B.1: teacher responses, up to 16 samples per prompt, keep 1 correct).
# Adaptive sampling: 4 per prompt, then only still-unsolved prompts are resampled, up to 16 in total; costed as
# 4 + 12 * unsolved_after_4 samples per prompt. All values are ESTIMATES (no teacher was run in the pilot).
#   gen_s_per_1k: training-GPU seconds per 1,000 teacher completions (7B in vLLM on the A100; long traces cost more)
#   sft_row_cost: SFT cost of one teacher row relative to one GSM8K-gold row (measured pilot unit), from sequence
#                 length: gold ~220 tokens (prompt + answer); Qwen2.5-7B short CoT ~400; R1-distill ~1.5-2k
TEACHERS = {
    "qwen2.5-7b-instruct": {"label": "Qwen2.5-7B-Instruct (short CoT)", "gen_s_per_1k": 60.0,
                            "unsolved_after_4": 0.05, "keep": 0.97, "sft_row_cost": 1.8},
    "r1-distill-qwen-7b": {"label": "DeepSeek-R1-Distill-Qwen-7B (long traces ~1-2k tok)", "gen_s_per_1k": 300.0,
                           "unsolved_after_4": 0.05, "keep": 0.97, "sft_row_cost": 8.0},
}
DEFAULT_TEACHER = "qwen2.5-7b-instruct"


def teacher_samples_per_prompt(t: dict) -> float:
    return 4 + 12 * t["unsolved_after_4"]


@dataclasses.dataclass(frozen=True)
class Prices:
    train_hr: float = 2.10
    eval_hr: float = 0.80
    overhead: float = 0.06
    container_s: float = 90.0
    cpu_core_hr: float = 0.047
    mem_gib_hr: float = 0.008
    gen_s_per_1k: float = 10.0
    engine_start_s: float = 90.0
    distill_step_s: float = 120.0  # stage-2 GKD measured about 118 s/step (HF generate inside GKDTrainer)
    general_row_cost: float = 3.0
    rft_keep: float = 0.9
    pool_cpu_s: float = 1200.0
    wise_cpu_s: float = 300.0

    @classmethod
    def from_env(cls) -> "Prices":
        kw = {
            "train_hr": _env("RLRAZOR_GPU_PRICE_PER_HR", cls.train_hr),
            "eval_hr": _env("RLRAZOR_EVAL_GPU_PRICE_PER_HR", cls.eval_hr),
            "overhead": _env("RLRAZOR_OVERHEAD_FRAC", cls.overhead),
            "container_s": _env("RLRAZOR_CONTAINER_S", cls.container_s),
            "cpu_core_hr": _env("RLRAZOR_CPU_CORE_PRICE_PER_HR", cls.cpu_core_hr),
            "mem_gib_hr": _env("RLRAZOR_MEM_GIB_PRICE_PER_HR", cls.mem_gib_hr),
        }
        kw.update({f: _env(var, default) for f, (var, default, _) in _ASSUMED.items()})
        return cls(**kw)

    def assumptions(self) -> str:
        own = "; ".join(f"{getattr(self, f):g} {desc}" for f, (_, _, desc) in _ASSUMED.items())
        teachers = "; ".join(
            f"teacher {k}: {t['gen_s_per_1k']:g} s per 1k samples, {teacher_samples_per_prompt(t):g} samples/prompt "
            f"(adaptive, <=16), keep {t['keep']:g}, SFT row cost x{t['sft_row_cost']:g}" for k, t in TEACHERS.items())
        return f"{own}; {teachers}"

    def train_usd(self, seconds: float) -> float:
        return (seconds + self.container_s) / 3600 * self.train_hr * (1 + self.overhead)

    def eval_usd(self, seconds: float) -> float:
        return (seconds + self.container_s) / 3600 * self.eval_hr * (1 + self.overhead)

    def cpu_usd(self, seconds: float, cores: int = 4, mem_gib: int = 16) -> float:
        """A CPU-only function (modal_app.wise / build_general_pool: cpu=4, memory=16 GiB)."""
        return (seconds + self.container_s) / 3600 * (cores * self.cpu_core_hr + mem_gib * self.mem_gib_hr)


def sft_steps(n_rows: float, epochs: int = 2) -> int:
    return math.ceil(n_rows * epochs / SFT_BATCH)


@dataclasses.dataclass(frozen=True)
class Units:
    """Measured unit times (seconds): eval per checkpoint, SFT per optimizer step, GRPO per step."""
    eval_s: float
    sft_step_s: float
    grpo_step_s: float
    sources: dict


def measure(rows: list[dict], train_times: list[dict]) -> Units:
    real = [r for r in rows if not r.get("smoke") and r.get("gpu_seconds") is not None]
    ckpt = [r for r in real if r.get("run_id") != "base" and r.get("arm") != "base"]
    use = ckpt or [r for r in real if r.get("run_id") == "base"]
    if not use:
        raise SystemExit("no non-smoke results row with gpu_seconds: run the pilot first")
    eval_s = sum(r["gpu_seconds"] for r in use) / len(use)
    sft, grpo = [], []
    for t in train_times:
        cfg = t.get("config") or {}
        if cfg.get("smoke"):
            continue
        if cfg.get("method") == "sft" and not cfg.get("mix_f") and not cfg.get("lora_r"):
            sft.append((t["run_id"], t["wall_seconds"] / sft_steps(N_TRAIN, cfg.get("epochs", 2))))
        elif cfg.get("method") == "grpo":
            grpo.append((t["run_id"], t["wall_seconds"] / cfg.get("grpo_steps", 400)))
    if not sft or not grpo:
        raise SystemExit(f"need a non-smoke SFT and GRPO training time (have sft={sft}, grpo={grpo})")
    return Units(
        eval_s=eval_s,
        sft_step_s=sum(s for _, s in sft) / len(sft),
        grpo_step_s=sum(s for _, s in grpo) / len(grpo),
        sources={"eval_rows": [f"{r['run_id']}@{r['step']}" for r in use], "sft": [r for r, _ in sft],
                 "grpo": [r for r, _ in grpo]},
    )


# ---------------------------------------------------------------- per-run cost model

class NoCostModel(ValueError):
    """The cost model does not know this config's method."""


def train_seconds(method: str, u: Units, p: Prices, *, epochs: int = 2, grpo_steps: int = 400, rounds: int = 4,
                  mix_f: float = 0.0, teacher: str = DEFAULT_TEACHER) -> tuple[float, bool]:
    """Training-GPU seconds of one run (excluding the container overhead) and whether it rests on an assumption."""
    gen = lambda n: p.engine_start_s + n / 1000 * p.gen_s_per_1k  # noqa: E731  one 0.5B sampling phase
    sft = lambda rows, ep=epochs: sft_steps(rows, ep) * u.sft_step_s  # noqa: E731
    keep = p.rft_keep * N_TRAIN
    if method in ("sft", "lora_sft"):
        if mix_f > 0:  # GSM8K held fixed, Tulu rows added to reach general fraction f; Tulu rows cost more per row
            general = N_TRAIN * mix_f / (1 - mix_f)
            rows = N_TRAIN + general
            return sft(rows) * (N_TRAIN + p.general_row_cost * general) / rows, True
        return sft(N_TRAIN), method == "lora_sft"  # LoRA assumed to cost what full SFT does
    if method == "grpo":
        return grpo_steps * u.grpo_step_s, False
    if method in ("rft_offline", "rl_distill_sft"):
        return gen(8 * N_TRAIN) + sft(keep), True
    if method == "rft_iter":  # every round restarts vLLM (sampling) and the SFT trainer inside one container
        return rounds * (gen(4 * N_TRAIN) + sft(keep, 1)), True
    if method == "self_rewrite":
        return gen(N_TRAIN) + sft(N_TRAIN), True
    if method == "teacher_sft":  # SFT(teacher) on the cached teacher set (generation costed separately)
        t = TEACHERS[teacher]
        task = t["keep"] * N_TRAIN
        if mix_f > 0:  # Q3 (R54): teacher set held fixed, Tulu rows added to reach general fraction f
            general = task * mix_f / (1 - mix_f)
            rows = task + general
            return sft(rows) * (task * t["sft_row_cost"] + general * p.general_row_cost) / rows, True
        return sft(task) * t["sft_row_cost"], True
    if method == "distill":
        return 100 * p.distill_step_s, True
    if method == "teacher_data":  # the one-off teacher set as its own budgeted job (R44); no evals
        return teacher_data_seconds(p, teacher), True
    raise NoCostModel(f"no cost model for method {method!r}")


def teacher_data_seconds(p: Prices, teacher: str = DEFAULT_TEACHER) -> float:
    """One-off teacher set: adaptive sampling (4, then unsolved prompts again, up to 16) over the GSM8K train
    prompts (estimate; includes one 7B engine start and model download, costed as 3 engine starts)."""
    t = TEACHERS[teacher]
    return 3 * p.engine_start_s + teacher_samples_per_prompt(t) * N_TRAIN / 1000 * t["gen_s_per_1k"]


def run_cost(cfg, u: Units, p: Prices, *, train: bool = True, n_evals: int | None = None,
             teacher: str = DEFAULT_TEACHER) -> tuple[float, bool]:
    """Projected $ of one config (a RunConfig or its dict): training (unless train=False) plus its checkpoint
    evals (default: rounds for rft_iter, 1 for smoke and WiSE-FT, else n_ckpts). A WiSE-FT config (wise_alpha > 0)
    costs its CPU interpolation (when train) plus its evals. Returns ($, estimated)."""
    c = cfg if isinstance(cfg, dict) else dataclasses.asdict(cfg)
    smoke = bool(c.get("smoke"))
    n_evals_default = n_evals is None
    if n_evals is None:
        n_evals = (0 if c["method"] == "teacher_data" else 1 if smoke
                   else c.get("rounds", 4) if c["method"] == "rft_iter" else c.get("n_ckpts", 5))
    usd, est = n_evals * p.eval_usd(u.eval_s), False
    if c.get("wise_alpha"):  # WiSE-FT (eval-only): one CPU interpolation and one checkpoint eval, no training GPU
        if n_evals_default:
            usd = p.eval_usd(u.eval_s)
        return usd + (p.cpu_usd(p.wise_cpu_s) if train else 0.0), True
    if train:
        s, est = train_seconds(c["method"], u, p, epochs=c.get("epochs", 2), grpo_steps=c.get("grpo_steps", 400),
                               rounds=c.get("rounds", 4), mix_f=c.get("mix_f", 0.0), teacher=teacher)
        usd += p.train_usd(s)
    return usd, est


# ---------------------------------------------------------------- the grid (spec §4/§6, Q0 amended: R39 + paper)

@dataclasses.dataclass(frozen=True)
class Group:
    """`count` identical jobs: training-GPU seconds each, checkpoints evaluated each, CPU $ each, cut tags."""
    arm: str
    label: str
    count: int
    train_s: float
    n_evals: int
    tags: frozenset = frozenset()
    ckpt_cut: bool = True  # False: evaluated checkpoints are rounds/fixed (not "5 evenly spaced")
    train_gpu: bool = True  # False: no training-GPU container (WiSE-FT, pool build)
    estimated: bool = False  # True: rests on an assumption, not on a pilot measurement
    cpu_usd: float = 0.0
    is_run: bool = True  # counts as a training run (the teacher data job and CPU jobs do not)


def grid(u: Units, p: Prices, grpo_steps: int = 400, teacher: str = DEFAULT_TEACHER) -> list[Group]:
    """46 training runs (Q0 10, Q1 12, Q2 12, Q3 9, Q4 3) plus the teacher data job and CPU jobs.

    Q0 follows the paper (App. B.1; coordinator amendment of R39): SFT(teacher) LR {1e-5, 3e-5, 7e-5} + 2 seeds,
    GRPO LR {1e-5, 3e-5, 5e-5} + 2 seeds (LR does not change the cost). No SFT(gold) arm."""
    def g(arm, label, count, method, n_evals=5, tags=(), **kw):
        s, est = train_seconds(method, u, p, grpo_steps=grpo_steps, mix_f=kw.pop("mix_f", 0.0), teacher=teacher)
        return Group(arm, label, count, s, n_evals, frozenset(tags), estimated=est, **kw)

    out = [
        # Q0
        Group("q0", f"teacher data ({teacher}, adaptive <=16/prompt), one-off", 1, teacher_data_seconds(p, teacher),
              0, ckpt_cut=False, estimated=True, is_run=False),
        g("q0", "SFT(teacher), LR 1e-5/3e-5/7e-5 + 2 seeds", 5, "teacher_sft"),
        g("q0", "GRPO, LR 1e-5/3e-5/5e-5 + 2 seeds", 5, "grpo"),
        # Q1
        g("q1", "offline RFT (k=8), sweep + seeds", 5, "rft_offline"),
        g("q1", "iterative RFT, 4 rounds", 3, "rft_iter", n_evals=4, ckpt_cut=False),
        g("q1", "self-rewrite SFT", 3, "self_rewrite"),
        g("q1", "RL-distilled SFT (control)", 1, "rl_distill_sft"),
        # Q2
        g("q2", "SFT label noise p=0.2", 3, "sft", tags={"p0.2"}),
        g("q2", "SFT label noise p=0.4", 3, "sft", tags={"p0.4"}),
        g("q2", "GRPO reward noise p=0.2", 3, "grpo", tags={"p0.2"}),
        g("q2", "GRPO reward noise p=0.4", 3, "grpo", tags={"p0.4"}),
        # Q3 (+ the general pool, built once on CPU; also used by the Q4 distillation)
        Group("q3", "general (Tulu) pool build, CPU, once", 1, 0.0, 0, ckpt_cut=False, train_gpu=False,
              estimated=True, cpu_usd=p.cpu_usd(p.pool_cpu_s), is_run=False),
    ]
    for f in (0.02, 0.05, 0.10, 0.25, 0.50):
        out.append(g("q3", f"SFT + general f={f:g}", 1, "sft", mix_f=f))
    # 2 extra seeds at the two f values bracketing the crossover (unknown until the sweep): costed at f=0.10, 0.25
    out.append(g("q3", "extra seeds f=0.1 (x2)", 2, "sft", tags={"q3_extra_seeds"}, mix_f=0.10))
    out.append(g("q3", "extra seeds f=0.25 (x2)", 2, "sft", tags={"q3_extra_seeds"}, mix_f=0.25))
    out += [
        g("q4", "LoRA SFT r=64", 2, "lora_sft", tags={"lora"}),
        g("q4", "on-policy distillation (GKD, 100 steps)", 1, "distill"),
        Group("q4", "WiSE-FT (2 runs x 3 alphas): CPU interp + eval", 6, 0.0, 1, ckpt_cut=False, train_gpu=False,
              estimated=True, cpu_usd=p.cpu_usd(p.wise_cpu_s), is_run=False),
    ]
    return out


def staleness(u: Units, p: Prices, grpo_steps: int = 400) -> list[Group]:
    """Optional async-staleness experiment: k in {0,4,16,64}-step-old rollouts; k=0 reuses Q0's GRPO, so 3 runs."""
    return [Group("q5", "GRPO with k-step-stale rollouts, k=4/16/64", 3, grpo_steps * u.grpo_step_s, 5)]


def group_usd(x: Group, p: Prices, eval_s: float, ckpts: int | None = None) -> tuple[float, float, int]:
    """(train-GPU + CPU $, eval $, evals) of a whole group."""
    k = ckpts if (ckpts is not None and x.ckpt_cut) else x.n_evals
    train = x.count * ((p.train_usd(x.train_s) if x.train_gpu else 0.0) + x.cpu_usd)
    return train, x.count * k * p.eval_usd(eval_s), x.count * k


def cost(groups: list[Group], p: Prices, eval_s: float, ckpts: int | None = None) -> dict:
    """$ and counts; `ckpts` overrides n_evals of groups that evaluate evenly spaced checkpoints."""
    train = evals = 0.0
    n_runs = n_evals = 0
    for x in groups:
        t, e, k = group_usd(x, p, eval_s, ckpts)
        train, evals, n_evals = train + t, evals + e, n_evals + k
        n_runs += x.count if x.is_run else 0
    return {"runs": n_runs, "evals": n_evals, "train_usd": train, "eval_usd": evals, "total_usd": train + evals}


def project(u: Units, p: Prices, grpo_steps: int = 400, teacher: str = DEFAULT_TEACHER) -> dict:
    g = grid(u, p, grpo_steps, teacher)
    e = u.eval_s
    stage1 = [x for x in g if x.arm in ("q0", "q1")]
    rest = [x for x in g if x.arm not in ("q0", "q1")]
    st = staleness(u, p, grpo_steps)
    out = {
        "full_grid": cost(g, p, e),
        "stage1_q0_q1": cost(stage1, p, e),
        "remainder_q2_q4": cost(rest, p, e),
        "staleness_optional": cost(st, p, e),
        "groups": [],
    }
    for x in g:
        t, ev, _ = group_usd(x, p, e)
        t3, ev3, _ = group_usd(x, p, e, 3)
        out["groups"].append({"arm": x.arm, "label": x.label, "runs": x.count, "train_h_each": x.train_s / 3600,
                              "evals_each": x.n_evals, "estimated": x.estimated, "train_usd": t, "eval_usd": ev,
                              "total_usd_3ckpts": t3 + ev3})
    # Spec §6 cut order, cumulative: (1) 3 ckpts/run, (2) drop extra Q3 seeds, (3) drop p=0.4, (4) drop LoRA.
    cuts, cur = [("none (full grid, 5 ckpts/run)", cost(g, p, e))], list(g)
    cuts.append(("(1) evaluate 3 checkpoints per run", cost(cur, p, e, 3)))
    for label, tag in (("(2) drop extra Q3 seeds", "q3_extra_seeds"), ("(3) drop p=0.4", "p0.4"),
                       ("(4) drop LoRA", "lora")):
        cur = [x for x in cur if tag not in x.tags]
        cuts.append((label, cost(cur, p, e, 3)))
    out["cut_order"] = [{"cut": c, **v} for c, v in cuts]
    out["stage1_q0_q1_3ckpts"] = cost(stage1, p, e, 3)
    out["remainder_q2_q4_3ckpts"] = cost(rest, p, e, 3)
    out["staleness_optional_3ckpts"] = cost(st, p, e, 3)
    out["teacher"] = teacher
    out["stage1_by_teacher"] = {}
    for t in TEACHERS:
        s1 = [x for x in grid(u, p, grpo_steps, t) if x.arm in ("q0", "q1")]
        out["stage1_by_teacher"][t] = {"5ckpts": cost(s1, p, e), "3ckpts": cost(s1, p, e, 3)}
    out["estimated_share_full_grid"] = sum(x["train_usd"] + x["eval_usd"] for x in out["groups"] if x["estimated"]) / (
        out["full_grid"]["total_usd"] or 1.0)
    return out


# ---------------------------------------------------------------- I/O and printing

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def fetch(results: Path, train_times: Path) -> None:
    """Export on the Volume, then download results.jsonl, train_times.jsonl and dataset_stats.jsonl (the last next
    to results.jsonl; CPU functions only)."""
    import modal

    from modal_app import app, export_results, read_files

    with modal.enable_output(), app.run():
        export_results.remote()
        files = read_files.remote(["results.jsonl", "train_times.jsonl", "dataset_stats.jsonl"])
    for name, dst in (("results.jsonl", results), ("train_times.jsonl", train_times),
                      ("dataset_stats.jsonl", results.parent / "dataset_stats.jsonl")):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(files.get(name, ""))


def _usd(x: float) -> str:
    return f"${x:,.2f}"


def report(rows, train_times, p: Prices, spent: float, budget: float, grpo_steps: int,
           teacher: str = DEFAULT_TEACHER) -> dict:
    lines = []
    w = lines.append
    w(f"prices: train GPU {_usd(p.train_hr)}/h, eval GPU {_usd(p.eval_hr)}/h, CPU+memory overhead "
      f"{p.overhead:.0%} of GPU $, {p.container_s:.0f} s per container outside the timed code; CPU-only jobs "
      f"${p.cpu_core_hr:.3f}/core-h + ${p.mem_gib_hr:.3f}/GiB-h")
    w(f"ASSUMPTIONS (not measured by the pilot; groups using them are tagged est.): {p.assumptions()}")
    w("")
    w("per-row eval cost (gpu_seconds = scored time inside eval_ckpt / base_eval)")
    w(f"  {'run_id':40s} {'step':>5s} {'gpu_s':>7s} {'$':>7s}")
    for r in rows:
        if r.get("smoke") or r.get("gpu_seconds") is None:
            continue
        w(f"  {r['run_id']:40s} {r['step']:>5d} {r['gpu_seconds']:>7.0f} {p.eval_usd(r['gpu_seconds']):>7.3f}")
    w("")
    w("per-run training cost (wall seconds inside the train function)")
    for t in train_times:
        cfg = t.get("config") or {}
        if cfg.get("smoke"):
            continue
        w(f"  {t['run_id']:40s} {cfg.get('method', '?'):6s} {t['wall_seconds']:>7.0f} s "
          f"{p.train_usd(t['wall_seconds']):>7.2f} $")
    u = measure(rows, train_times)
    sft_run = sft_steps(N_TRAIN) * u.sft_step_s
    grpo_run = grpo_steps * u.grpo_step_s
    w("")
    w(f"units: eval {u.eval_s:.0f} s/ckpt -> {_usd(p.eval_usd(u.eval_s))} per checkpoint eval; "
      f"SFT {u.sft_step_s:.2f} s/step -> {_usd(p.train_usd(sft_run))} per SFT run ({sft_steps(N_TRAIN)} steps); "
      f"GRPO {u.grpo_step_s:.1f} s/step -> {_usd(p.train_usd(grpo_run))} per GRPO run ({grpo_steps} steps)")
    proj = project(u, p, grpo_steps, teacher)
    w("")
    w(f"{'projection':44s} {'runs':>5s} {'evals':>6s} {'train $':>9s} {'eval $':>8s} {'total $':>9s}")

    def row(label, c):
        w(f"  {label:42s} {c['runs']:>5d} {c['evals']:>6d} {c['train_usd']:>9.2f} {c['eval_usd']:>8.2f} "
          f"{c['total_usd']:>9.2f}")

    w(f"(Q0 as amended: SFT(teacher) + GRPO, 5 runs each; teacher for the grid rows: {proj['teacher']})")
    row("full grid (5 ckpts/run)", proj["full_grid"])
    row("  stage 1 = Q0 + Q1", proj["stage1_q0_q1"])
    row("  remainder = Q2-Q4", proj["remainder_q2_q4"])
    row("optional async-staleness (3 GRPO runs)", proj["staleness_optional"])
    row("  STAGE 1 (Q0+Q1) at 3 ckpts/run", proj["stage1_q0_q1_3ckpts"])
    row("  remainder at 3 ckpts/run", proj["remainder_q2_q4_3ckpts"])
    row("  staleness at 3 ckpts/run", proj["staleness_optional_3ckpts"])
    w("")
    w("STAGE 1 (Q0+Q1) by teacher choice (ESTIMATES: teacher generation and SFT(teacher) rows are not measured)")
    for t, c in proj["stage1_by_teacher"].items():
        row(f"  {t}, 5 ckpts/run", c["5ckpts"])
        row(f"  {t}, 3 ckpts/run", c["3ckpts"])
    w("")
    w(f"spec §6 cut order (cumulative; spec gate ${budget:.0f}; already spent {_usd(spent)})")
    for c in proj["cut_order"]:
        w(f"  {c['cut']:42s} {c['runs']:>5d} {c['evals']:>6d} {c['train_usd']:>9.2f} {c['eval_usd']:>8.2f} "
          f"{c['total_usd']:>9.2f}  running total incl. spent {_usd(c['total_usd'] + spent)}  [grid {'<=' if c['total_usd'] <= budget else '>'} ${budget:.0f}]")
    w("")
    w(f"by group (full grid; est. = rests on an assumption above; estimated groups are "
      f"{proj['estimated_share_full_grid']:.0%} of the full-grid $)")
    for x in proj["groups"]:
        tag = "est." if x["estimated"] else "    "
        w(f"  {tag} {x['arm']} {x['label']:48s} x{x['runs']:<2d} train {x['train_h_each']:.2f} h each  "
          f"evals {x['evals_each']}  train {_usd(x['train_usd'])}  eval {_usd(x['eval_usd'])}  "
          f"(3 ckpts: {_usd(x['total_usd_3ckpts'])})")
    print("\n".join(lines))
    return {"prices": dataclasses.asdict(p), "units": dataclasses.asdict(u), "per_sft_run_usd": p.train_usd(sft_run),
            "per_grpo_run_usd": p.train_usd(grpo_run), "per_eval_usd": p.eval_usd(u.eval_s),
            "spent_usd": spent, **proj}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "outputs" / "results.jsonl")
    ap.add_argument("--train-times", type=Path, default=ROOT / "outputs" / "train_times.jsonl")
    ap.add_argument("--fetch", action="store_true", help="export + download the result files from the Volume first")
    ap.add_argument("--spent", type=float, default=0.0, help="$ already spent, added to the running totals")
    ap.add_argument("--budget", type=float, default=SPEC_BUDGET, help="gate for the cut order (spec: 75)")
    ap.add_argument("--grpo-steps", type=int, default=400, help="GRPO steps per grid run (spec: about 400)")
    ap.add_argument("--teacher", choices=sorted(TEACHERS), default=DEFAULT_TEACHER,
                    help="teacher assumed for the grid rows (stage 1 is printed for both)")
    ap.add_argument("--json", type=Path, help="also write the numbers here")
    a = ap.parse_args(argv)
    if a.fetch:
        fetch(a.results, a.train_times)
    out = report(read_jsonl(a.results), read_jsonl(a.train_times), Prices.from_env(), a.spent, a.budget,
                 a.grpo_steps, a.teacher)
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        if not isinstance(e.code, int) and e.code is not None:
            print(e.code, file=sys.stderr)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
