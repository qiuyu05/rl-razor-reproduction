"""Run orchestration on a volume root: train one config, score checkpoints, build the base row.

modal_app.py wraps these with GPU functions and the `rlrazor` Volume mounted at /vol; keeping the logic here
(paths passed in, trainers imported lazily) lets it be tested on CPU with a temp dir as the volume.

Volume layout under `root`:
  ckpts/<run_id>/step-<n>/         full bf16 model dirs; ckpts/<run_id>/ckpts.json lists them when training is done
  ckpts/wise/<run_id>__step-<n>__a<alpha>/   WiSE-FT interpolations
  cache/general_pool/              Tulu-3 single-turn pool (save_to_disk; ruling R9)
  cache/kl/, cache/kl_smoke/       base KL samples (real / smoke rows)
  cache/<run_id>/                  dataset_stats.json (RFT / self-rewrite), train_time.json
  cache/teacher_data/              teacher SFT set (R42/R43): dataset/ (save_to_disk), dataset_stats.json,
                                   done.json written LAST (scripts/launch.py waits for it); cache/teacher_data_smoke/
  results/                         one JSON row per (run_id, step) (ruling R5); results.jsonl is its export
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

from rlrazor.config import RunConfig, dump_config, run_id
from rlrazor.results import append_row, export_jsonl, get_row

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BASE_RUN_ID = "base"
BASE_SMOKE_RUN_ID = "base-smoke"

RFT_K_OFFLINE = 8  # spec Q1: k=8 samples per prompt from the base model
RFT_K_ITER = 4  # spec Q1: k=4 per prompt from the current checkpoint, each round
RFT_K_DISTILL = 8  # RL-distilled SFT: same k as offline RFT (spec leaves k unstated)
DISTILL_PROMPTS = 2000  # spec Q4: 2,000 Tulu-3 prompts
# GKD materializes full-vocab (152k) fp32 log-probs for student and teacher over prompt + 512 sampled tokens; the
# Tulu pool bounds response length but not prompt length, and a ~900-token prompt OOMed an A100-40GB at batch 4
# in stage 2. Distillation prompts are drawn from pool rows whose prompt is at most this many tokens.
DISTILL_MAX_PROMPT_TOKENS = 256
POOL_SEED = 0  # the general pool is one fixed subset; per-run selection from it is keyed on the run seed
SMOKE_TRAIN_ROWS = 64  # smoke runs train (and, for RFT/rewrite, generate) on this many GSM8K rows
SMOKE_DISTILL_PROMPTS = 16
# Teacher data job (method "teacher_data"): vLLM holds the whole GPU (nothing else runs in that process); the
# prompt (~150 tokens) + 512 new tokens fit well inside 2,048, which keeps vLLM's minimum KV requirement small.
TEACHER_GPU_UTIL = 0.90
TEACHER_MAX_MODEL_LEN = 2048
TEACHER_DATA_DONE_REL = "cache/teacher_data/done.json"  # = scripts/launch.py TEACHER_DATA_DONE


def ckpt_dir(root, rid: str) -> Path:
    return Path(root) / "ckpts" / rid


def resolve_model(root, ref: str | None) -> str:
    """`init_from` / WiSE reference -> model path: "base" (or None) is the hub base model, else ckpts-relative."""
    if ref is None or ref == "base":
        return BASE_MODEL
    p = Path(ref)
    return str(p if p.is_absolute() else Path(root) / "ckpts" / p)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    os.replace(tmp, path)


# ---------------------------------------------------------------- general (Tulu-3) pool, ruling R9

def _build_pool_main(out_dir: str) -> None:
    """Child-process entry: stream the pool, save it, then hard-exit (streaming + pyarrow can hang at exit)."""
    from rlrazor.data.mixing import load_general_pool

    tmp = out_dir + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    load_general_pool(seed=POOL_SEED).save_to_disk(tmp)
    os.replace(tmp, out_dir)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def general_pool_path(root) -> Path:
    return Path(root) / "cache" / "general_pool"


def needs_general_pool(cfg: RunConfig) -> bool:
    return cfg.mix_f > 0 or cfg.method == "distill"


def short_prompt_rows(pool, max_tokens: int, tokenizer=None):
    """Pool rows whose user prompt (`question`) is at most `max_tokens` tokens (base model tokenizer)."""
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    return pool.filter(lambda q: len(tokenizer.encode(q)) <= max_tokens, input_columns="question")


def general_pool(root, build: bool = False):
    """The cached Tulu-3 pool. Only the dedicated pre-step (build=True, a CPU container) creates it;
    training raises if it is missing, so parallel GPU runs never stream it concurrently."""
    from datasets import load_from_disk

    path = general_pool_path(root)
    if not path.exists():
        if not build:
            raise RuntimeError(f"{path} missing: run the build_general_pool step before training this config")
        path.parent.mkdir(parents=True, exist_ok=True)
        code = f"from rlrazor.pipeline import _build_pool_main; _build_pool_main({str(path)!r})"
        subprocess.run([sys.executable, "-c", code], check=True, timeout=3600)
    return load_from_disk(str(path))


# ---------------------------------------------------------------- training

def verify_bf16_ckpt(path) -> None:
    """Saved checkpoints must hold bf16 weights (ruling R13) and a tokenizer."""
    from safetensors import safe_open

    files = sorted(Path(path).glob("*.safetensors"))
    if not files or not (Path(path) / "tokenizer_config.json").exists():
        raise RuntimeError(f"{path}: missing weights or tokenizer")
    dtypes = set()
    for f in files:
        with safe_open(str(f), framework="pt") as sf:
            dtypes |= {str(sf.get_slice(k).get_dtype()) for k in sf.keys()}
    if dtypes != {"BF16"}:
        raise RuntimeError(f"{path}: checkpoint dtypes {sorted(dtypes)}, expected only BF16")


def _rft_iter(cfg: RunConfig, ds, out: Path, stats_path: Path, backend: str) -> list[tuple[int, str]]:
    """`rounds` x (RFT set from the current checkpoint -> 1 epoch of SFT); one checkpoint per round (step = round)."""
    from rlrazor.data.rft import build_rft_set
    from rlrazor.train.sft import train_sft

    round_cfg = dataclasses.replace(cfg, epochs=1, n_ckpts=1)
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    current, saved = BASE_MODEL, []
    for r in range(1, cfg.rounds + 1):
        dst = out / f"step-{r}"
        if not dst.exists():  # resumable after preemption: finished rounds are kept
            rft_ds, st = build_rft_set(current, ds, k=RFT_K_ITER, seed=cfg.seed * 1000 + r, backend=backend)
            if len(rft_ds) == 0:
                raise RuntimeError(f"round {r}: no prompt had a correct sample")
            stats[f"round_{r}"] = st
            _write_json(stats_path, stats)
            tmp = out / f"round-{r}"
            shutil.rmtree(tmp, ignore_errors=True)
            pairs = train_sft(round_cfg, rft_ds, str(tmp), model_path=current)
            os.replace(pairs[-1][1], dst)
            shutil.rmtree(tmp, ignore_errors=True)
        current = str(dst)
        saved.append((r, current))
    return saved


# ---------------------------------------------------------------- teacher data (R42/R43)

def teacher_data_dir(root, smoke: bool) -> Path:
    return Path(root) / "cache" / ("teacher_data_smoke" if smoke else "teacher_data")


def teacher_data_done(root, smoke: bool) -> bool:
    return (teacher_data_dir(root, smoke) / "done.json").exists()


def build_teacher_data(cfg: RunConfig, root, backend: str = "vllm") -> dict:
    """The one-off teacher SFT set (paper App. B.1): adaptive sampling from the teacher over the GSM8K train
    prompts (smoke: the first SMOKE_TRAIN_ROWS), keep 1 correct per prompt. done.json is written only after the
    dataset and its stats are complete; until then teacher_sft runs refuse to start. Idempotent."""
    from rlrazor.data import teacher as T
    from rlrazor.data.gsm8k import load_gsm8k
    from rlrazor.gen import Sampler

    if cfg.method != "teacher_data":
        raise ValueError(f"build_teacher_data needs method 'teacher_data', got {cfg.method!r}")
    d = teacher_data_dir(root, cfg.smoke)
    done = d / "done.json"
    if done.exists():
        return json.loads(done.read_text())
    rid = run_id(cfg)
    dump_config(cfg, ckpt_dir(root, rid) / "config.yaml")  # export() pairs train_time.json with the config
    t0 = time.perf_counter()
    ds = load_gsm8k("train", limit=SMOKE_TRAIN_ROWS if cfg.smoke else None)
    with Sampler(T.TEACHER_MODEL, T.TEMPERATURE, T.MAX_TOKENS, backend=backend,
                 gpu_memory_utilization=TEACHER_GPU_UTIL, max_model_len=TEACHER_MAX_MODEL_LEN) as sample:
        out, stats = T.build_teacher_set(ds, sample, seed=cfg.seed)
    stats.update(teacher=T.TEACHER_MODEL, run_id=rid, wall_seconds=time.perf_counter() - t0)
    tmp, dst = d / "dataset.tmp", d / "dataset"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(dst, ignore_errors=True)
    out.save_to_disk(str(tmp))
    os.replace(tmp, dst)
    _write_json(d / "dataset_stats.json", stats)
    _write_json(Path(root) / "cache" / rid / "train_time.json", {"wall_seconds": time.perf_counter() - t0})
    summary = {k: stats[k] for k in ("teacher", "run_id", "n_prompts", "n_rows", "keep_rate", "samples_per_prompt")}
    _write_json(done, summary)  # last: the launcher and teacher_sft runs key on it
    return summary


def load_teacher_set(root, smoke: bool):
    from datasets import load_from_disk

    if not teacher_data_done(root, smoke):
        raise RuntimeError(f"{teacher_data_dir(root, smoke) / 'done.json'} missing: run the teacher_data job first")
    return load_from_disk(str(teacher_data_dir(root, smoke) / "dataset"))


def train_run(cfg: RunConfig, root, backend: str = "vllm", use_vllm: bool = True) -> list[tuple[int, str]]:
    """Train one config; checkpoints go to <root>/ckpts/<run_id>/step-<n>. Idempotent once ckpts.json exists."""
    from rlrazor.data.gsm8k import load_gsm8k

    if cfg.method == "teacher_data":
        raise ValueError("teacher_data produces a data set, not checkpoints: use build_teacher_data")
    if cfg.wise_alpha > 0:
        return wise_run(cfg, root)
    rid = run_id(cfg)
    out = ckpt_dir(root, rid)
    index = out / "ckpts.json"
    if index.exists():
        return [tuple(x) for x in json.loads(index.read_text())["ckpts"]]
    if cfg.method == "teacher_sft" and not teacher_data_done(root, cfg.smoke):
        load_teacher_set(root, cfg.smoke)  # raises before anything is written
    if needs_general_pool(cfg) and not general_pool_path(root).exists():
        general_pool(root)  # raises: the pool is built by its own CPU step, never inside a GPU run
    dump_config(cfg, out / "config.yaml")
    cache = Path(root) / "cache" / rid
    stats_path = cache / "dataset_stats.json"
    t0 = time.perf_counter()
    ds = load_gsm8k("train", limit=SMOKE_TRAIN_ROWS if cfg.smoke else None)
    m = cfg.method

    if m == "rft_iter":
        saved = _rft_iter(cfg, ds, out, stats_path, backend)
        total = cfg.rounds
    else:
        if m == "grpo":
            from rlrazor.train.grpo import train_grpo

            saved = train_grpo(cfg, ds, str(out), use_vllm=use_vllm)
        elif m == "distill":
            from rlrazor.train.distill import train_distill

            n = SMOKE_DISTILL_PROMPTS if cfg.smoke else DISTILL_PROMPTS
            pool = short_prompt_rows(general_pool(root), DISTILL_MAX_PROMPT_TOKENS)
            pool = pool.shuffle(seed=cfg.seed).select(range(n))
            # General prompts as a plain user turn (Qwen's default system prompt), not the GSM8K boxed instruction.
            prompts = pool.map(lambda q: {"prompt": [{"role": "user", "content": q}]},
                               input_columns="question", remove_columns=pool.column_names)
            student = resolve_model(root, cfg.init_from)
            saved = train_distill(cfg, student, BASE_MODEL, prompts, str(out))
        else:
            from rlrazor.train.sft import train_sft

            if m in ("sft", "lora_sft", "teacher_sft"):
                if m == "teacher_sft":  # SFT on the teacher set (R42/R43); Q2/Q3/Q4 variants add noise/mix/LoRA
                    ds = load_teacher_set(root, cfg.smoke)
                if cfg.noise_p > 0:
                    from rlrazor.data.noise import corrupt_labels

                    ds = corrupt_labels(ds, cfg.noise_p, cfg.seed)
                if cfg.mix_f > 0:
                    from rlrazor.data.mixing import build_mixture

                    ds = build_mixture(ds, general_pool(root), cfg.mix_f, cfg.seed)
            elif m in ("rft_offline", "rl_distill_sft"):
                from rlrazor.data.rft import build_rft_set

                if m == "rl_distill_sft" and cfg.init_from in (None, "base"):
                    raise ValueError("rl_distill_sft needs init_from = '<grpo run_id>/step-<n>'")
                src = resolve_model(root, cfg.init_from) if m == "rl_distill_sft" else BASE_MODEL
                ds, st = build_rft_set(src, ds, k=RFT_K_OFFLINE if m == "rft_offline" else RFT_K_DISTILL,
                                       seed=cfg.seed, backend=backend)
                _write_json(stats_path, {**st, "source": src, "n_rows": len(ds)})
            elif m == "self_rewrite":
                from rlrazor.data.rewrite import build_rewrite_set

                ds, fb = build_rewrite_set(BASE_MODEL, ds, backend=backend)
                _write_json(stats_path, {"fallback_rate": fb, "n_rows": len(ds)})
            else:
                raise ValueError(f"unhandled method {m!r}")
            saved = train_sft(cfg, ds, str(out))
        total = max(step for step, _ in saved)

    for _, path in saved:
        verify_bf16_ckpt(path)
    meta = {}
    if (out / "diverged.json").exists():  # GRPO dead-run stop (R50): label the run, keep the planned length
        meta["diverged"] = json.loads((out / "diverged.json").read_text())
        from rlrazor.train import SMOKE_STEPS

        total = SMOKE_STEPS if cfg.smoke else cfg.grpo_steps  # the planned length
    _write_json(cache / "train_time.json", {"wall_seconds": time.perf_counter() - t0})
    _write_json(index, {"total_steps": total, "ckpts": [list(x) for x in saved], **meta})
    return [tuple(x) for x in saved]


def run_status(root, cfg: RunConfig) -> dict:
    """What already exists for a config, so a relaunch can skip GPU work client-side (no GPU needed)."""
    index = ckpt_dir(root, run_id(cfg)) / "ckpts.json"
    idx = json.loads(index.read_text()) if index.exists() else None
    results = Path(root) / "results"
    rid = run_id(cfg)
    return {
        "index": idx,
        "scored_steps": [] if idx is None else [s for s, _ in idx["ckpts"] if get_row(results, rid, s) is not None],
        "base_row": get_row(results, BASE_SMOKE_RUN_ID if cfg.smoke else BASE_RUN_ID, 0),
        "general_pool": general_pool_path(root).exists(),
        "teacher_data_done": teacher_data_done(root, cfg.smoke),
    }


def ckpt_index(root, cfg: RunConfig) -> dict:
    index = ckpt_dir(root, run_id(cfg)) / "ckpts.json"
    if not index.exists():
        raise FileNotFoundError(f"{index}: run not trained yet")
    return json.loads(index.read_text())


# ---------------------------------------------------------------- evaluation

def lm_eval_raw_dir(root, rid: str, step: int) -> Path:
    """Raw lm-eval results JSONs of one scored checkpoint."""
    return Path(root) / "cache" / "lm_eval_raw" / rid / f"step-{step}"


def kl_cache_dir(root, smoke: bool) -> Path:
    return Path(root) / "cache" / ("kl_smoke" if smoke else "kl")


RUNTIME_PACKAGES = ("torch", "transformers", "trl", "vllm", "lm-eval", "datasets", "peft", "accelerate")


def gpu_slug() -> str:
    import re

    import torch

    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return re.sub(r"[^A-Za-z0-9.-]+", "_", name).strip("_")


def runtime_versions() -> dict:
    """Package versions (and GPU) of the running image; the Modal image is the authoritative runtime (R32)."""
    import importlib.metadata
    import platform

    import torch

    out = {"python": platform.python_version(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    for pkg in RUNTIME_PACKAGES:
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def base_eval(root, smoke: bool = False, backend: str = "vllm") -> dict:
    """Build the base KL cache (dedicated step, ruling R21), then score the base model once as step 0.

    When it creates the row it also records the runtime package versions to
    cache/runtime_versions/<gpu>.json (R32).

    The row's run_id is "base" ("base-smoke" for smoke rows, whose base_row must also be smoke).
    """
    from rlrazor.eval.run_eval import ensure_base_kl_samples, eval_vllm_env, evaluate_checkpoint

    results = Path(root) / "results"
    rid = BASE_SMOKE_RUN_ID if smoke else BASE_RUN_ID
    row = get_row(results, rid, 0)
    if row is not None:
        return row
    _write_json(Path(root) / "cache" / "runtime_versions" / f"{gpu_slug()}.json", runtime_versions())
    cache = kl_cache_dir(root, smoke)
    with eval_vllm_env():  # the T=1 base samples are generated by vLLM: same settings as every eval (R38)
        ensure_base_kl_samples(cache, BASE_MODEL, backend)
    cfg = RunConfig(arm="base", method="sft", lr=0.0, smoke=smoke)
    with tempfile.TemporaryDirectory(prefix="base-row-") as tmp:
        row = evaluate_checkpoint(cfg, BASE_MODEL, 0, tmp, None, smoke=smoke, total_steps=1,
                                  cache_dir=cache, base_model=BASE_MODEL, backend=backend,
                                  raw_dir=lm_eval_raw_dir(root, rid, 0))
    row.update(run_id=rid, arm="base", method="base", lr=0.0, init_from=None)
    append_row(results, row)
    return row


def eval_ckpt(cfg: RunConfig, ckpt_path: str, step: int, total_steps: int, root,
              backend: str = "vllm") -> dict | None:
    from rlrazor.eval.run_eval import evaluate_checkpoint

    rid = BASE_SMOKE_RUN_ID if cfg.smoke else BASE_RUN_ID
    base_row = get_row(Path(root) / "results", rid, 0)
    if base_row is None:
        raise RuntimeError(f"no {rid!r} row yet: run base_eval(smoke={cfg.smoke}) first")
    index = ckpt_dir(root, run_id(cfg)) / "ckpts.json"
    div = json.loads(index.read_text()).get("diverged") if index.exists() else None
    extra = {"diverged": div is not None, "diverged_step": None if div is None else div["step"]}
    return evaluate_checkpoint(cfg, ckpt_path, step, Path(root) / "results", base_row, smoke=cfg.smoke,
                               total_steps=total_steps, cache_dir=kl_cache_dir(root, cfg.smoke),
                               base_model=BASE_MODEL, backend=backend,
                               raw_dir=lm_eval_raw_dir(root, run_id(cfg), step), extra=extra)


def wise(root, base: str, ft: str, alpha: float) -> str:
    """WiSE-FT interpolation (1-alpha)*base + alpha*ft; refs as in resolve_model. Returns the output dir."""
    from rlrazor.eval.wise import save_interpolated

    ft_path = Path(resolve_model(root, ft))
    out = Path(root) / "ckpts" / "wise" / f"{ft_path.parent.name}__{ft_path.name}__a{alpha:g}"
    if not (out / "config.json").exists():
        tmp = out.with_name(out.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        save_interpolated(resolve_model(root, base), str(ft_path), alpha, tmp)
        os.replace(tmp, out)
    verify_bf16_ckpt(out)
    return str(out)


def wise_run(cfg: RunConfig, root) -> list[tuple[int, str]]:
    """WiSE-FT eval-only config (wise_alpha > 0, spec Q4): interpolate base and `init_from` (no training) and
    index the result as this run's single checkpoint, at the source checkpoint's step, so run.py / launch.py
    evaluate it like any other run. CPU only. Idempotent once ckpts.json exists."""
    if cfg.wise_alpha <= 0 or cfg.init_from in (None, "base"):
        raise ValueError("a WiSE-FT config needs wise_alpha > 0 and init_from = '<run_id>/step-<n>'")
    out = ckpt_dir(root, run_id(cfg))
    index = out / "ckpts.json"
    if index.exists():
        return [tuple(x) for x in json.loads(index.read_text())["ckpts"]]
    src_run, src_step = str(cfg.init_from).split("/")
    step = int(src_step.removeprefix("step-"))
    src_index = ckpt_dir(root, src_run) / "ckpts.json"
    if not src_index.exists():
        raise RuntimeError(f"{src_index} missing: the WiSE-FT source run is not trained")
    total = json.loads(src_index.read_text())["total_steps"]
    path = wise(root, "base", cfg.init_from, cfg.wise_alpha)
    dump_config(cfg, out / "config.yaml")
    _write_json(index, {"total_steps": total, "ckpts": [[step, path]], "wise_source": cfg.init_from})
    return [(step, path)]


def export(root) -> int:
    """results/ -> results.jsonl, every dataset_stats.json -> dataset_stats.jsonl (ruling R29) and every
    train_time.json -> train_times.jsonl (with the run's config). Returns the results row count."""
    root = Path(root)
    n = export_jsonl(root / "results", root / "results.jsonl")
    lines = []
    for p in sorted((root / "cache").glob("*/dataset_stats.json")):
        lines.append(json.dumps({"run_id": p.parent.name, **json.loads(p.read_text())}, sort_keys=True))
    (root / "dataset_stats.jsonl").write_text("".join(x + "\n" for x in lines))
    # Training time per run (wall seconds inside the train function) + its config, for scripts/cost_report.py.
    lines = []
    for p in sorted((root / "cache").glob("*/train_time.json")):
        cfg_path = ckpt_dir(root, p.parent.name) / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else None
        lines.append(json.dumps({"run_id": p.parent.name, **json.loads(p.read_text()), "config": cfg},
                                sort_keys=True))
    (root / "train_times.jsonl").write_text("".join(x + "\n" for x in lines))
    return n

