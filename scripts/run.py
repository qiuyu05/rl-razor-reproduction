"""Train and evaluate one config on Modal.

    uv run python scripts/run.py <config.yaml> [--smoke] [--eval-only] [--require-prepared]

Steps: CPU status check (finished training and already-scored checkpoints are skipped) -> general pool
pre-step if the config needs it -> train (skipped with --eval-only) -> base_eval (builds the base KL cache and base row once, before the
parallel evals; ruling R21) -> eval_ckpt over every checkpoint in parallel -> export /vol/results.jsonl.
Several configs in parallel: use scripts/launch.py, which builds the pool and base row once and then runs
this script per config with --require-prepared (ruling R36).
GPU types: RLRAZOR_GPU for training (default A100-40GB), RLRAZOR_EVAL_GPU for evaluation (default L4).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # modal_app.py at the repo root

from rlrazor.config import config_to_yaml, load_config, run_id  # noqa: E402
from rlrazor.pipeline import needs_general_pool  # noqa: E402


def unprepared(cfg, st: dict) -> list[str]:
    """Shared state (built once by launch.py, R36) that a config needs but the Volume lacks."""
    missing = []
    if cfg.method == "teacher_data":
        return missing  # a data job: no base row, pool or checkpoints involved
    if st["base_row"] is None:
        missing.append("the base row")
    if st["index"] is None and needs_general_pool(cfg) and not st["general_pool"]:
        missing.append("the general pool")
    if st["index"] is None and cfg.method == "teacher_sft" and not st.get("teacher_data_done"):
        missing.append("the teacher data set (run the teacher_data config first)")
    return missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config")
    ap.add_argument("--smoke", action="store_true", help="force smoke: true (2 steps, tiny evals)")
    ap.add_argument("--eval-only", action="store_true", help="skip training; evaluate existing checkpoints")
    ap.add_argument("--require-prepared", action="store_true",
                    help="never build the general pool or the base row (launch.py builds them once, R36); "
                         "exit before any GPU work if they are missing")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    if a.smoke and not cfg.smoke:
        cfg = dataclasses.replace(cfg, smoke=True)
    cfg_yaml = config_to_yaml(cfg)
    rid = run_id(cfg)
    print(f"run_id {rid}  method {cfg.method}  smoke {cfg.smoke}  "
          f"train gpu {os.environ.get('RLRAZOR_GPU', 'A100-40GB')}  eval gpu {os.environ.get('RLRAZOR_EVAL_GPU', 'L4')}")

    import modal

    from modal_app import (app, base_eval, build_general_pool, eval_ckpt, export_results, run_status, teacher_data,
                           train, wise_run)

    timings = {}
    rows = []
    with modal.enable_output(), app.run():
        # Cheap CPU look-up first, so a relaunch never starts GPU containers for finished work.
        st = run_status.remote(cfg_yaml)
        if a.eval_only and st["index"] is None:
            raise SystemExit(f"{rid}: not trained yet (no ckpts.json)")
        missing = unprepared(cfg, st) if a.require_prepared else []
        if missing:
            raise SystemExit(f"{rid}: --require-prepared but {' and '.join(missing)} missing: run scripts/launch.py")
        if cfg.method == "teacher_data":  # a data job (R42/R43): no checkpoints, no evals
            if st["teacher_data_done"]:
                print("teacher data: done already")
            else:
                t = time.time()
                summary = teacher_data.remote(cfg_yaml)
                print(f"teacher data: {json.dumps(summary, sort_keys=True)}  ({time.time() - t:.0f} s)")
            print(f"results.jsonl rows {export_results.remote()}")
            return 0
        if st["index"] is None:
            if needs_general_pool(cfg) and not st["general_pool"]:
                t = time.time()
                print(f"general pool: {build_general_pool.remote()} rows")
                timings["general_pool_s"] = time.time() - t
            t = time.time()
            if cfg.wise_alpha > 0:  # WiSE-FT: CPU interpolation, no training GPU (spec Q4)
                wise_run.remote(cfg_yaml)
                timings["wise_s"] = time.time() - t
            else:
                train.remote(cfg_yaml)
                timings["train_s"] = time.time() - t
            st = run_status.remote(cfg_yaml)
        ckpts, total = st["index"]["ckpts"], st["index"]["total_steps"]
        todo = [(step, path) for step, path in ckpts if step not in st["scored_steps"]]
        print(f"{len(ckpts)} checkpoints, {len(ckpts) - len(todo)} already scored")
        base = st["base_row"]
        if base is None:  # builds the KL cache, then the base row, before the parallel evals (R21)
            t = time.time()
            base = base_eval.remote(smoke=cfg.smoke)
            timings["base_eval_s"] = time.time() - t
        if todo:
            t = time.time()
            rows = list(eval_ckpt.starmap([(cfg_yaml, path, step, total) for step, path in todo]))
            timings["eval_s"] = time.time() - t
        n = export_results.remote()

    for (step, path), row in zip(todo, rows):
        if row is None:
            print(f"step {step}: already scored")
        else:
            keys = ("gsm8k_acc", "kl_fwd", "forgetting_mean", "gpu_seconds")
            print(f"step {step} ({path}): " + json.dumps({k: row[k] for k in keys} | {"bench": row["bench"]}))
    print(f"base gsm8k_acc {base['gsm8k_acc']:.4f}  kl_fwd {base['kl_fwd']:.2e}")
    print(f"timings {json.dumps({k: round(v, 1) for k, v in timings.items()})}  results.jsonl rows {n}")
    return 0


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        if not isinstance(e.code, int) and e.code is not None:
            print(e.code, file=sys.stderr)
    except BaseException:
        import traceback

        traceback.print_exc()
    finally:
        # Ruling R9: datasets/pyarrow can hang at interpreter exit; always hard-exit after flushing.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
