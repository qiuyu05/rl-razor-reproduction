"""Forgetting suite: six lm-eval benchmarks, run through the lm-eval CLI on vLLM.

Task names and metric keys were checked against the installed lm-eval 0.4.13 task YAMLs:
- hellaswag:      task `hellaswag`, emits `acc_norm,none`.
- mmlu:           group `mmlu` (57 subtasks); the group row reports `acc,none`, the
                  size-weighted mean over subtasks. `--limit` is applied per leaf task
                  (evaluator.simple_evaluate -> get_sample_size runs once per subtask), so
                  `--limit 10` means 10 questions per subject.
- truthfulqa:     task `truthfulqa_mc2`; its process_results returns `acc`, so the key is
                  `acc,none` (there is no `mc2` metric key).
- winogrande:     task `winogrande`, `acc,none`.
- ifeval:         task `ifeval`, prompt-level strict = `prompt_level_strict_acc,none`.
- humaneval:      task `humaneval_instruct` is the chat variant (instruction prompt plus a
                  `gen_prefix`; plain `humaneval` is raw-completion). pass@1 is expanded from
                  `pass_at_k` and reported under the filter name: `pass@1,create_test`.
                  Needs `--confirm_run_unsafe_code` and HF_ALLOW_CODE_EVAL=1 (HF evaluate's
                  code_eval executes model code: only run inside Modal).
Generation lengths are left at each task's own default (ifeval 1280, humaneval 1024).

Benchmarks are run in as few lm-eval calls as possible (ruling R34): every vLLM engine start costs
about a minute. lm-eval 0.4.13's `--limit` is one value per call that applies to every leaf task,
and `--apply_chat_template` is per call too, so only benchmarks with the same limit and chat setting
can share a call: [hellaswag, truthfulqa, winogrande] (500 each), [mmlu] (10 per subject) and
[ifeval, humaneval] (all, chat). Each call's results JSON is sliced back into one per benchmark
(`slice_lm_eval`), so per-benchmark versions/scores are the same as with one call per benchmark.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BENCHMARKS: dict[str, dict] = {
    "hellaswag": {"task": "hellaswag", "metric": "acc_norm,none", "limit": 500, "chat": False},
    "mmlu": {"task": "mmlu", "metric": "acc,none", "limit": 10, "chat": False},
    "truthfulqa": {"task": "truthfulqa_mc2", "metric": "acc,none", "limit": 500, "chat": False},
    "winogrande": {"task": "winogrande", "metric": "acc,none", "limit": 500, "chat": False},
    "ifeval": {"task": "ifeval", "metric": "prompt_level_strict_acc,none", "limit": None, "chat": True},
    "humaneval": {"task": "humaneval_instruct", "metric": "pass@1,create_test", "limit": None, "chat": True},
}

SMOKE_LIMIT = 5

# lm-eval is started through this bootstrap instead of `python -m lm_eval`: IFEval's language checks call
# langdetect, whose DetectorFactory is unseeded by default, so identical generations could score differently
# between runs (one IFEval prompt flipped across two otherwise identical L4 evals).
LANGDETECT_SEED = 0
LM_EVAL_BOOTSTRAP = (
    "import sys\n"
    "try:\n"
    "    from langdetect import DetectorFactory\n"
    f"    DetectorFactory.seed = {LANGDETECT_SEED}\n"
    "except ImportError:\n"
    "    pass\n"
    "sys.argv[0] = 'lm_eval'\n"
    "from lm_eval.__main__ import cli_evaluate\n"
    "cli_evaluate()\n"
)


def lm_eval_calls() -> list[tuple[str, ...]]:
    """Benchmarks grouped into lm-eval calls by (limit, chat), in BENCHMARKS order.

    Smoke runs use the same grouping as full runs (it only lowers the limit).
    """
    groups: dict[tuple, list[str]] = {}
    for name, b in BENCHMARKS.items():
        groups.setdefault((b["limit"], b["chat"]), []).append(name)
    return [tuple(v) for v in groups.values()]


def lm_eval_args(model_path: str, names: str | tuple[str, ...] | list[str], smoke: bool = False) -> list[str]:
    """lm-eval CLI arguments (without the program and --output_path) for one call over `names`."""
    names = [names] if isinstance(names, str) else list(names)
    bs = [BENCHMARKS[n] for n in names]
    if len({(b["limit"], b["chat"]) for b in bs}) != 1:
        raise ValueError(f"{names}: benchmarks in one lm-eval call need the same limit and chat setting")
    b = bs[0]
    args = [
        "run",
        "--model", "vllm",
        "--model_args", f"pretrained={model_path},dtype=bfloat16",
        "--tasks", ",".join(x["task"] for x in bs),
        "--batch_size", "auto",
    ]
    if b["chat"]:
        args.append("--apply_chat_template")
    if "humaneval" in names:
        args.append("--confirm_run_unsafe_code")
    limit = SMOKE_LIMIT if smoke else b["limit"]
    if limit is not None:
        args += ["--limit", str(limit)]
    return args


def parse_lm_eval(results_json: dict, name: str) -> float:
    b = BENCHMARKS[name]
    return float(results_json["results"][b["task"]][b["metric"]])


def _task_tree(results_json: dict, task: str) -> set[str]:
    """`task` plus every (sub)group and leaf task under it, from lm-eval's `group_subtasks`."""
    tree, todo = set(), [task]
    subtasks = results_json.get("group_subtasks", {})
    while todo:
        t = todo.pop()
        if t not in tree:
            tree.add(t)
            todo.extend(subtasks.get(t, []))
    return tree


def slice_lm_eval(results_json: dict, name: str) -> dict:
    """The part of a (multi-task) lm-eval results JSON that belongs to one benchmark.

    Per-task sections (results, groups, versions, n-shot, n-samples, configs, higher_is_better,
    group_subtasks) keep only the benchmark's task tree; run-level fields (config, versions of the
    package, ...) are kept as they are.
    """
    tree = _task_tree(results_json, BENCHMARKS[name]["task"])
    per_task = ("results", "groups", "versions", "n-shot", "n-samples", "configs", "higher_is_better",
                "group_subtasks")
    out = {k: v for k, v in results_json.items() if k not in per_task and k != "task_hashes"}
    for k in per_task:
        if k in results_json:
            out[k] = {t: v for t, v in results_json[k].items() if t in tree}
    return out


def task_versions(results_json: dict) -> dict:
    """All task/group versions lm-eval recorded (for mmlu: the group and all 57 subtasks)."""
    return dict(results_json["versions"])


def run_lm_eval(model_path: str, names: str | tuple[str, ...], smoke: bool = False) -> dict:
    """Run one lm-eval call over one or more benchmarks and return its (combined) results JSON."""
    env = {**os.environ, "HF_ALLOW_CODE_EVAL": "1"}
    with tempfile.TemporaryDirectory(prefix="lm_eval-") as out:
        cmd = [sys.executable, "-c", LM_EVAL_BOOTSTRAP, *lm_eval_args(model_path, names, smoke),
               "--output_path", out]
        subprocess.run(cmd, check=True, env=env)
        files = sorted(Path(out).rglob("results_*.json"))
        if len(files) != 1:
            raise RuntimeError(f"expected one lm-eval results file in {out}, found {files}")
        return json.loads(files[0].read_text())


def run_benchmarks(model_path: str, smoke: bool = False, run=None) -> dict[str, dict]:
    """Per-benchmark results JSONs (sliced) from the grouped lm-eval calls, in BENCHMARKS order."""
    run = run or run_lm_eval
    raw = {}
    for names in lm_eval_calls():
        res = run(model_path, names, smoke)
        raw.update({n: slice_lm_eval(res, n) for n in names})
    return {n: raw[n] for n in BENCHMARKS}


def run_forgetting_suite(model_path: str, smoke: bool = False) -> dict[str, float]:
    return {name: parse_lm_eval(r, name) for name, r in run_benchmarks(model_path, smoke).items()}


def forgetting_mean(bench: dict, base: dict) -> float:
    """Mean per-benchmark difference (checkpoint - base); negative = forgetting."""
    if set(bench) != set(base):
        raise ValueError(f"benchmark sets differ: {sorted(bench)} vs {sorted(base)}")
    return sum(bench[k] - base[k] for k in bench) / len(bench)
