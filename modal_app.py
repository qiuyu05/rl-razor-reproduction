"""Modal app: GPU functions for training, checkpoint evaluation, WiSE-FT and the base row.

Volume `rlrazor` is mounted at /vol (layout in rlrazor/pipeline.py). The GPU type comes from the env var
RLRAZOR_GPU at deploy time (default A100-40GB) for training and RLRAZOR_EVAL_GPU (default L4) for evaluation. Usage:
    uv run python scripts/run.py configs/smoke/sft.yaml --smoke      # train + eval one config
    uv run modal run modal_app.py::base_eval                          # base row (step 0, run_id "base")
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

GPU = os.environ.get("RLRAZOR_GPU", "A100-40GB")  # training
# Evaluation (eval_ckpt and base_eval). The base row and its KL cache are tied to this GPU class (R26),
# so every checkpoint of the study must be scored on it (ruling R34: L4).
EVAL_GPU = os.environ.get("RLRAZOR_EVAL_GPU", "L4")
# The one-off teacher data job (Qwen2.5-7B-Instruct in vLLM, bf16: ~15 GB of weights + KV cache): the eval GPU class
# (L4, 24 GB), enough for it. scripts/launch.py prices it at the eval GPU's rate when the two are equal.
TEACHER_GPU = os.environ.get("RLRAZOR_TEACHER_GPU", EVAL_GPU)
# Eval containers per run.py app: each run's checkpoints are scored one at a time after training, so a
# run holds at most ONE GPU at any moment and launch.py --max-parallel N bounds the workspace at N GPUs (the
# Starter plan caps a workspace at 10 concurrent GPUs; queued containers are not billed but wait).
EVAL_MAX_CONTAINERS = int(os.environ.get("RLRAZOR_EVAL_MAX_CONTAINERS", "1"))
VOL_PATH = "/vol"
HOURS = 3600

app = modal.App("rlrazor")
vol = modal.Volume.from_name("rlrazor", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    # requirements-modal.txt: the project pins (uv.lock) plus vllm 0.33.0, compiled for linux / py3.11.
    .uv_pip_install(requirements=[str(Path(__file__).parent / "requirements-modal.txt")])
    .env({
        "HF_HOME": f"{VOL_PATH}/hf",  # model + dataset downloads persist on the Volume
        "HF_ALLOW_CODE_EVAL": "1",  # HumanEval executes model code: only inside the Modal sandbox
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        # vLLM's FlashInfer top-k/top-p sampler JIT-compiles with nvcc, which debian-slim lacks (no flashinfer-cubin
        # wheel for 0.7.2); vLLM's PyTorch sampler draws from the same distribution.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    })
    .add_local_python_source("rlrazor")
)

common = dict(image=image, volumes={VOL_PATH: vol})


def _cfg(cfg_yaml: str):
    from rlrazor.config import config_from_yaml

    return config_from_yaml(cfg_yaml)


@app.function(gpu=GPU, timeout=6 * HOURS, **common)
def train(cfg_yaml: str) -> list[tuple[int, str]]:
    """Train one config (dispatch on cfg.method); returns [(step, ckpt_dir)]. Idempotent per run_id."""
    from rlrazor import pipeline

    vol.reload()  # see checkpoints other runs committed (init_from sources)
    try:
        return pipeline.train_run(_cfg(cfg_yaml), VOL_PATH)
    finally:
        vol.commit()


@app.function(gpu=TEACHER_GPU, timeout=4 * HOURS, retries=0, **common)
def teacher_data(cfg_yaml: str) -> dict:
    """Build the teacher SFT set (method "teacher_data") into /vol/cache/teacher_data/; done.json is written last."""
    from rlrazor import pipeline

    vol.reload()
    try:
        return pipeline.build_teacher_data(_cfg(cfg_yaml), VOL_PATH)
    finally:
        vol.commit()


@app.function(**common, timeout=10 * 60)
def run_status(cfg_yaml: str) -> dict:
    """What exists for a config (ckpt index, scored steps, base row, general pool); CPU only."""
    from rlrazor import pipeline

    vol.reload()
    return pipeline.run_status(VOL_PATH, _cfg(cfg_yaml))


@app.function(**common, timeout=10 * 60)
def archive(paths: list[str], dest: str) -> list[str]:
    """Move Volume paths (relative to /vol) under /vol/<dest>/, keeping their relative layout. Never deletes."""
    vol.reload()
    moved = []
    for rel in paths:
        src, dst = Path(VOL_PATH) / rel, Path(VOL_PATH) / dest / rel
        if src.exists():
            if dst.exists():
                raise FileExistsError(dst)
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
            moved.append(rel)
    vol.commit()
    return moved


@app.function(**common, timeout=10 * 60)
def ckpt_index(cfg_yaml: str) -> dict:
    """{"total_steps", "ckpts": [[step, dir], ...]} of a finished training run."""
    from rlrazor import pipeline

    vol.reload()
    return pipeline.ckpt_index(VOL_PATH, _cfg(cfg_yaml))


@app.function(gpu=EVAL_GPU, timeout=2 * HOURS, max_containers=EVAL_MAX_CONTAINERS, **common)
def eval_ckpt(cfg_yaml: str, ckpt_path: str, step: int, total_steps: int | None = None) -> dict | None:
    """Score one checkpoint and write its row; None if (run_id, step) was already scored."""
    from rlrazor import pipeline

    vol.reload()
    cfg = _cfg(cfg_yaml)
    if total_steps is None:
        total_steps = pipeline.ckpt_index(VOL_PATH, cfg)["total_steps"]
    try:
        return pipeline.eval_ckpt(cfg, ckpt_path, step, total_steps, VOL_PATH)
    finally:
        vol.commit()


@app.function(gpu=EVAL_GPU, timeout=2 * HOURS, **common)
def base_eval(smoke: bool = False) -> dict:
    """Build the base KL cache, then score the base model as step 0 (run_id "base"); cached afterwards."""
    import json

    from rlrazor import pipeline

    vol.reload()
    try:
        row = pipeline.base_eval(VOL_PATH, smoke=smoke)
    finally:
        vol.commit()
    print(json.dumps(row, indent=1, sort_keys=True))
    return row


@app.function(cpu=4, memory=16384, timeout=30 * 60, **common)
def wise(base: str, ft: str, alpha: float) -> str:
    """WiSE-FT checkpoint (1-alpha)*base + alpha*ft; refs are "base" or ckpts-relative paths."""
    from rlrazor import pipeline

    vol.reload()
    try:
        return pipeline.wise(VOL_PATH, base, ft, alpha)
    finally:
        vol.commit()


@app.function(cpu=4, memory=16384, timeout=30 * 60, **common)
def wise_run(cfg_yaml: str) -> list[tuple[int, str]]:
    """WiSE-FT eval-only config: interpolate on CPU and index the result as the run's checkpoint (no GPU)."""
    from rlrazor import pipeline

    vol.reload()
    try:
        return pipeline.wise_run(_cfg(cfg_yaml), VOL_PATH)
    finally:
        vol.commit()


@app.function(cpu=4, memory=16384, timeout=1 * HOURS, **common)
def build_general_pool() -> int:
    """Stream + cache the Tulu-3 pool to /vol/cache/general_pool once (ruling R9)."""
    from rlrazor import pipeline

    vol.reload()
    try:
        return len(pipeline.general_pool(VOL_PATH, build=True))
    finally:
        vol.commit()


@app.function(**common, timeout=10 * 60)
def export_results() -> int:
    """/vol/results/ -> /vol/results.jsonl (+ dataset_stats.jsonl, train_times.jsonl); returns the row count."""
    from rlrazor import pipeline

    vol.reload()
    try:
        return pipeline.export(VOL_PATH)
    finally:
        vol.commit()


@app.function(**common, timeout=10 * 60)
def read_files(paths: list[str]) -> dict[str, str]:
    """Text of small Volume files (paths relative to /vol), e.g. result rows and raw lm-eval outputs."""
    vol.reload()
    return {p: (Path(VOL_PATH) / p).read_text() for p in paths if (Path(VOL_PATH) / p).is_file()}


@app.function(**common, timeout=10 * 60)
def list_files(rel_dir: str) -> list[str]:
    """Files under a Volume directory (relative to /vol), recursively."""
    vol.reload()
    root = Path(VOL_PATH) / rel_dir
    return sorted(str(p.relative_to(VOL_PATH)) for p in root.rglob("*") if p.is_file())
