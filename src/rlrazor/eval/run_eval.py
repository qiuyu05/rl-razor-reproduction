"""Score one checkpoint (GSM8K accuracy, forward KL to the base, forgetting suite)."""
from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
import tempfile
import time
from pathlib import Path

import torch

from rlrazor.config import RunConfig, run_id
from rlrazor.eval.forgetting import (
    BENCHMARKS,
    LANGDETECT_SEED,
    forgetting_mean,
    lm_eval_calls,
    parse_lm_eval,
    run_lm_eval,
    slice_lm_eval,
    task_versions,
)
from rlrazor.eval.gsm8k import gsm8k_accuracy
from rlrazor.eval.kl import N_KL_PROMPTS, forward_kl, kl_prompts, response_token_counts, sequence_logprobs
from rlrazor.gen import generate
from rlrazor.results import append_row, has_row

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
KL_CACHE_FILE = "base_kl_samples.json"
KL_SEED = 0
KL_TEMPERATURE = 1.0
MAX_TOKENS = 512
SMOKE_GSM8K_LIMIT = 20
# vLLM settings for every evaluation generation/scoring (lm-eval subprocesses, GSM8K greedy, KL samples).
# Batch-invariant mode makes outputs independent of batch composition: without it, re-evaluating the same
# model on L4 moved IFEval/HumanEval by 1-2 pt; with it, two repeats were identical on all six benchmarks
# and GSM8K, at no measurable extra time (ruling R37).
EVAL_VLLM_ENV = {"VLLM_BATCH_INVARIANT": "1"}


@contextlib.contextmanager
def eval_vllm_env():
    """Set EVAL_VLLM_ENV in os.environ (inherited by lm-eval subprocesses and vLLM engines), then restore."""
    old = {k: os.environ.get(k) for k in EVAL_VLLM_ENV}
    os.environ.update(EVAL_VLLM_ENV)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# KL log-probs are scored in fp32 on every device (base and checkpoints alike): the 0.5B
# model is cheap, and bf16 rounding would add noise comparable to small KL values.
KL_DTYPE = torch.float32


def _scoring_device() -> str:
    """The device class the base log-probs are scored on (e.g. "NVIDIA A100-SXM4-40GB" or "cpu")."""
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def _kl_cache_spec(base_model: str) -> dict:
    """Everything that defines the cached base samples; a cache must match it exactly.

    It includes the scoring device class (ruling R26): base and checkpoint log-probs must come from the same
    kind of GPU, or kernel differences would add a small constant offset to every kl_fwd.
    """
    return {
        "base_model": base_model,
        "n_samples": N_KL_PROMPTS,
        "seed": KL_SEED,
        "temperature": KL_TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "dtype": str(KL_DTYPE).removeprefix("torch."),
        "device": _scoring_device(),
    }


def _logprobs(model_path: str, prompts, responses) -> torch.Tensor:
    """Teacher-forced response log-probs in fp32 (on CUDA when available)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cuda = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=KL_DTYPE).to(
        "cuda" if cuda else "cpu"
    )
    tok = AutoTokenizer.from_pretrained(model_path)
    try:
        return sequence_logprobs(model, tok, prompts, responses)
    finally:
        del model
        if cuda:
            torch.cuda.empty_cache()


def _atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def validate_base_kl_cache(cache_dir, base_model: str = BASE_MODEL) -> Path | None:
    """The cache path if it exists and matches this process's spec (incl. GPU class); None if absent.

    Raises ValueError on any mismatch. Cheap (no GPU work), so callers run it before paid evals.
    """
    path = Path(cache_dir) / KL_CACHE_FILE
    if not path.exists():
        return None
    cache = json.loads(path.read_text())
    for key, want in _kl_cache_spec(base_model).items():
        if cache.get(key) != want:
            raise ValueError(f"{path}: cached {key}={cache.get(key)!r}, expected {want!r}")
    return path


def ensure_base_kl_samples(cache_dir, base_model: str = BASE_MODEL, backend: str = "vllm") -> Path:
    """Cache the 256 base-model KL samples (T=1, seed 0) and the base's log-probs on them.

    Generated once; later calls reuse the file after checking that it was built with the
    same spec (base model, n_samples, seed, temperature, max_tokens, scoring dtype).
    Concurrent first callers may each compute it, but every writer renames a complete file
    into place, so readers never see a partial cache.
    """
    path = validate_base_kl_cache(cache_dir, base_model)
    if path is not None:
        return path
    path = Path(cache_dir) / KL_CACHE_FILE
    spec = _kl_cache_spec(base_model)
    prompts = list(kl_prompts(seed=KL_SEED)["prompt"])
    outs = generate(base_model, prompts, n=1, temperature=KL_TEMPERATURE,
                    max_tokens=MAX_TOKENS, seed=KL_SEED, backend=backend)
    responses = [o[0] for o in outs]
    base_logps = _logprobs(base_model, prompts, responses)
    _atomic_write_json(path, {
        **spec,
        "responses": responses,
        "base_logps": base_logps.tolist(),
    })
    return path


def checkpoint_kl_detail(ckpt_path: str, cache_path) -> dict:
    """KL(pi0 || pi_ckpt) on the cached base samples, plus per-sequence log-ratio sums
    log pi0(y|x) - log pi(y|x) and response token counts (R46), so per-token KL can be reported later."""
    from transformers import AutoTokenizer

    cache = json.loads(Path(cache_path).read_text())
    prompts = list(kl_prompts(seed=KL_SEED)["prompt"])
    model_logps = _logprobs(ckpt_path, prompts, cache["responses"])
    base = torch.tensor(cache["base_logps"])
    n_tok = response_token_counts(AutoTokenizer.from_pretrained(ckpt_path), cache["responses"])
    sums = (base - model_logps).tolist()
    return {"kl_fwd": forward_kl(base, model_logps), "kl_logratio_sums": sums, "kl_n_tokens": n_tok,
            "kl_fwd_per_token": sum(sums) / max(1, sum(n_tok))}


def checkpoint_kl(ckpt_path: str, cache_path) -> float:
    """KL(pi0 || pi_ckpt) on the cached base samples."""
    return checkpoint_kl_detail(ckpt_path, cache_path)["kl_fwd"]


def _score(ckpt_path, smoke, cache_dir, base_model, backend, raw_dir, base_row):
    """lm-eval benchmarks, GSM8K accuracy and forward KL of one checkpoint (inside eval_vllm_env)."""
    # lm-eval runs first, in its own process, before this process holds any GPU memory. Benchmarks
    # are grouped into as few calls as possible, then sliced back into one results JSON each.
    raw = {}
    for names in lm_eval_calls():
        res = run_lm_eval(ckpt_path, names, smoke)
        raw.update({name: slice_lm_eval(res, name) for name in names})
    if raw_dir is not None:
        for name, res in raw.items():
            _atomic_write_json(Path(raw_dir) / f"{name}.json", res)
    bench = {name: parse_lm_eval(raw[name], name) for name in BENCHMARKS}
    versions = {name: task_versions(raw[name]) for name in BENCHMARKS}
    if base_row is not None and base_row.get("lm_eval_versions") != versions:
        raise ValueError("base_row lm_eval_versions differ from this checkpoint's; rows not comparable")

    acc = gsm8k_accuracy(ckpt_path, limit=SMOKE_GSM8K_LIMIT if smoke else None, backend=backend)
    kl = checkpoint_kl_detail(ckpt_path, ensure_base_kl_samples(cache_dir, base_model, backend))
    return bench, versions, acc, kl


def evaluate_checkpoint(
    cfg: RunConfig,
    ckpt_path: str,
    step: int,
    results_path,
    base_row: dict | None,
    smoke: bool = False,
    *,
    total_steps: int,
    cache_dir=None,
    base_model: str = BASE_MODEL,
    backend: str = "vllm",
    raw_dir=None,
    extra: dict | None = None,
) -> dict | None:
    """Score a checkpoint and append its row to the results directory.

    If `raw_dir` is given, each benchmark's raw lm-eval results JSON is also saved there as <name>.json (audit trail).

    Returns None when (run_id, step) is already scored. `base_row=None` means this row is
    the base model itself (forgetting_mean = 0); otherwise base_row must have the same
    `smoke` flag and lm-eval versions, or a ValueError is raised and no row is written.
    `cache_dir` holds the base KL samples and defaults to `<results_path>/../kl_cache`,
    so it lives on the same (shared) volume.
    """
    rid = run_id(cfg)
    if has_row(results_path, rid, step):
        return None
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}")
    if base_row is not None and base_row.get("smoke") != smoke:
        raise ValueError(f"base_row smoke={base_row.get('smoke')!r} but this row smoke={smoke!r}")
    if cache_dir is None:
        cache_dir = Path(results_path).parent / "kl_cache"
    # Cheap guards before any paid GPU work: the KL cache must match this process (base model,
    # sampling spec, scoring GPU class) and a checkpoint row needs an existing cache and a base row
    # from the same lm-eval package.
    package = importlib.metadata.version("lm-eval")
    if validate_base_kl_cache(cache_dir, base_model) is None and base_row is not None:
        raise RuntimeError(f"no base KL cache in {cache_dir}: build it (base_eval) before scoring checkpoints")
    if base_row is not None and base_row.get("lm_eval_package") != package:
        raise ValueError("base_row lm_eval_package differs from this checkpoint's; rows not comparable")
    if base_row is not None and base_row.get("eval_vllm_env") != EVAL_VLLM_ENV:
        raise ValueError("base_row eval_vllm_env differs from this checkpoint's; rows not comparable")
    if base_row is not None and base_row.get("langdetect_seed") != LANGDETECT_SEED:
        # IFEval scoring depends on the langdetect seed set by the lm-eval bootstrap (R38)
        raise ValueError("base_row langdetect_seed differs from this checkpoint's; rows not comparable")
    t0 = time.perf_counter()
    with eval_vllm_env():
        scores = _score(ckpt_path, smoke, cache_dir, base_model, backend, raw_dir, base_row)
    bench, versions, acc, kl = scores

    row = {
        "run_id": rid,
        "arm": cfg.arm,
        "method": cfg.method,
        "lr": cfg.lr,
        "seed": cfg.seed,
        "noise_p": cfg.noise_p,
        "mix_f": cfg.mix_f,
        "lora": cfg.lora_r,
        "wise_alpha": cfg.wise_alpha,
        "init_from": cfg.init_from,
        "step": step,
        "frac_trained": step / total_steps,
        "gsm8k_acc": acc,
        "kl_fwd": kl["kl_fwd"],
        "kl_fwd_per_token": kl["kl_fwd_per_token"],
        "kl_logratio_sums": kl["kl_logratio_sums"],
        "kl_n_tokens": kl["kl_n_tokens"],
        "bench": bench,
        "forgetting_mean": 0.0 if base_row is None else forgetting_mean(bench, base_row["bench"]),
        "smoke": smoke,
        "lm_eval_versions": versions,
        "lm_eval_package": package,
        "eval_vllm_env": dict(EVAL_VLLM_ENV),
        "langdetect_seed": LANGDETECT_SEED,
    }
    row.update(extra or {})  # e.g. {"diverged", "diverged_step"} from the run's ckpts.json (R50)
    row["gpu_seconds"] = time.perf_counter() - t0
    append_row(results_path, row)
    return row
