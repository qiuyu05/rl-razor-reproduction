"""Checkpoint schedule and the callback that saves full, loadable model dirs at it."""
import copy
from pathlib import Path

import torch
from transformers import TrainerCallback


def ckpt_steps(total_steps: int, n_ckpts: int) -> list[int]:
    """n_ckpts evenly spaced optimizer steps (rounded half up), the last one being total_steps."""
    if n_ckpts < 1 or total_steps < n_ckpts:
        # fewer steps than checkpoints would give step 0 or duplicate steps
        raise ValueError(f"need 1 <= n_ckpts <= total_steps, got n_ckpts={n_ckpts}, total_steps={total_steps}")
    # integer round-half-up of total_steps * k / n_ckpts; spacing >= 1 so steps are distinct and > 0
    return [(2 * total_steps * k + n_ckpts) // (2 * n_ckpts) for k in range(1, n_ckpts + 1)]


def n_ckpts_for(cfg) -> int:
    """smoke=True saves a single checkpoint at the final step."""
    return 1 if cfg.smoke else cfg.n_ckpts


def save_full_model(model, tokenizer, path: str) -> None:
    """Save weights + config + tokenizer in bf16 (the base model's dtype); a PEFT model is merged first.

    Works on a copy so the training model (fp32 master weights, possibly unmerged LoRA) is untouched.
    """
    model = copy.deepcopy(model)
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    model.to(torch.bfloat16).save_pretrained(path)
    tokenizer.save_pretrained(path)


class SaveAtSteps(TrainerCallback):
    """Save a full model dir to out_dir/step-{n} at each listed global step; records (step, dir)."""

    def __init__(self, steps: list[int], out_dir: str, tokenizer):
        self.steps = set(steps)
        self.out_dir = Path(out_dir)
        self.tokenizer = tokenizer
        self.saved: list[tuple[int, str]] = []

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step in self.steps and state.is_world_process_zero:
            path = str(self.out_dir / f"step-{step}")
            save_full_model(model, self.tokenizer, path)
            self.saved.append((step, path))


def run_with_ckpts(trainer, total_steps: int, n_ckpts: int, out_dir: str,
                   early_stop=None) -> list[tuple[int, str]]:
    """Train, saving at ckpt_steps(total_steps, n_ckpts); returns the (step, ckpt_dir) pairs.

    Raises RuntimeError if the saved steps are not exactly the scheduled ones (e.g. training stopped early), unless
    `early_stop` (a callback with `diverged_at`) fired: then the stop step is saved too (if not already) and the
    checkpoints reached so far are returned.
    """
    expected = ckpt_steps(total_steps, n_ckpts)
    cb = SaveAtSteps(expected, out_dir, trainer.processing_class)
    trainer.add_callback(cb)
    trainer.train()
    if early_stop is not None and early_stop.diverged_at is not None:
        step = trainer.state.global_step
        if trainer.is_world_process_zero() and step not in [s for s, _ in cb.saved]:
            path = str(Path(out_dir) / f"step-{step}")
            save_full_model(trainer.model, trainer.processing_class, path)
            cb.saved.append((step, path))
        return cb.saved
    saved = [step for step, _ in cb.saved]
    if trainer.is_world_process_zero() and saved != expected:
        raise RuntimeError(f"saved checkpoints at steps {saved}, expected {expected}")
    return cb.saved
