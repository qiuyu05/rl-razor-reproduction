"""On-policy distillation (reverse KL on student rollouts) with TRL's GKDTrainer."""
import warnings

from transformers import AutoTokenizer

from rlrazor.config import RunConfig
from rlrazor.train import SMOKE_BATCH, SMOKE_MAX_NEW_TOKENS, SMOKE_STEPS, device_args
from rlrazor.train.schedule import n_ckpts_for, run_with_ckpts

with warnings.catch_warnings():
    # GKD lives in trl.experimental in trl 1.18.2, whose import emits a TRLExperimentalWarning; we accept that.
    warnings.simplefilter("ignore")
    from trl.experimental.gkd import GKDConfig, GKDTrainer

DISTILL_STEPS = 100
# 4 x 8 accumulation = 32 prompts per optimizer step on 1 GPU. Per-device stays small because GKD materializes
# full-vocab (152k) logits and fp32 log-softmax for both student and teacher over ~700-token sequences.
PER_DEVICE_BATCH = 4
ACCUM = 8


def _to_messages(prompt: list[dict]) -> dict:
    # GKDTrainer's ChatML collator reads a "messages" conversation and treats the last turn as the completion.
    # With lmbda=1.0 every batch's completion is replaced by a fresh student rollout, so it is an empty placeholder.
    return {"messages": prompt + [{"role": "assistant", "content": ""}]}


def distill_args(cfg: RunConfig, out_dir: str) -> GKDConfig:
    if cfg.smoke:
        steps, per_device, accum, max_new = SMOKE_STEPS, SMOKE_BATCH, 1, SMOKE_MAX_NEW_TOKENS
    else:
        steps, per_device, accum, max_new = DISTILL_STEPS, PER_DEVICE_BATCH, ACCUM, 512
    return GKDConfig(
        output_dir=out_dir,
        lmbda=1.0,  # fully on-policy: train on student samples
        beta=1.0,  # generalized JSD at beta=1 is the reverse KL(student || teacher)
        temperature=1.0,
        max_new_tokens=max_new,
        max_steps=steps,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=accum,
        learning_rate=cfg.lr,
        # Constant LR: intermediate checkpoints are frontier points, and decay would tie them to the horizon.
        lr_scheduler_type="constant",
        warmup_steps=0,
        **device_args(),
        seed=cfg.seed,
        save_strategy="no",  # checkpoints are written by SaveAtSteps instead
        report_to="none",
    )


def train_distill(
    cfg: RunConfig, student_path: str, teacher_path: str, prompts_ds, out_dir: str
) -> list[tuple[int, str]]:
    args = distill_args(cfg, out_dir)
    ds = prompts_ds.map(_to_messages, input_columns="prompt", remove_columns=prompts_ds.column_names)
    trainer = GKDTrainer(
        model=student_path,
        teacher_model=teacher_path,
        args=args,
        train_dataset=ds,
        processing_class=AutoTokenizer.from_pretrained(student_path),
    )
    return run_with_ckpts(trainer, args.max_steps, n_ckpts_for(cfg), out_dir)
