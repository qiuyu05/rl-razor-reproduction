"""Supervised fine-tuning (full or LoRA) with TRL's SFTTrainer."""
import math

from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from rlrazor.config import RunConfig
from rlrazor.train import BASE, SMOKE_BATCH, SMOKE_STEPS, device_args
from rlrazor.train.schedule import n_ckpts_for, run_with_ckpts

EFFECTIVE_BATCH = 32
PER_DEVICE_BATCH = 8  # x 4 accumulation = 32 sequences per optimizer step on 1 GPU
WARMUP_STEPS = 50


def sft_args(cfg: RunConfig, n_rows: int, out_dir: str) -> SFTConfig:
    if cfg.smoke:
        total, per_device, accum = SMOKE_STEPS, SMOKE_BATCH, 1
    else:
        # Training is bounded by max_steps (= cfg.epochs over the data at batch 32) rather than num_train_epochs,
        # so the last optimizer step is exactly `total` and the final checkpoint always lands.
        total = math.ceil(n_rows * cfg.epochs / EFFECTIVE_BATCH)
        per_device, accum = PER_DEVICE_BATCH, EFFECTIVE_BATCH // PER_DEVICE_BATCH
    return SFTConfig(
        output_dir=out_dir,
        max_steps=total,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=accum,
        learning_rate=cfg.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=WARMUP_STEPS,
        optim="adamw_torch",
        weight_decay=0.0,
        max_grad_norm=1.0,
        completion_only_loss=True,
        **device_args(),
        seed=cfg.seed,
        save_strategy="no",  # checkpoints are written by SaveAtSteps instead
        report_to="none",
    )


def train_sft(cfg: RunConfig, train_ds, out_dir: str, model_path: str = BASE) -> list[tuple[int, str]]:
    args = sft_args(cfg, len(train_ds), out_dir)
    peft_config = None
    if cfg.lora_r > 0:
        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=2 * cfg.lora_r,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
    trainer = SFTTrainer(
        model=model_path,
        args=args,
        train_dataset=train_ds.select_columns(["prompt", "completion"]),
        processing_class=AutoTokenizer.from_pretrained(model_path),
        peft_config=peft_config,
    )
    return run_with_ckpts(trainer, args.max_steps, n_ckpts_for(cfg), out_dir)
