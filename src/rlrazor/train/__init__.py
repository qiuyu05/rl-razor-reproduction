"""Trainer wrappers (SFT/LoRA, GRPO, GKD) that save full-model checkpoints."""
import torch

from rlrazor.data.mixing import TOKENIZER_NAME

# The base model and its tokenizer are the same hub id; reuse the one constant.
BASE = TOKENIZER_NAME


def device_args() -> dict:
    """Trainer args that depend on the hardware (we assume 1 GPU in real runs, CPU in tests)."""
    # bf16 mixed precision on GPU. Master weights stay fp32 (TRL's default load dtype), so small-lr updates
    # (GRPO ~1e-6) are not lost to bf16 rounding. CPU (tests) falls back to plain fp32 and no pinned memory.
    cuda = torch.cuda.is_available()
    return {"bf16": cuda, "dataloader_pin_memory": cuda}


# smoke=True overrides every trainer to this: 2 steps, batch 2, 32 new tokens, a single checkpoint.
SMOKE_STEPS = 2
SMOKE_BATCH = 2
SMOKE_MAX_NEW_TOKENS = 32
