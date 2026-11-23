"""Full-run trainer configs, built without loading a model."""
import pytest

from rlrazor.config import RunConfig
from rlrazor.train.distill import distill_args
from rlrazor.train.grpo import grpo_args
from rlrazor.train.schedule import ckpt_steps, run_with_ckpts
from rlrazor.train.sft import sft_args


def test_grpo_full_config(tmp_path):
    args = grpo_args(RunConfig(arm="grpo", method="grpo", lr=1e-6), str(tmp_path), use_vllm=False)
    assert args.generation_batch_size == 256  # 32 prompts x group of 8
    assert args.generation_batch_size // args.num_generations == 32
    assert args.steps_per_generation == 16
    assert args.per_device_train_batch_size * args.gradient_accumulation_steps == 256
    assert args.scale_rewards == "none"
    assert args.loss_type == "dr_grpo" and args.beta == 0.0
    assert args.max_steps == 400
    # paper Table 2 (RL): constant LR with 50 warmup steps (R46)
    assert args.lr_scheduler_type == "constant_with_warmup" and args.warmup_steps == 50
    assert grpo_args(RunConfig(arm="q0", method="grpo", lr=1e-5, grpo_steps=234), str(tmp_path),
                     use_vllm=False).max_steps == 234


def test_distill_full_config(tmp_path):
    args = distill_args(RunConfig(arm="distill", method="distill", lr=1e-5), str(tmp_path))
    assert (args.per_device_train_batch_size, args.gradient_accumulation_steps) == (4, 8)
    assert (args.lmbda, args.beta, args.max_new_tokens, args.max_steps) == (1.0, 1.0, 512, 100)
    assert args.lr_scheduler_type == "constant" and args.warmup_steps == 0


def test_sft_full_config(tmp_path):
    args = sft_args(RunConfig(arm="sft", method="sft", lr=1e-5), 7473, str(tmp_path))
    assert args.max_steps == 468  # ceil(7473 * 2 / 32)
    assert args.per_device_train_batch_size * args.gradient_accumulation_steps == 32
    assert args.lr_scheduler_type == "constant_with_warmup" and args.warmup_steps == 50


def test_ckpt_steps_rounds_half_up():
    assert ckpt_steps(5, 2) == [3, 5]  # 2.5 -> 3, not banker's 2


@pytest.mark.parametrize("total,n", [(3, 5), (0, 1), (4, 0)])
def test_ckpt_steps_rejects_degenerate_schedule(total, n):
    with pytest.raises(ValueError):
        ckpt_steps(total, n)


class _StoppedEarlyTrainer:
    """Stands in for a trainer whose run ends before any scheduled step."""

    processing_class = None

    def add_callback(self, cb):
        pass

    def train(self):
        pass

    def is_world_process_zero(self):
        return True


def test_run_with_ckpts_raises_when_saves_mismatch(tmp_path):
    with pytest.raises(RuntimeError, match="expected"):
        run_with_ckpts(_StoppedEarlyTrainer(), 4, 2, str(tmp_path))


def test_grpo_vllm_engine_is_seeded_by_run_seed(monkeypatch, tmp_path):
    """R33: TRL builds the colocated engine with LLM(seed=rank); train_grpo must add the run seed."""
    import trl.generation.vllm_generation as vg
    from datasets import Dataset

    import rlrazor.train.grpo as grpo

    built = []
    monkeypatch.setattr(vg, "LLM", lambda *a, **k: built.append(k["seed"]), raising=False)

    class FakeTrainer:
        def __init__(self, **kw):
            vg.LLM(model="m", seed=0)  # what VLLMGeneration._init_vllm does on rank 0

        def add_callback(self, cb):
            pass

    monkeypatch.setattr(grpo, "GRPOTrainer", FakeTrainer)
    monkeypatch.setattr(grpo, "AutoTokenizer", type("T", (), {"from_pretrained": staticmethod(lambda *a, **k: None)}))
    monkeypatch.setattr(grpo, "run_with_ckpts", lambda *a, **k: [])
    ds = Dataset.from_list([{"prompt": [{"role": "user", "content": "q"}], "answer": "1"}])
    for seed in (0, 7):
        grpo.train_grpo(RunConfig(arm="q0", method="grpo", lr=1e-6, seed=seed, smoke=True), ds, str(tmp_path))
    assert built == [0, 7]
    assert vg.LLM.__name__ == "<lambda>"  # the wrapper is removed after construction


# ---- R50: GRPO dead-run early stop

class _State:
    def __init__(self, step):
        self.global_step, self.is_world_process_zero = step, True


class _Control:
    should_training_stop = False


def _feed(cb, logs):
    ctl = _Control()
    for step, (reward, clipped) in enumerate(logs, start=1):
        cb.on_log(None, _State(step), ctl, logs={"reward": reward, "completions/clipped_ratio": clipped})
    return ctl


def test_dead_run_stop_needs_20_consecutive_dead_steps():
    from rlrazor.train.grpo import DEAD_PATIENCE, DeadRunStop

    assert DEAD_PATIENCE == 20
    cb = DeadRunStop()
    ctl = _feed(cb, [(0.4, 0.1)] * 5 + [(0.0, 0.95)] * 19)
    assert not ctl.should_training_stop and cb.diverged_at is None
    ctl = _feed(cb, [(0.0, 0.95)])  # 20th consecutive dead step (step 1 of this feed = global step 1 here)
    assert ctl.should_training_stop and cb.diverged_at == 1


def test_dead_run_stop_resets_on_any_live_step():
    from rlrazor.train.grpo import DeadRunStop

    cb = DeadRunStop()
    logs = [(0.0, 0.95)] * 19 + [(0.01, 0.95)] + [(0.0, 0.95)] * 19 + [(0.0, 0.5)] + [(0.0, 0.95)] * 19
    assert not _feed(cb, logs).should_training_stop
    cb2 = DeadRunStop()
    assert _feed(cb2, [(0.0, 0.95)] * 20).should_training_stop and cb2.diverged_at == 20
    cb3 = DeadRunStop()
    cb3.on_log(None, _State(1), _Control(), logs={"loss": 0.1})  # logs without the metrics are ignored
    assert cb3.streak == 0


def test_grpo_logs_every_step(tmp_path):
    args = grpo_args(RunConfig(arm="q0", method="grpo", lr=1e-6, grpo_steps=234), str(tmp_path), use_vllm=False)
    assert args.logging_steps == 1  # the dead-run rule counts optimizer steps


def test_run_with_ckpts_early_stop_saves_stop_step(tmp_path):
    """A stopped run keeps the checkpoints it reached plus one at the stop step; no schedule error."""
    from rlrazor.train.grpo import DeadRunStop

    saved_paths = []

    class Trainer:
        processing_class = None
        callbacks = []

        def add_callback(self, cb):
            self.callbacks.append(cb)

        def is_world_process_zero(self):
            return True

        def train(self):
            for cb in self.callbacks:
                if hasattr(cb, "saved"):
                    cb.saved.append((5, str(tmp_path / "step-5")))
                if isinstance(cb, DeadRunStop):
                    cb.diverged_at = 7
            self.state = _State(7)

    import rlrazor.train.schedule as sch

    orig = sch.save_full_model
    sch.save_full_model = lambda model, tok, path: saved_paths.append(path)
    try:
        t = Trainer()
        t.model = None
        stop = DeadRunStop()
        t.add_callback(stop)
        out = run_with_ckpts(t, 15, 3, str(tmp_path), early_stop=stop)
    finally:
        sch.save_full_model = orig
    assert out == [(5, str(tmp_path / "step-5")), (7, str(tmp_path / "step-7"))]
    assert saved_paths == [str(tmp_path / "step-7")]
