"""Grid configs (spec §4 as amended by R30/R40/R42/R43): stage sizes, LRs, methods, arms, init_from rules."""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import make_configs as mc  # noqa: E402
from rlrazor.config import load_config, run_id  # noqa: E402

GRPO_FINAL = "grpo-lr3e-05-s0-abcdef/step-400"
SFT_FINAL = "teacher_sft-lr3e-05-s0-abcdef/step-454"
RFT_FINAL = "rft_offline-lr3e-05-s0-abcdef/step-420"
BEST = dict(best_sft_lr=3e-5, best_grpo_lr=1e-5, best_rft_lr=7e-5, grpo_final=GRPO_FINAL, sft_final=SFT_FINAL,
            rft_final=RFT_FINAL, crossover=(0.05, 0.1))
INIT = re.compile(r"^[\w.+-]+/step-\d+$")


def _all():
    return {s: mc.stage_configs(s, **BEST) for s in mc.STAGES}


def _training(cfgs):
    return [c for _, c in cfgs if mc.is_training(c)]


def test_stage_counts():
    stages = _all()
    counts = {s: len(_training(c)) for s, c in stages.items()}
    assert counts == {"q0_sweep": 9, "q0_seeds": 6, "q1": 7, "q2": 12, "q3": 5, "q3_seeds": 4, "q4": 3}
    assert sum(counts.values()) == 46
    q4 = stages["q4"]
    assert sum(c.wise_alpha > 0 for _, c in q4) == 6  # WiSE-FT: eval-only, not training configs
    assert sum(c.lora_r > 0 for _, c in q4) == 2 and sum(c.method == "distill" for _, c in q4) == 1
    q1 = [c.method for c in _training(stages["q1"])]
    assert sorted(q1) == ["rft_iter"] * 3 + ["rl_distill_sft"] + ["self_rewrite"] * 3


def test_q0_sweep_methods_and_lrs():
    cfgs = mc.stage_configs("q0_sweep")
    names = [n for n, _ in cfgs]
    assert names == sorted(names) and len(set(names)) == len(names)  # launch order = file order
    first = cfgs[0][1]
    assert first.method == "teacher_data" and not mc.is_training(first)  # the one-off data job comes first
    by = {}
    for _, c in cfgs[1:]:
        by.setdefault(c.method, []).append(c.lr)
    assert by == {"teacher_sft": [1e-5, 3e-5, 7e-5], "grpo": [1e-5, 3e-5, 5e-5], "rft_offline": [1e-5, 3e-5, 7e-5]}
    assert "sft" not in by  # SFT(gold) arm dropped (R42)
    assert all(c.seed == 0 for _, c in cfgs)
    assert all(c.grpo_steps == 234 for _, c in cfgs if c.method == "grpo")  # R46: one epoch of prompts


def test_every_config_valid_and_fixed_fields():
    seen = set()
    for stage, cfgs in _all().items():
        for _, c in cfgs:
            # R40 as amended by R46: SFT-like runs evaluate end of epoch 1 and 2; GRPO / GKD 3 checkpoints
            if mc.is_training(c) and c.method in ("grpo", "distill"):
                assert c.n_ckpts == 3
            elif mc.is_training(c) and c.method != "rft_iter":
                assert c.n_ckpts == 2 and c.epochs == 2, c
            if c.method == "rft_iter":
                assert c.rounds == 4
            assert c.arm in ("q0", "q1", "q2", "q3", "q4")  # R30
            assert c.method != "sft" and c.method != "lora_sft"  # every "SFT" arm is SFT on teacher data
            assert not c.smoke
            rid = run_id(c)
            assert rid not in seen, rid
            seen.add(rid)


def test_arms():
    s = _all()
    arm = {(st, c.method): c.arm for st, cfgs in s.items() for _, c in cfgs}
    assert arm[("q0_sweep", "teacher_sft")] == arm[("q0_sweep", "grpo")] == arm[("q0_seeds", "grpo")] == "q0"
    assert arm[("q0_sweep", "rft_offline")] == arm[("q0_seeds", "rft_offline")] == "q1"  # spec: offline RFT is Q1
    assert {c.arm for _, c in s["q1"]} == {"q1"}
    assert {c.arm for _, c in s["q2"]} == {"q2"}
    assert {c.arm for _, c in s["q3"] + s["q3_seeds"]} == {"q3"}
    assert {c.arm for _, c in s["q4"]} == {"q4"}


def test_best_lrs_and_seeds_applied():
    s = _all()
    seeds = {}
    for _, c in s["q0_seeds"]:
        seeds.setdefault(c.method, []).append((c.lr, c.seed))
    assert seeds == {"teacher_sft": [(3e-5, 1), (3e-5, 2)], "grpo": [(1e-5, 1), (1e-5, 2)],
                     "rft_offline": [(7e-5, 1), (7e-5, 2)]}
    for _, c in s["q1"]:
        assert c.lr == 3e-5  # iterative RFT, self-rewrite and the control use teacher_sft's best LR
    q2 = {(c.method, c.noise_p, c.lr) for _, c in s["q2"]}
    assert q2 == {("teacher_sft", p, 3e-5) for p in (0.2, 0.4)} | {("grpo", p, 1e-5) for p in (0.2, 0.4)}
    assert sorted(c.mix_f for _, c in s["q3"]) == [0.02, 0.05, 0.1, 0.25, 0.5]
    assert all(c.method == "teacher_sft" and c.lr == 3e-5 and c.seed == 0 for _, c in s["q3"])
    assert sorted((c.mix_f, c.seed) for _, c in s["q3_seeds"]) == [(0.05, 1), (0.05, 2), (0.1, 1), (0.1, 2)]
    lora = sorted((c.method, c.lr, c.lora_r) for _, c in s["q4"] if c.lora_r)
    assert lora == [("teacher_sft", 1e-4, 64), ("teacher_sft", 3e-4, 64)]


def test_init_from_rules():
    for stage, cfgs in _all().items():
        for _, c in cfgs:
            if c.method == "rl_distill_sft":
                assert c.init_from == GRPO_FINAL
            elif c.method == "distill":
                assert c.init_from == SFT_FINAL
            elif c.wise_alpha > 0:
                assert c.init_from in (SFT_FINAL, RFT_FINAL)
            else:
                assert c.init_from is None, (stage, c)
            if c.init_from is not None:
                assert INIT.match(c.init_from)
    wise = sorted((c.init_from, c.wise_alpha, c.method) for _, c in mc.stage_configs("q4", **BEST) if c.wise_alpha)
    assert wise == sorted([(SFT_FINAL, a, "teacher_sft") for a in (0.25, 0.5, 0.75)]
                          + [(RFT_FINAL, a, "rft_offline") for a in (0.25, 0.5, 0.75)])


@pytest.mark.parametrize("stage,missing", [("q0_seeds", "best_grpo_lr"), ("q1", "grpo_final"),
                                           ("q2", "best_sft_lr"), ("q3_seeds", "crossover"), ("q4", "rft_final")])
def test_later_stages_require_their_inputs(stage, missing):
    kw = {k: v for k, v in BEST.items() if k != missing}
    with pytest.raises(ValueError, match=missing):
        mc.stage_configs(stage, **kw)


def test_bad_init_path_rejected():
    with pytest.raises(ValueError, match="step"):
        mc.stage_configs("q1", **{**BEST, "grpo_final": "/vol/ckpts/grpo"})


def test_write_stage_round_trips_and_removes_stale(tmp_path):
    stale = tmp_path / "q0_sweep" / "99_old.yaml"
    stale.parent.mkdir(parents=True)
    stale.write_text("x: 1\n")
    paths = mc.write_stage("q0_sweep", tmp_path)
    assert not stale.exists()
    assert [p.name for p in paths] == sorted(p.name for p in (tmp_path / "q0_sweep").glob("*.yaml"))
    want = mc.stage_configs("q0_sweep")
    assert [load_config(p) for p in paths] == [c for _, c in want]


def test_main_cli(tmp_path):
    assert mc.main(["--stage", "q3_seeds", "--best-sft-lr", "3e-5", "--crossover", "0.05,0.1",
                    "--out", str(tmp_path)]) == 0
    got = sorted((load_config(p).mix_f, load_config(p).seed) for p in (tmp_path / "q3_seeds").glob("*.yaml"))
    assert got == [(0.05, 1), (0.05, 2), (0.1, 1), (0.1, 2)]


def test_q0_lowlr_stage_r47_r50():
    """Forced deviation: GRPO {1e-6, 3e-6} and, symmetrically, teacher_sft / offline RFT at 3e-6 (R47/R50)."""
    cfgs = [c for _, c in mc.stage_configs("q0_lowlr")]
    got = [(c.method, c.lr, c.arm, c.n_ckpts) for c in cfgs]
    assert got == [("grpo", 1e-6, "q0", 3), ("grpo", 3e-6, "q0", 3), ("teacher_sft", 3e-6, "q0", 2),
                   ("rft_offline", 3e-6, "q1", 2)]
    assert all(c.grpo_steps == 234 for c in cfgs if c.method == "grpo") and all(c.seed == 0 for c in cfgs)
    assert "q0_lowlr" in mc.ALL_STAGES and "q0_lowlr" not in mc.STAGES  # not one of the spec's 46
    sweep = {run_id(c) for _, c in mc.stage_configs("q0_sweep")}
    assert not sweep & {run_id(c) for c in cfgs}


def test_writing_lowlr_leaves_q0_sweep_untouched(tmp_path):
    mc.write_stage("q0_sweep", tmp_path)
    before = sorted(p.read_text() for p in (tmp_path / "q0_sweep").glob("*.yaml"))
    assert mc.main(["--stage", "q0_lowlr", "--out", str(tmp_path)]) == 0
    assert sorted(p.read_text() for p in (tmp_path / "q0_sweep").glob("*.yaml")) == before
    assert len(list((tmp_path / "q0_lowlr").glob("*.yaml"))) == 4


def test_seeds_and_q1_take_low_lrs():
    s = mc.stage_configs("q0_seeds", best_sft_lr=1e-5, best_grpo_lr=3e-6, best_rft_lr=1e-5)
    assert {(c.method, c.lr) for _, c in s} == {("teacher_sft", 1e-5), ("grpo", 3e-6), ("rft_offline", 1e-5)}
    q1 = mc.stage_configs("q1", best_sft_lr=1e-5, grpo_final="grpo-lr3e-06-s0-abcdef/step-234")
    assert {c.lr for _, c in q1} == {1e-5}
    assert [c.init_from for _, c in q1 if c.method == "rl_distill_sft"] == ["grpo-lr3e-06-s0-abcdef/step-234"]


def test_stage2_r54_arms_order_and_values():
    """Stage 2 (R54): teacher_sft 3e-6 and GRPO 1e-6 arms; q4 launches WiSE-FT before LoRA / GKD."""
    kw = dict(best_sft_lr=3e-6, best_grpo_lr=1e-6, best_rft_lr=3e-6,
              grpo_final="grpo-lr1e-06-s0-6f9f01/step-234", sft_final="teacher_sft-lr3e-06-s0-72863e/step-459",
              rft_final="rft_offline-lr3e-06-s0-863d2a/step-384", crossover=(0.05, 0.1))
    q2 = [c for _, c in mc.stage_configs("q2", **kw)]
    assert sorted((c.method, c.lr, c.noise_p, c.seed) for c in q2) == sorted(
        [("teacher_sft", 3e-6, p, s) for p in (0.2, 0.4) for s in (0, 1, 2)]
        + [("grpo", 1e-6, p, s) for p in (0.2, 0.4) for s in (0, 1, 2)])
    q3 = [c for _, c in mc.stage_configs("q3", **kw)]
    assert [(c.method, c.lr, c.mix_f, c.seed) for c in q3] == [("teacher_sft", 3e-6, f, 0)
                                                                for f in (0.02, 0.05, 0.10, 0.25, 0.50)]
    q3s = [c for _, c in mc.stage_configs("q3_seeds", **kw)]
    assert [(c.mix_f, c.seed) for c in q3s] == [(0.05, 1), (0.05, 2), (0.1, 1), (0.1, 2)]
    # the committed q3_seeds configs bracket the 1-seed crossover at f = 0.02 and 0.05
    committed = sorted((Path(__file__).resolve().parent.parent / "configs/grid/q3_seeds").glob("*.yaml"))
    q3c = mc.stage_configs("q3_seeds", **{**kw, "crossover": (0.02, 0.05)})
    assert [(c.mix_f, c.seed) for _, c in q3c] == [(0.02, 1), (0.02, 2), (0.05, 1), (0.05, 2)]
    assert [p.name for p in committed] == [name for name, _ in q3c]
    assert [load_config(p) for p in committed] == [c for _, c in q3c]
    q4 = [c for _, c in mc.stage_configs("q4", **kw)]
    assert [c.wise_alpha > 0 for c in q4] == [True] * 6 + [False] * 3  # WiSE-FT first
    assert {c.init_from for c in q4[:6]} == {kw["sft_final"], kw["rft_final"]}
    assert [(c.lora_r, c.lr) for c in q4[6:8]] == [(64, 1e-4), (64, 3e-4)]
    assert q4[8].method == "distill" and q4[8].init_from == kw["sft_final"]
