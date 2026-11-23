"""Pilot run ids, the R36 launcher's shared-state plan, train-time export and the cost report."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from rlrazor import pipeline
from rlrazor.analysis.stats import filter_rows
from rlrazor.config import RunConfig, load_config, run_id

ROOT = Path(__file__).resolve().parent.parent


def _script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses look their module up in sys.modules
    spec.loader.exec_module(mod)
    return mod


def test_pilot_configs_are_full_length_middle_lr_and_never_grid_rows():
    sft = load_config(ROOT / "configs" / "pilot" / "sft_lr3e-5.yaml")
    grpo = load_config(ROOT / "configs" / "pilot" / "grpo_lr3e-6.yaml")
    assert (sft.method, sft.lr, sft.epochs, sft.n_ckpts, sft.smoke) == ("sft", 3e-5, 2, 2, False)
    assert (grpo.method, grpo.lr, grpo.grpo_steps, grpo.n_ckpts, grpo.smoke) == ("grpo", 3e-6, 400, 2, False)
    for cfg in (sft, grpo):
        assert cfg.arm == "pilot" and run_id(cfg).startswith("pilot-")
    # grid run ids are unchanged by the pilot prefix
    assert not run_id(RunConfig(arm="q0", method="sft", lr=3e-5)).startswith("pilot")
    # analysis never treats pilot rows as grid data (R30): dropped as an unknown arm
    rows = [{"run_id": "base", "arm": "base", "step": 0},
            {"run_id": run_id(sft), "arm": "pilot", "step": 468},
            {"run_id": "sft-lr3e-05-s0-abc", "arm": "q0", "step": 468}]
    kept, notes = filter_rows(rows)
    assert [r["arm"] for r in kept] == ["base", "q0"] and "pilot" in notes[0]


def test_launch_plans_shared_state_once():
    launch = _script("launch")
    mix = RunConfig(arm="q3", method="sft", lr=3e-5, mix_f=0.1)
    plain = RunConfig(arm="q0", method="sft", lr=3e-5)
    smoke = RunConfig(arm="q0", method="grpo", lr=3e-6, smoke=True)
    st = lambda base, pool: {"base_row": base, "general_pool": pool, "index": None}  # noqa: E731
    assert launch.prepare_plan([mix, plain, smoke], [st(None, False), st(None, False), st(None, False)]) == \
        {"pool": True, "base": [False, True]}
    assert launch.prepare_plan([mix, plain], [st({"x": 1}, True), st({"x": 1}, True)]) == {"pool": False, "base": []}


def test_run_require_prepared_refuses_missing_shared_state():
    run = _script("run")
    mix = RunConfig(arm="q3", method="sft", lr=3e-5, mix_f=0.1)
    assert run.unprepared(mix, {"base_row": None, "general_pool": False, "index": None}) == \
        ["the base row", "the general pool"]
    assert run.unprepared(mix, {"base_row": {}, "general_pool": True, "index": None}) == []
    # a finished run needs no pool any more
    assert run.unprepared(mix, {"base_row": {}, "general_pool": False, "index": {"ckpts": []}}) == []


def test_run_require_prepared_teacher_jobs():
    run = _script("run")
    tsft = RunConfig(arm="q0", method="teacher_sft", lr=3e-5)
    st = {"base_row": {}, "general_pool": False, "index": None, "teacher_data_done": False}
    assert run.unprepared(tsft, st) == ["the teacher data set (run the teacher_data config first)"]
    assert run.unprepared(tsft, {**st, "teacher_data_done": True}) == []
    assert run.unprepared(tsft, {**st, "index": {"ckpts": []}}) == []  # trained: evals only
    data = RunConfig(arm="q0", method="teacher_data", lr=0.0)
    assert run.unprepared(data, {**st, "base_row": None}) == []  # a data job needs no base row


def test_export_writes_train_times_with_config(tmp_path):
    cfg = RunConfig(arm="pilot", method="grpo", lr=3e-6, n_ckpts=2)
    rid = run_id(cfg)
    from rlrazor.config import dump_config
    dump_config(cfg, tmp_path / "ckpts" / rid / "config.yaml")
    (tmp_path / "cache" / rid).mkdir(parents=True)
    (tmp_path / "cache" / rid / "train_time.json").write_text(json.dumps({"wall_seconds": 4000.0}))
    pipeline.export(tmp_path)
    (line,) = (tmp_path / "train_times.jsonl").read_text().splitlines()
    t = json.loads(line)
    assert t["run_id"] == rid and t["wall_seconds"] == 4000.0 and t["config"]["method"] == "grpo"


def _measured():
    rows = [{"run_id": "base", "arm": "base", "step": 0, "gpu_seconds": 800.0, "smoke": False},
            {"run_id": "pilot-sft", "arm": "pilot", "step": 234, "gpu_seconds": 700.0, "smoke": False},
            {"run_id": "pilot-sft", "arm": "pilot", "step": 468, "gpu_seconds": 900.0, "smoke": False},
            {"run_id": "x", "arm": "q0", "step": 2, "gpu_seconds": 5.0, "smoke": True}]
    times = [{"run_id": "pilot-sft", "wall_seconds": 468 * 1.5, "config": {"method": "sft", "epochs": 2}},
             {"run_id": "pilot-grpo", "wall_seconds": 400 * 10.0, "config": {"method": "grpo", "grpo_steps": 400}},
             {"run_id": "smoke", "wall_seconds": 1e6, "config": {"method": "grpo", "smoke": True}}]
    return rows, times


def test_cost_report_units_prices_and_grid_counts():
    cr = _script("cost_report")
    rows, times = _measured()
    u = cr.measure(rows, times)
    assert u.eval_s == pytest.approx(800.0)  # checkpoint rows only (base and smoke excluded)
    assert u.sft_step_s == pytest.approx(1.5) and u.grpo_step_s == pytest.approx(10.0)
    p = cr.Prices(train_hr=3.6, eval_hr=0.36, overhead=0.0, container_s=0.0)
    assert p.train_usd(3600) == pytest.approx(3.6) and p.eval_usd(1000) == pytest.approx(0.1)
    assert cr.Prices(overhead=0.5, container_s=0.0, train_hr=1.0).train_usd(3600) == pytest.approx(1.5)
    proj = cr.project(u, p)
    # spec §6: 46 training runs; evals: 43 runs x 5 + 3 iterative x 4 rounds + 6 WiSE = 233
    assert (proj["full_grid"]["runs"], proj["full_grid"]["evals"]) == (46, 233)
    assert proj["stage1_q0_q1"]["runs"] + proj["remainder_q2_q4"]["runs"] == 46
    assert proj["staleness_optional"]["runs"] == 3
    cuts = [(c["runs"], c["evals"]) for c in proj["cut_order"]]
    assert cuts == [(46, 233), (46, 147), (42, 135), (36, 117), (34, 111)]
    totals = [c["total_usd"] for c in proj["cut_order"]]
    assert totals == sorted(totals, reverse=True)
    # GRPO group: 5 runs x 400 steps x 10 s = 20000 s at $3.6/h = $20
    g = next(x for x in proj["groups"] if x["label"].startswith("GRPO, LR"))
    assert g["train_usd"] == pytest.approx(20.0)
    assert proj["full_grid"]["eval_usd"] == pytest.approx(233 * 0.08)


def test_cost_report_env_prices(monkeypatch):
    cr = _script("cost_report")
    monkeypatch.setenv("RLRAZOR_GPU_PRICE_PER_HR", "3.0")
    monkeypatch.setenv("RLRAZOR_EVAL_GPU_PRICE_PER_HR", "0.5")
    p = cr.Prices.from_env()
    assert (p.train_hr, p.eval_hr) == (3.0, 0.5)


def test_cost_report_prints_assumptions_and_tags_estimated_groups(capsys):
    cr = _script("cost_report")
    rows, times = _measured()
    out = cr.report(rows, times, cr.Prices(), spent=7.4, budget=75.0, grpo_steps=400)
    text = capsys.readouterr().out
    line = next(x for x in text.splitlines() if x.startswith("ASSUMPTIONS"))
    for needle in ("10 s per 1k 0.5B vLLM samples", "120 s per GKD step", "3 x cost of a Tulu row", "0.9 RFT keep",
                   "teacher qwen2.5-7b-instruct", "teacher r1-distill-qwen-7b"):
        assert needle in line
    est = {g["label"]: g["estimated"] for g in out["groups"]}
    assert est["GRPO, LR 1e-5/3e-5/5e-5 + 2 seeds"] is False
    assert all(v for k, v in est.items() if k.startswith(("teacher data", "SFT(teacher)", "offline RFT",
                                                         "iterative RFT", "on-policy distillation", "LoRA")))
    assert "est. q1 iterative RFT" in text and "STAGE 1 (Q0+Q1) at 3 ckpts/run" in text


def test_projection_q0_amended_and_teachers():
    """Q0 = SFT(teacher) 5 + GRPO 5 (no SFT(gold) arm); stage 1 printed for both teacher choices."""
    cr = _script("cost_report")
    u = cr.measure(*_measured())
    p = cr.Prices()
    proj = cr.project(u, p)
    q0 = [g for g in proj["groups"] if g["arm"] == "q0"]
    assert [g["runs"] for g in q0] == [1, 5, 5] and not any("gold" in g["label"] for g in q0)
    assert proj["stage1_q0_q1_3ckpts"]["runs"] == 22 and proj["stage1_q0_q1_3ckpts"]["evals"] == 30 + 39
    by = proj["stage1_by_teacher"]
    assert set(by) == {"qwen2.5-7b-instruct", "r1-distill-qwen-7b"}
    assert by["r1-distill-qwen-7b"]["3ckpts"]["total_usd"] > by["qwen2.5-7b-instruct"]["3ckpts"]["total_usd"]
    # teacher data generation: adaptive 4 + 12 * unsolved samples per prompt
    t = cr.TEACHERS["qwen2.5-7b-instruct"]
    assert cr.teacher_data_seconds(p) == pytest.approx(
        3 * p.engine_start_s + (4 + 12 * t["unsolved_after_4"]) * cr.N_TRAIN / 1000 * t["gen_s_per_1k"])
    # R1-distill rows are longer, so SFT(teacher) costs more per run
    assert cr.train_seconds("teacher_sft", u, p, teacher="r1-distill-qwen-7b")[0] > \
        cr.train_seconds("teacher_sft", u, p)[0]


def test_projection_includes_round_overhead_and_cpu_jobs():
    cr = _script("cost_report")
    u = cr.measure(*_measured())
    p0, p = cr.Prices(engine_start_s=0.0), cr.Prices(engine_start_s=90.0)
    # every iterative-RFT round restarts generation: 4 rounds -> 4 engine starts
    assert cr.train_seconds("rft_iter", u, p)[0] - cr.train_seconds("rft_iter", u, p0)[0] == pytest.approx(360)
    groups = {g.label: g for g in cr.grid(u, p)}
    pool = next(g for k, g in groups.items() if k.startswith("general (Tulu) pool"))
    wise = next(g for k, g in groups.items() if k.startswith("WiSE-FT"))
    assert pool.cpu_usd == pytest.approx(p.cpu_usd(p.pool_cpu_s)) and pool.cpu_usd > 0 and not pool.is_run
    assert wise.cpu_usd > 0 and wise.count == 6
    assert p.cpu_usd(3600 - p.container_s) == pytest.approx(4 * p.cpu_core_hr + 16 * p.mem_gib_hr)


def test_run_cost_per_config():
    cr = _script("cost_report")
    u = cr.measure(*_measured())
    p = cr.Prices(overhead=0.0, container_s=0.0, train_hr=3.6, eval_hr=0.36)
    grpo = RunConfig(arm="q0", method="grpo", lr=1e-5, n_ckpts=3)
    usd, est = cr.run_cost(grpo, u, p)
    assert usd == pytest.approx(400 * 10.0 / 1000 + 3 * 0.08) and est is False
    assert cr.run_cost(grpo, u, p, train=False, n_evals=1)[0] == pytest.approx(0.08)
    it = RunConfig(arm="q1", method="rft_iter", lr=1e-5, rounds=4, n_ckpts=3)
    assert cr.run_cost(it, u, p, train=False)[0] == pytest.approx(4 * 0.08)  # one eval per round
    with pytest.raises(cr.NoCostModel):
        cr.run_cost({"method": "teacher_gen_only"}, u, p)


def test_run_cost_teacher_mix_and_wise():
    """Q3 mixing adds Tulu rows to the teacher set (cost grows with f); WiSE-FT is a CPU interpolation + 1 eval."""
    cr = _script("cost_report")
    u = cr.measure(*_measured())
    p = cr.Prices(overhead=0.0, container_s=0.0, train_hr=3.6, eval_hr=0.36)
    costs = [cr.train_seconds("teacher_sft", u, p, mix_f=f)[0] for f in (0.0, 0.02, 0.1, 0.25, 0.5)]
    assert costs == sorted(costs) and costs[-1] > 2.5 * costs[0]  # (1.8 + 3) / 1.8 at f = 0.5
    wise = RunConfig(arm="q4", method="teacher_sft", lr=3e-6, init_from="x-s0/step-9", wise_alpha=0.5, n_ckpts=3)
    usd, est = cr.run_cost(wise, u, p)
    assert usd == pytest.approx(0.08 + p.cpu_usd(p.wise_cpu_s)) and est is True
    assert cr.run_cost(wise, u, p, train=False, n_evals=0)[0] == 0.0
