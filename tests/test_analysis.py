import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rlrazor.analysis.stats import (
    best_setting_rows,
    filter_rows,
    fit_exponential,
    forgetting_at_accuracy,
    mixing_crossover,
    near_frontier,
    pareto_frontier,
    quadratic_r2,
    frontier_interp,
    seed_band,
)

ROOT = Path(__file__).resolve().parents[1]
ANALYZE = ROOT / "scripts" / "analyze.py"
FIGURES = ["frontier_q0", "kl_vs_forgetting", "q1_selfgen", "q2_noise", "q3_mixing", "q4_mitigations"]
BENCH = ("hellaswag", "mmlu", "truthfulqa", "winogrande", "ifeval", "humaneval")


def test_pareto_frontier_simple():
    pts = [(0.4, -0.01), (0.5, -0.05), (0.45, -0.10)]
    front = pareto_frontier(pts)
    assert (0.45, -0.10) not in front
    assert sorted(front) == [(0.4, -0.01), (0.5, -0.05)]


def test_near_frontier_keeps_points_within_tolerance():
    pts = [(0.4, -0.01), (0.5, -0.05), (0.49, -0.06), (0.45, -0.10)]
    near = near_frontier(pts, tol=0.02)
    assert (0.49, -0.06) in near  # 1 acc-point left of the frontier
    assert (0.45, -0.10) not in near  # 5 acc-points left of the frontier


def test_fit_exponential_passes_near_points():
    xs = np.linspace(0.3, 0.6, 6)
    pts = [(x, 0.01 - 0.002 * np.exp(8 * (x - 0.3))) for x in xs]
    f = fit_exponential(pts)
    for x, y in pts:
        assert f(x) == pytest.approx(y, abs=1e-4)


def test_forgetting_at_accuracy_out_of_range_is_none():
    pts = [(0.30, 0.0), (0.40, -0.02), (0.50, -0.05)]
    out = forgetting_at_accuracy(pts, [0.25, 0.40, 0.55])
    assert out[0] is None and out[2] is None
    assert out[1] == pytest.approx(-0.02, abs=1e-3)


def test_quadratic_r2_perfect():
    x = np.linspace(0, 2, 20)
    coef, r2 = quadratic_r2(x, x**2)
    assert r2 > 0.999
    assert coef[0] == pytest.approx(1.0)


def test_mixing_crossover():
    by_f = {0: -0.1, 0.02: -0.06, 0.05: -0.02, 0.1: -0.01}
    assert mixing_crossover(by_f, (-0.015, 0.01)) == 0.05
    assert mixing_crossover(by_f, (0.0, 0.001)) is None


def test_seed_band_over_seeds():
    rows = []
    for seed, slope in ((0, -0.10), (1, -0.20), (2, -0.30)):
        for step, acc in ((1, 0.35), (2, 0.40), (3, 0.45)):
            rows.append(_row(f"sft-s{seed}", "q0", "sft", acc, slope * (acc - 0.3), seed=seed, step=step))
    band = seed_band(rows, "sft", 0.40, base_point=(0.3, 0.0))
    assert band.mean == pytest.approx(-0.02, abs=1e-3)
    assert band.std == pytest.approx(0.01, abs=1e-3)
    assert (band.k, band.n, band.target, band.partial) == (3, 3, 0.40, False)


def test_seed_band_does_not_drop_seeds_that_miss_the_target():
    """Reviewer case: 3 GRPO seeds, only seed 0 reaches 45%. The band must not report seed 0 alone as a
    3-seed band with std 0: it moves to the highest target every seed reaches, or is marked partial."""
    rows = []
    for seed, (slope, top) in enumerate(((-0.05, 0.47), (-0.10, 0.43), (-0.15, 0.43))):
        for step, acc in enumerate((0.33, 0.36, 0.40, top), start=1):
            rows.append(_row(f"grpo-s{seed}", "q0", "grpo", acc, slope * (acc - 0.3), seed=seed, step=step))
    targets = [0.33, 0.36, 0.39, 0.42, 0.45]
    band = seed_band(rows, "grpo", targets, base_point=(0.3, 0.0))
    assert (band.target, band.k, band.n) == (0.42, 3, 3)
    assert band.mean == pytest.approx(-0.10 * 0.12, abs=1e-6)
    assert band.std == pytest.approx(0.05 * 0.12, abs=1e-6)
    only45 = seed_band(rows, "grpo", [0.45], base_point=(0.3, 0.0))
    assert (only45.k, only45.n, only45.partial, only45.std) == (1, 3, True, None)


def test_fit_exponential_saturating_frontier():
    """Reviewer case: a saturating frontier must be fit with b < 0, not a line that overshoots."""
    pts = [(0.30, 0.0), (0.33, -0.030), (0.36, -0.045), (0.39, -0.052), (0.42, -0.055), (0.45, -0.056)]
    f = fit_exponential(pts)
    assert f.kind == "exponential"
    for x, y in pts:
        assert f(x) == pytest.approx(y, abs=0.003)  # within 0.3 pp of every point
    assert f.rmse < 0.003
    assert min(f(x) for x in np.linspace(0.30, 0.45, 50)) >= -0.059


def test_frontier_interp_follows_observed_frontier():
    pts = [(0.30, 0.0), (0.40, -0.02), (0.40, -0.05), (0.50, -0.04)]
    out = frontier_interp(pts, [0.25, 0.35, 0.45, 0.55])
    assert out[0] is None and out[3] is None
    assert out[1] == pytest.approx(-0.01) and out[2] == pytest.approx(-0.03)


def test_filter_rows_drops_smoke_only_when_real_rows_exist():
    real, smoke = _row("a", "q0", "sft", 0.4, -0.01), {**_row("b", "q0", "sft", 0.4, -0.01), "smoke": True}
    rows, notes = filter_rows([real, smoke])
    assert rows == [real] and any("smoke" in n for n in notes)
    rows, _ = filter_rows([smoke])
    assert rows == [smoke]


def test_best_setting_prefers_most_seeds_then_highest_accuracy():
    rows = [_row(f"a-s{s}", "q0", "sft", 0.40, -0.01, lr=1e-5, seed=s) for s in range(3)]
    rows += [_row(f"b-s{s}", "q0", "sft", 0.50, -0.01, lr=3e-5, seed=s) for s in range(2)]
    assert {r["lr"] for r in best_setting_rows(rows, "sft")} == {1e-5}  # 3 seeds beat higher accuracy
    rows += [_row("b-s2", "q0", "sft", 0.50, -0.01, lr=3e-5, seed=2)]
    assert {r["lr"] for r in best_setting_rows(rows, "sft")} == {3e-5}  # tie on seeds: higher accuracy


# ---------- synthetic results and the CLI ----------


def _row(run_id, arm, method, acc, forg, *, lr=1e-5, seed=0, step=1, noise_p=0.0, mix_f=0.0,
         lora=0, wise_alpha=0.0, init_from=None, kl=None):
    return {
        "run_id": run_id, "arm": arm, "method": method, "lr": lr, "seed": seed,
        "noise_p": noise_p, "mix_f": mix_f, "lora": lora, "wise_alpha": wise_alpha,
        "init_from": init_from, "step": step, "frac_trained": step / 5,
        "gsm8k_acc": acc, "kl_fwd": kl if kl is not None else max(0.0, -forg) * 20,
        "bench": {b: 0.5 + forg + 0.001 * i for i, b in enumerate(BENCH)},
        "forgetting_mean": forg, "gpu_seconds": 1.0, "smoke": False,
        "lm_eval_versions": {}, "lm_eval_package": "0.4.13",
    }


def _run(rows, run_id, arm, method, slope, accs, **kw):
    for i, a in enumerate(accs, start=1):
        rows.append(_row(run_id, arm, method, a, slope * (a - 0.30), step=i, **kw))


def _synthetic(only_q0=False):
    rows = [_row("base", "base", "sft", 0.30, 0.0, step=0, kl=0.0)]
    accs = [0.33, 0.36, 0.40, 0.44, 0.47]
    for lr, slope in ((1e-5, -0.30), (3e-5, -0.40)):
        for seed in (0, 1, 2) if lr == 1e-5 else (0,):
            _run(rows, f"sft-lr{lr:g}-s{seed}", "q0", "sft", slope - 0.02 * seed, accs, lr=lr, seed=seed)
    for lr, slope in ((1e-6, -0.05), (3e-6, -0.08)):
        for seed in (0, 1, 2) if lr == 3e-6 else (0,):
            _run(rows, f"grpo-lr{lr:g}-s{seed}", "q0", "grpo", slope - 0.01 * seed, accs, lr=lr, seed=seed)
    if only_q0:
        return rows
    for m, slope in (("rft_offline", -0.10), ("rft_iter", -0.07), ("self_rewrite", -0.2), ("rl_distill_sft", -0.06)):
        for seed in (0, 1, 2):
            _run(rows, f"{m}-s{seed}", "q1", m, slope - 0.01 * seed, accs, seed=seed)
    for p in (0.2, 0.4):
        for seed in (0, 1, 2):
            _run(rows, f"sft-n{p}-s{seed}", "q2", "sft", -0.4 - p, [a - p / 4 for a in accs], noise_p=p, seed=seed)
            _run(rows, f"grpo-n{p}-s{seed}", "q2", "grpo", -0.1 - p / 10, [a - p / 20 for a in accs],
                 lr=3e-6, noise_p=p, seed=seed)
    for f in (0.02, 0.05, 0.10, 0.25, 0.50):
        _run(rows, f"sft-f{f}-s0", "q3", "sft", -0.30 * (1 - f) ** 6, accs, mix_f=f)
    for src, slope in (("sft-lr1e-05-s0", -0.30), ("rft_offline-s0", -0.10)):
        for alpha in (0.25, 0.5, 0.75):
            rows.append(_row(f"{src}-w{alpha}", "q4", src.split("-")[0], 0.30 + 0.17 * alpha,
                             slope * 0.17 * alpha**2, step=5, wise_alpha=alpha, init_from=f"{src}/step-5"))
    for lr in (1e-4, 3e-4):
        _run(rows, f"lora-lr{lr:g}", "q4", "lora_sft", -0.15, accs, lr=lr, lora=64)
    _run(rows, "distill-s0", "q4", "distill", -0.05, [0.45, 0.43, 0.42], init_from="sft-lr1e-05-s0/step-5")
    rows.append(_row("async-s0", "q5", "grpo", 0.4, -0.01))  # unknown arm: ignored with a note
    return rows


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _analyze(*args):
    return subprocess.run([sys.executable, str(ANALYZE), *map(str, args)], capture_output=True, text=True,
                          cwd=ROOT)


def test_analyze_runs_on_synthetic_results(tmp_path, analyze_mod):
    res = tmp_path / "results.jsonl"
    _write_jsonl(res, _synthetic())
    out = tmp_path / "figures"
    proc = _analyze("--results", res, "--out", out)
    assert proc.returncode == 0, proc.stderr
    for name in FIGURES:
        assert (out / f"{name}.png").stat().st_size > 0
        assert (out / f"{name}.pdf").stat().st_size > 0
    md = (out / "tables.md").read_text()
    tex = (out / "tables.tex").read_text()
    assert "Matched accuracy" in md and "hellaswag" in md
    assert "\\begin{tabular}" in tex and "\\toprule" in tex
    assert "q5" in proc.stdout + proc.stderr  # unknown arm is logged, not fatal
    assert "no data yet" not in md.lower()
    # tables.tex carries the numbers of tables.md, except the md-only tables (MD_ONLY: repeated in the write-up's
    # body, or page-long), whose titles do not appear in tex
    import re
    num = re.compile(r"-?\d+\.\d+")
    assert set(num.findall(tex)) <= set(num.findall(md))
    md_only = [t for t in re.findall(r"^\*\*(.+)\*\*$", md, flags=re.M) if t.startswith(analyze_mod.MD_ONLY)]
    assert md_only and not any(t in tex for t in md_only)
    nums = json.loads((out / "numbers.json").read_text())
    assert set(nums["results"]) | set(nums["constants"]) == set(num.findall(md))
    assert nums["smoke"] is False and "0.0001" in nums["constants"]  # LoRA lr: a hyperparameter, not a result
    assert "0.0001" not in nums["results"]


def test_analyze_with_only_q0_rows(tmp_path):
    res = tmp_path / "results.jsonl"
    _write_jsonl(res, _synthetic(only_q0=True))
    out = tmp_path / "figures"
    proc = _analyze("--results", res, "--out", out)
    assert proc.returncode == 0, proc.stderr
    for name in FIGURES:
        assert (out / f"{name}.png").exists() and (out / f"{name}.pdf").exists()
    md = (out / "tables.md").read_text()
    assert "no data yet" in md.lower()
    assert "Matched accuracy" in md


@pytest.fixture
def analyze_mod(monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location("analyze", ANALYZE)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "analyze", mod)  # dataclasses resolve their module
    spec.loader.exec_module(mod)
    return mod


def _linear():
    """Closed-form data: forgetting = slope · (acc − 0.30) along every run, base at (0.30, 0)."""
    rows = [_row("base", "base", "sft", 0.30, 0.0, step=0, kl=0.0)]
    accs = [0.33, 0.36, 0.40, 0.44, 0.47]
    for seed in (0, 1, 2):
        _run(rows, f"sft-lr1e-05-s{seed}", "q0", "sft", -0.30, accs, lr=1e-5, seed=seed)
        _run(rows, f"grpo-lr3e-06-s{seed}", "q0", "grpo", -0.05 * (seed + 1), accs, lr=3e-6, seed=seed)
    _run(rows, "grpo-lr1e-06-s0", "q0", "grpo", -0.20, [0.32, 0.34], lr=1e-6)  # dominated, 1 seed
    # a mis-armed WiSE-FT row in q0 that would dominate the SFT frontier if it were let in
    rows.append(_row("sft-lr1e-05-s0-w0.5", "q0", "sft", 0.46, 0.0, step=5, wise_alpha=0.5,
                     init_from="sft-lr1e-05-s0/step-5"))
    _run(rows, "grpo-lr1e-06-n0.2-s0", "q2", "grpo", -0.10, accs, lr=1e-6, noise_p=0.2)
    _run(rows, "sft-lr1e-05-n0.2-s0", "q2", "sft", -0.50, accs, lr=1e-5, noise_p=0.2)
    for f, slope in ((0.02, -0.20), (0.05, -0.14), (0.10, -0.10)):
        _run(rows, f"sft-lr1e-05-f{f}-s0", "q3", "sft", slope, accs, mix_f=f)
    rows.append(_row("sft-lr1e-05-s0-w0.5b", "q4", "sft", 0.385, -0.01, step=5, wise_alpha=0.5,
                     init_from="sft-lr1e-05-s0/step-5"))
    return rows


def test_analyze_numbers_closed_form(tmp_path, analyze_mod):
    s = analyze_mod.analyze(_linear(), tmp_path)
    q0 = s["q0"]
    assert q0["targets"] == pytest.approx([0.33, 0.36, 0.39, 0.42, 0.45])
    # SFT pooled frontier is the -0.30 line; the mis-armed WiSE row (0.46, 0.0) is excluded (R30)
    assert (0.46, 0.0) not in q0["points"]["sft"]
    assert any("excluded 1 q0 row" in n for n in s["notes"])
    assert q0["fit"]["sft"][2] == pytest.approx(-0.30 * 0.09, abs=1e-6)
    assert q0["interp"]["sft"][2] == pytest.approx(-0.30 * 0.09, abs=1e-9)
    assert q0["interp"]["grpo"][2] == pytest.approx(-0.05 * 0.09, abs=1e-9)  # frontier = best GRPO seed
    band = q0["bands"]["grpo"]
    assert (band.target, band.k, band.n) == (pytest.approx(0.45), 3, 3)
    assert band.mean == pytest.approx(-0.10 * 0.15, abs=1e-6)
    assert band.std == pytest.approx(0.05 * 0.15, abs=1e-6)
    # Q2: p = 0 baseline is the Q0 run at the noisy runs' LR (lr 1e-6, one seed), not GRPO's best LR
    assert s["q2"]["grpo"][0.0]["run_ids"] == ["grpo-lr1e-06-s0"]
    assert s["q2"]["sft"][0.0]["n"] == 3
    assert s["q2"]["sft"][0.0]["forgetting"][0] == pytest.approx(-0.30 * 0.17)
    # Q3: at 45% f=0.05 gives -0.021 >= GRPO mean - std = -0.0225; f=0.02 gives -0.03
    q3 = s["q3"]
    assert q3["target"] == pytest.approx(0.45) and q3["crossover"] == 0.05
    assert q3["by_f"][0.0]["k"] == q3["by_f"][0.0]["n"] == 3
    assert q3["by_f"][0.05]["mean"] == pytest.approx(-0.14 * 0.15, abs=1e-6)
    # R54 secondary criterion: the same rule on IFEval alone (in this data IFEval moves with the mean)
    assert q3["crossover_ifeval"] == 0.05
    assert "Secondary criterion (pre-registered, R54): IFEval change" in (tmp_path / "tables.md").read_text()
    # the primary criterion's margin vs the threshold is printed with 3 decimals
    md = (tmp_path / "tables.md").read_text()
    thr = 100 * (band.mean - band.std)  # -2.25 pp
    assert s["q3_detail"]["threshold"] == pytest.approx(band.mean - band.std)
    assert f"| 0.05 | 1/1 | -2.100 | {thr:.3f} | {-2.1 - thr:.3f} |" in md
    assert "Primary criterion in detail" in md and "GRPO − sft (pp)" in md
    # Q4: WiSE-FT path links to its source run exactly: alpha 1 is that run's final checkpoint
    method, path = s["q4"]["wise"]["sft-lr1e-05-s0"]
    assert method == "sft" and (1.0, 0.47, pytest.approx(-0.30 * 0.17)) in path
    assert (0.0, 0.30, 0.0) in path and (0.5, 0.385, -0.01) in path


def test_teacher_sft_is_the_sft_reference(tmp_path, analyze_mod):
    """R42/R43: with SFT on teacher answers (method teacher_sft) as the SFT arm, Q0's gap column, Q3's f = 0
    baseline and Q4's LoRA rows (teacher_sft + lora) use it exactly as they used SFT (gold)."""
    rows = []
    for r in _linear():
        r = dict(r)
        if r["method"] == "sft" and r["arm"] != "base":
            r["method"] = "teacher_sft"
        rows.append(r)
    accs = [0.33, 0.36, 0.40]
    _run(rows, "teacher_sft-lr1e-4-r64", "q4", "teacher_sft", -0.15, accs, lr=1e-4, lora=64)
    s = analyze_mod.analyze(rows, tmp_path)
    assert s["q0"]["interp"]["teacher_sft"][2] == pytest.approx(-0.30 * 0.09, abs=1e-9)
    assert s["q3"]["crossover"] == 0.05 and s["q3"]["by_f"][0.0]["k"] == 3
    md = (tmp_path / "tables.md").read_text()
    assert "teacher_sft − grpo interp. (pp)" in md
    assert "r=64" in md  # the LoRA table picks up teacher_sft LoRA rows
    assert analyze_mod.sft_ref(["grpo", "sft"]) == "sft"


def test_diverged_runs_are_excluded_and_listed(tmp_path, analyze_mod):
    """R47/R50: flagged (dead-run stop) or collapsed (final GSM8K < 1%) runs leave the frontier and the KL fit
    and are listed in an appendix with the pooled R² including them."""
    rows = _linear()
    _run(rows, "grpo-lr5e-05-s0", "q0", "grpo", -0.09, [0.0, 0.0, 0.0], lr=5e-5)  # legacy collapse: no flag
    _run(rows, "grpo-lr1e-05-s0", "q0", "grpo", -0.02, [0.35, 0.2], lr=1e-5)
    for r in rows:
        if r["run_id"] == "grpo-lr1e-05-s0":
            r["diverged"] = True
    s = analyze_mod.analyze(rows, tmp_path)
    assert s["diverged"] == ["grpo-lr1e-05-s0", "grpo-lr5e-05-s0"]
    pts = s["q0"]["points"]["grpo"]
    assert not any(p[0] in (0.0, 0.2) for p in pts)  # neither diverged run is on the GRPO curve
    md = (tmp_path / "tables.md").read_text()
    assert "Appendix: diverged runs" in md and "grpo-lr5e-05-s0" in md
    assert "kl_r2_with_diverged" in s and any("diverged run" in n for n in s["notes"])
    assert analyze_mod.analyze(_linear(), tmp_path / "clean").get("diverged") is None


def test_best_setting_per_benchmark_table(tmp_path, analyze_mod):
    """Per-benchmark change at each method's best setting, final checkpoints, mean ± sd over
    seeds, with an uncorrected Welch p vs GRPO (post hoc)."""
    from scipy.stats import ttest_ind

    base = _row("base", "base", "sft", 0.30, 0.0, step=0, kl=0.0)
    base["bench"] = {b: 0.5 for b in BENCH}
    rows = [base]
    ifeval = {"grpo": [0.018, 0.015, 0.009], "teacher_sft": [-0.044, -0.039, -0.022]}
    for m, vals in ifeval.items():
        for seed, v in enumerate(vals):
            for step in (1, 2):  # only the final checkpoint counts
                r = _row(f"{m}-s{seed}", "q0", m, 0.40 + 0.02 * seed + 0.03 * step, -0.01 * step - 0.003 * seed,
                         lr=1e-6, seed=seed, step=step)
                r["bench"] = {b: 0.5 for b in BENCH}
                r["bench"]["ifeval"] = 0.5 + (v if step == 2 else -0.3)
                rows.append(r)
    s = analyze_mod.analyze(rows, tmp_path)
    bb = s["best_bench"]
    mean, sd, n = bb["teacher_sft"]["ifeval"]
    assert n == 3 and mean == pytest.approx(np.mean(ifeval["teacher_sft"])) and sd == pytest.approx(
        np.std(ifeval["teacher_sft"], ddof=1))
    want = ttest_ind(ifeval["teacher_sft"], ifeval["grpo"], equal_var=False).pvalue
    assert bb["p_vs_grpo"]["teacher_sft"]["ifeval"] == pytest.approx(want)
    md = (tmp_path / "tables.md").read_text()
    assert "Per-benchmark change at each method's best setting" in md and "-3.5 ± 1.2" in md


def test_collect_numbers_separates_constants(analyze_mod):
    t = analyze_mod.Table("t", ["p"], [[analyze_mod.par(0.2), "1.5"]])
    sec = analyze_mod.Section("s", [t, ("f = ", analyze_mod.Const("0.05"), " gives ", "-2.5")])
    results, consts = analyze_mod.collect_numbers([sec], [])
    assert results == {"1.5", "-2.5"} and consts == {"0.2", "0.05"}


def _numbers(out, results, constants, smoke=False):
    out.mkdir(parents=True, exist_ok=True)
    (out / "numbers.json").write_text(json.dumps({"smoke": smoke, "results": results, "constants": constants}))


@pytest.mark.parametrize("text,code", [
    ("acc 45.0", 0), ("acc -45.0", 1),      # never add a sign
    ("forgets 4.5 pp", 0), ("-4.5", 0),     # magnitude of a negative result is fine
    ("p = 0.2", 0), ("drop -0.2", 1),       # constants: accepted (listed), never sign-added
    ("12.34", 1),
])
def test_check_writeup_sign_rules(tmp_path, text, code):
    out = tmp_path / "fig"
    _numbers(out, ["45.0", "-4.5"], ["0.2"])
    w = tmp_path / "w.tex"
    w.write_text(text)
    assert _analyze("--check-writeup", w, "--out", out).returncode == code


def test_check_writeup_lists_constant_matches(tmp_path):
    out = tmp_path / "fig"
    _numbers(out, ["45.0"], ["0.2"])
    w = tmp_path / "w.md"
    w.write_text("SFT forgets 0.2 pp more")  # 0.2 only exists as a hyperparameter (p = 0.2)
    proc = _analyze("--check-writeup", w, "--out", out)
    assert proc.returncode == 0 and "matched constant: 0.2" in proc.stdout


def test_check_writeup_refuses_smoke_tables(tmp_path):
    out = tmp_path / "fig"
    _numbers(out, ["45.0"], [], smoke=True)
    w = tmp_path / "w.tex"
    w.write_text("45.0")
    proc = _analyze("--check-writeup", w, "--out", out)
    assert proc.returncode == 3 and "smoke" in proc.stdout
    res = tmp_path / "results.jsonl"  # --results regenerates the tables first: smoke rows -> smoke tables
    _write_jsonl(res, [{**r, "smoke": True} for r in _synthetic(only_q0=True)])
    assert _analyze("--check-writeup", w, "--out", tmp_path / "fresh", "--results", res).returncode == 3


def test_check_writeup_regenerates_from_results(tmp_path):
    res = tmp_path / "results.jsonl"
    _write_jsonl(res, _synthetic(only_q0=True))
    w = tmp_path / "w.tex"
    w.write_text("base accuracy 30.0")
    proc = _analyze("--check-writeup", w, "--out", tmp_path / "fresh", "--results", res)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (tmp_path / "fresh" / "tables.tex").exists()


@pytest.mark.parametrize("arm", ["q0", "q1", "q2", "q3", "q4"])
def test_analyze_each_arm_alone_without_base(tmp_path, analyze_mod, arm):
    """Partial grids (any single arm, no base row, no Q0 reference) render with notes, never crash."""
    rows = [r for r in _synthetic() if r["arm"] == arm]
    notes = analyze_mod.analyze(rows, tmp_path)["notes"]
    assert any("no base row" in n for n in notes)
    for name in FIGURES:
        assert (tmp_path / f"{name}.png").exists()
    assert (tmp_path / "tables.tex").exists()


def test_check_writeup_flags_unknown_number(tmp_path):
    res = tmp_path / "results.jsonl"
    _write_jsonl(res, _synthetic(only_q0=True))
    out = tmp_path / "figures"
    assert _analyze("--results", res, "--out", out).returncode == 0
    known = __import__("re").findall(r"-?\d+\.\d+", (out / "tables.md").read_text())[0]
    ok = tmp_path / "ok.md"
    ok.write_text(f"SFT reaches {known} here.\n")
    assert _analyze("--check-writeup", ok, "--out", out).returncode == 0
    bad = tmp_path / "WRITEUP.md"
    bad.write_text(f"SFT reaches {known} but forgets 987.65 points.\n")
    proc = _analyze("--check-writeup", bad, "--out", out)
    assert proc.returncode == 1
    assert "987.65" in proc.stdout


def test_check_writeup_accepts_tex(tmp_path):
    res = tmp_path / "results.jsonl"
    _write_jsonl(res, _synthetic(only_q0=True))
    out = tmp_path / "figures"
    assert _analyze("--results", res, "--out", out).returncode == 0
    known = __import__("re").findall(r"-?\d+\.\d+", (out / "tables.tex").read_text())[0]
    tex = tmp_path / "WRITEUP.tex"
    tex.write_text(f"\\section{{Results}} SFT reaches ${known}$\\%.\n")
    assert _analyze("--check-writeup", tex, "--out", out).returncode == 0
    tex.write_text(f"SFT reaches ${known}$ and $12.345$.\n")
    proc = _analyze("--check-writeup", tex, "--out", out)
    assert proc.returncode == 1 and "12.345" in proc.stdout


def test_check_writeup_without_tables_fails(tmp_path):
    w = tmp_path / "WRITEUP.tex"
    w.write_text("1.5\n")
    proc = _analyze("--check-writeup", w, "--out", tmp_path / "missing")
    assert proc.returncode != 0


# ---------- headline, KL detail, logs, cost ----------


def _kl_row(run_id, method, sums, tokens, *, arm="q0", acc=0.5, forg=0.0, lr=1e-6, seed=0, step=1):
    r = _row(run_id, arm, method, acc, forg, lr=lr, seed=seed, step=step, kl=float(np.mean(sums)))
    r["kl_logratio_sums"], r["kl_n_tokens"] = list(sums), list(tokens)
    return r


def test_kl_detail_statistics(analyze_mod):
    sums = [float(x) for x in range(1, 11)]  # 10 samples, 1..10 nats, longer samples drift more
    tokens = [10 * x for x in range(1, 11)]
    d = analyze_mod.kl_detail(_kl_row("a", "grpo", sums, tokens))
    assert d["mean"] == pytest.approx(5.5) and d["median"] == pytest.approx(5.5)
    assert d["per_token"] == pytest.approx(55 / 550) and d["median_per_token"] == pytest.approx(0.1)
    assert d["top10_share"] == pytest.approx(10 / 55)  # the top 1 of 10 samples
    assert d["frac_neg"] == 0.0 and d["rho_len"] == pytest.approx(1.0)
    neg = analyze_mod.kl_detail(_kl_row("b", "grpo", [-1.0, 3.0], [5, 10]))
    assert neg["frac_neg"] == 0.5 and neg["per_token"] == pytest.approx(2 / 15)
    base = _row("base", "base", "sft", 0.3, 0.0, step=0, kl=0.0)
    assert analyze_mod.kl_detail(base)["per_token"] == 0.0  # KL 0 by definition
    assert analyze_mod.kl_detail(_row("c", "q0", "grpo", 0.5, 0.0)) is None  # no saved detail


def test_top_kl_samples_and_rank_agreement(analyze_mod):
    a = _kl_row("a", "grpo", [5.0, 1.0, 9.0, 3.0], [10, 20, 30, 40])
    b = _kl_row("b", "teacher_sft", [0.5, 4.0, 1.0, 2.0], [10, 20, 30, 40])
    top = analyze_mod.top_kl_samples(a, b, k=2)
    assert [t["index"] for t in top] == [2, 0]
    assert top[0]["per_token"] == pytest.approx(9 / 30) and top[0]["other"] == 1.0 and top[0]["rank_other"] == 3
    assert analyze_mod.kl_rank_agreement(a, a) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        analyze_mod.top_kl_samples(a, _kl_row("c", "grpo", [1.0, 2.0, 3.0, 4.0], [1, 2, 3, 4]))


def test_kl_fit_subsets_headline_is_q0_only(analyze_mod):
    """R46/M2: the headline R² is the Q0-only fit (teacher SFT + GRPO + base), not the pooled one."""
    rows = [_row("base", "base", "sft", 0.30, 0.0, step=0, kl=0.0)]
    rng = np.random.default_rng(0)
    for i in range(6):
        rows.append(_row(f"t{i}", "q0", "teacher_sft", 0.4, -0.01 * i, lr=1e-5, step=i + 1, kl=2.0 * i))
        rows.append(_row(f"g{i}", "q0", "grpo", 0.4, 0.001 * i, lr=1e-6, step=i + 1, kl=5.0 * i))
        rows.append(_row(f"r{i}", "q1", "rft_offline", 0.4, float(rng.normal(0, 0.02)), lr=3e-6, step=i + 1,
                         kl=float(rng.uniform(0, 30))))
    fits = analyze_mod.kl_fit_subsets(rows)
    head = [f for f in fits if f[4]]
    assert len(head) == 1 and head[0][0].startswith("Q0") and head[0][1] == 13
    q0 = [r for r in rows if r["arm"] in ("q0", "base")]
    _, want = quadratic_r2([r["kl_fwd"] for r in q0], [r["forgetting_mean"] for r in q0])
    assert head[0][2] == pytest.approx(want)
    pooled = [f for f in fits if f[0].startswith("Q0 + Q1")][0]
    assert pooled[1] == 19 and pooled[2] != pytest.approx(want)
    low = [f for f in fits if f[0] == "SFT-like, LR ≤ 3e-6"][0]
    assert low[1] == 6  # only the 3e-6 RFT rows


def test_headline_table_welch_and_two_decimals(analyze_mod):
    from scipy.stats import ttest_ind

    base = _row("base", "base", "sft", 0.30, 0.0, step=0, kl=0.0)
    g, s = [0.0025, 0.0010, 0.0040], [-0.0005, 0.0010, -0.0020]
    rows = [base]
    for m, vals in (("grpo", g), ("teacher_sft", s)):
        for seed, v in enumerate(vals):
            rows.append(_row(f"{m}-s{seed}", "q0", m, 0.5, -0.05, lr=1e-6, seed=seed, step=1))
            rows.append(_row(f"{m}-s{seed}", "q0", m, 0.55, v, lr=1e-6, seed=seed, step=2))  # final
    tab, out = analyze_mod.headline_table(rows, base)
    res = ttest_ind(s, g, equal_var=False)
    assert out["teacher_sft"]["p"] == pytest.approx(res.pvalue) and out["teacher_sft"]["t"] == pytest.approx(
        res.statistic)
    assert out["teacher_sft"]["diff"] == pytest.approx(np.mean(s) - np.mean(g))
    cells = {r[0]: r for r in tab.rows}
    assert cells["grpo lr=1e-06"][4] == f"{100 * np.mean(g):.2f} ± {100 * np.std(g, ddof=1):.2f}"


def test_held_notes_name_the_held_point(analyze_mod):
    rows = [_row("g-a", "q0", "grpo", 0.524, 0.0056, step=3), _row("g-a", "q0", "grpo", 0.50, -0.001, step=1)]
    curves = {"grpo": {"points": [(0.524, 0.0056), (0.50, -0.001)]}}
    notes = analyze_mod.held_notes(curves, [0.511], rows)
    assert len(notes) == 1 and "g-a step 3" in "".join(notes[0]) and "52.4" in "".join(notes[0])
    assert analyze_mod.held_notes(curves, [0.53], rows) == []


def test_train_log_summary_and_launch_log(analyze_mod, tmp_path):
    lines = ["noise", *[str({"loss": "0.1", "entropy": str(0.3 - 0.01 * i), "reward": str(0.4 + 0.01 * i),
                             "completions/mean_length": "300"}) for i in range(25)], "{'train_runtime': '5'}"]
    s = analyze_mod.train_log_summary("\n".join(lines), n=10)
    assert s["steps"] == 25 and s["first"]["entropy"] == pytest.approx(0.3 - 0.045)
    assert s["last"]["reward"] == pytest.approx(0.4 + 0.01 * 19.5) and s["last"]["clipped"] is None
    assert analyze_mod.train_log_summary("nothing here") is None
    log = ("started x (projected $1.00, committed $2.00)\n"
           "finished grpo-lr1e-06-s0-abc: exit 0, charged $2.40, committed $30.56\n"
           "finished rft_offline-lr1e-05-s0-def: exit 1, charged $1.08, committed $19.01\n"
           "summary (committed $30.22 of budget $41.50)\n")
    got = analyze_mod.parse_launch_log(log, "l.log")
    assert got == [{"run_id": "grpo-lr1e-06-s0-abc", "exit": 0, "charged": 2.40, "log": "l.log"},
                   {"run_id": "rft_offline-lr1e-05-s0-def", "exit": 1, "charged": 1.08, "log": "l.log"}]
    (tmp_path / "logs").mkdir()
    (tmp_path / "launch-a.log").write_text(log)
    (tmp_path / "logs" / "grpo-lr1e-06-s0-abc.log").write_text("\n".join(lines))
    res = tmp_path / "res"
    res.mkdir()
    analyze_mod.logs_to_results(tmp_path, res)
    assert json.loads((res / "launcher_committed.json").read_text()) == {"committed": 30.22}
    assert len((res / "launcher_charges.jsonl").read_text().splitlines()) == 2
    assert json.loads((res / "train_log_summary.jsonl").read_text())["run_id"] == "grpo-lr1e-06-s0-abc"


def test_billing_tables_sum_phases_and_skip_failed_launches(analyze_mod):
    billing = {"rows": [{"interval_start": "2026-11-22T03:00:00", "description": "rlrazor", "cost": "1.25"},
                        {"interval_start": "2026-11-22T07:00:00", "description": "rlrazor", "cost": "2.50"},
                        {"interval_start": "2026-11-22T07:00:00", "description": "rlrazor-diag", "cost": "0.25"},
                        {"interval_start": "2026-11-22T11:00:00", "description": "rlrazor", "cost": "4.00"},
                        {"interval_start": "2026-11-22T16:00:00", "description": "rlrazor", "cost": "3.00"}]}
    charges = [{"run_id": "grpo-lr1e-06-s0-a", "exit": 0, "charged": 2.0},
               {"run_id": "grpo-lr1e-06-s1-b", "exit": 0, "charged": 3.0},
               {"run_id": "grpo-lr1e-05-s0-c", "exit": 1, "charged": 9.0}]
    tables, out = analyze_mod.billing_tables(billing, charges, 8.5, stage2_committed=12.0)
    assert out["total"] == pytest.approx(11.0) and out["by_method"] == {"grpo": 5.0}
    assert out["stage1"] == pytest.approx(8.0)
    phase = {r[0]: r[2] for r in tables[0].rows}
    # three stage-1 phases, the stage-1 subtotal, stage 2, the total
    assert list(phase.values()) == ["1.25", "2.75", "4.00", "8.00", "3.00", "11.00"]
    assert tables[2].rows[0][1:] == ["2", "5.00", "2.50"]
    guard = {r[0]: r[1] for r in tables[3].rows}
    assert guard["Guard ceiling"] == "41.50" and "12.00" in guard.values() and "96.00" in guard.values()


def test_committed_results_build_kl_and_cost_sections(tmp_path):
    """End to end on the committed study data: the Q0-only headline fit, the KL detail and the cost tables exist,
    and every number they print is recorded for check-writeup."""
    res = ROOT / "results" / "results.jsonl"
    proc = _analyze("--results", res, "--out", tmp_path)
    assert proc.returncode == 0, proc.stderr
    md = (tmp_path / "tables.md").read_text()
    for s in ("Q0: teacher SFT + GRPO + base, sequence KL (headline)", "Per-checkpoint forward KL",
              "Top-5 base samples", "Actual spend from Modal billing", "Not run: projected cost",
              "Headline: final checkpoints", "lies below grpo's lowest frontier point"):
        assert s in md, s
    nums = json.loads((tmp_path / "numbers.json").read_text())
    assert "40.74" in nums["results"] and "0.30" in nums["results"]


def test_frontier_zoom_limits_keep_near_frontier_and_count_the_rest():
    """Figure 1 zooms to the near-frontier points and the base, counting points outside."""
    from rlrazor.analysis.figures import zoom_limits

    curves = {"teacher_sft": {"near": [(0.50, -0.001), (0.515, -0.002)],
                              "points": [(0.50, -0.001), (0.515, -0.002), (0.273, -0.122), (0.384, -0.049)]},
              "grpo": {"near": [(0.524, 0.0056), (0.565, 0.0007)], "points": [(0.524, 0.0056), (0.565, 0.0007)]}}
    xlim, ylim, outside = zoom_limits(curves, (0.481, 0.0))
    assert xlim[0] < 0.481 < 0.565 < xlim[1] and ylim[0] < -0.002 and 0.0056 < ylim[1] < 0.02
    assert ylim[0] > -0.02 and outside == 2  # the two high-LR SFT points fall outside
    assert zoom_limits({}, None) == (None, None, 0)


def test_is_stage1_separates_stage2_rows(analyze_mod):
    """Stage-2 rows (Q2-Q4, iterative RFT, self-rewrite seeds 1-2) are kept out of the stage-1 snapshot
    tables that the stage-1 write-up quotes; train_times entries carry their config under "config"."""
    s1 = analyze_mod.is_stage1
    assert s1({"arm": "q0", "method": "grpo", "seed": 2}) and s1({"arm": "q1", "method": "self_rewrite", "seed": 0})
    assert s1({"arm": "q1", "method": "rft_offline", "seed": 1}) and s1({"arm": "pilot", "method": "sft"})
    assert not s1({"arm": "q1", "method": "self_rewrite", "seed": 1})
    assert not s1({"arm": "q1", "method": "rft_iter", "seed": 0})
    assert not any(s1({"arm": a, "method": "teacher_sft", "seed": 0}) for a in ("q2", "q3", "q4"))
    assert not s1({"run_id": "x", "config": {"arm": "q2", "method": "grpo"}, "wall_seconds": 1.0})


def test_writeup_md_mirrors_tex_structure_and_numbers():
    """WRITEUP.md mirrors writeup.tex (setup + answers only, no appendix), both pass check-writeup against
    the committed tables, and every decimal in WRITEUP.md appears in the tex."""
    import re

    tex = (ROOT / "writeup" / "writeup.tex").read_text()
    md = (ROOT / "WRITEUP.md").read_text()
    for gone in ("\\appendix", "tables.tex}", "Implications for Belvedir", "Cost and run time", "section*{Reproduction}"):
        assert gone not in tex, gone
    # the paper's idea first, the Belvedir proposal last, in this order in both files
    titles = ["The paper's main idea", "Setup", "Q0 baseline", "Q1:", "Q2:", "Q3:", "Why two of the three questions",
              "Limitations", "Proposal for Belvedir"]
    tex_heads = re.findall(r"\\section\{([^}]*)\}", tex)
    md_heads = [h for h in re.findall(r"^## (?:\d+\. )?(.+)$", md, flags=re.M) if h != "The task"]
    for heads in (tex_heads, md_heads):
        assert len(heads) == len(titles), heads
        for h, t in zip(heads, titles):
            assert h.startswith(t), (h, t)
    # proposal honesty rules: basis tags, KL advisory, post-hoc canary motive, untested repair ladder
    for text in (tex, md):
        prop = " ".join(text.split("Proposal for Belvedir", 1)[1].split())
        for must in ("[P]", "[D]", "[A]", "advisory", "post hoc", "untested by our data", "off-policy algorithms",
                     "No default general-data mix fraction"):
            assert must in prop, must
        assert "their site" not in prop and "their postings" not in prop
        # the proposal stands on the paper, our data, the public Belvedir source and labelled assumptions; it cites
        # no proposal document (that document is not exported)
        for gone in ("\\Pd", "P\u00a7", "proposal document", "workflow" "-proposal"):
            assert gone not in text, gone
        assert not re.search(r"Steps?[~ ]\d", prop), "proposal step labels"
        assert "assumed standard practice" in prop and "discrete cycles **[A]**" in prop.replace("\\textbf{[A]}", "**[A]**")
    assert "\\date{}" in tex  # no date in the title block
    assert "\\T{" in tex and "\\cite[" in tex  # reference keys and paper section/page citations
    # contents page first, no answer box, no extras/mitigations section, no Q4 label, no staleness
    assert "\\tableofcontents" in tex and "\\newpage" in tex and "\\fbox" not in tex
    assert "## Answers" not in md
    questions = ["Does SFT on the model's own correct answers forget like SFT or like RL?",
                 "Which method holds up better when you add noise to the labels or rewards?",
                 "How much general data do you need to mix into SFT before it forgets as little as RL?"]
    for text in (tex, md):
        flat = " ".join(text.split())
        for q in questions:
            assert q in flat, q
        for gone in (r"\bQ4\b", r"\bWiSE", r"\bLoRA\b", r"\bGKD\b"):  # case-sensitive, whole words
            assert not re.search(gone, text), gone
        for gone in ("staleness", "mitigation"):
            assert gone not in text.lower(), gone
        for status in ("partly reproduced", "answered on the mean: like rl", "half answered", "not answered in this setup"):
            assert status in text.lower(), status
    assert set(re.findall(r"\d+\.\d+", md)) <= set(re.findall(r"\d+\.\d+", tex))
    readme_summary = (ROOT / "README.md").read_text().split("**Results:**", 1)[1].split("\n## ", 1)[0]
    for gone in (r"\bQ4\b", r"\bWiSE", r"\bLoRA\b", r"\bGKD\b", r"[Mm]itigation"):
        assert not re.search(gone, readme_summary), gone


def test_kl_figure_omits_q4_runs(tmp_path, analyze_mod, monkeypatch):
    """With mixed arms, the KL-vs-forgetting figure draws no q4 run (no distill / LoRA method)."""
    drawn = []
    monkeypatch.setattr(analyze_mod.F, "kl_figure", lambda out, name, pts, *a, **k: drawn.append(dict(pts)))
    analyze_mod.analyze(_synthetic(), tmp_path)
    assert drawn and "distill" not in drawn[-1] and "lora_sft" not in drawn[-1]
    assert "grpo" in drawn[-1]
    for path in (ROOT / "writeup" / "writeup.tex", ROOT / "WRITEUP.md"):
        assert _analyze("--check-writeup", path, "--out", ROOT / "figures").returncode == 0, path
