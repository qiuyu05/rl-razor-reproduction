"""Statistics for the comparison protocol (spec §3.2).

Points are (gsm8k_acc, forgetting_mean) pairs in fractions (0.45 = 45 %). Higher forgetting_mean means
less forgetting (it is checkpoint − base, negative = forgetting), so the Pareto frontier maximises both.
Nothing here is random: every fit is deterministic given its inputs.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

KNOWN_ARMS = ("base", "q0", "q1", "q2", "q3", "q4")
TARGET_STEP = 0.03  # matched-accuracy targets: base+3, base+6, ... accuracy points
MIN_EXP_POINTS = 4  # fewer frontier points than this -> piecewise-linear interpolation


# ---------- frontier and fits ----------


def pareto_frontier(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    """Points not dominated in (accuracy, forgetting_mean), both maximised; sorted by accuracy."""
    pts = sorted({(float(a), float(f)) for a, f in points}, key=lambda p: (-p[0], -p[1]))
    front, best_f = [], -np.inf
    for a, f in pts:  # by accuracy descending: keep a point only if it forgets less than all more-accurate ones
        if f > best_f:
            front.append((a, f))
            best_f = f
    return sorted(front)


def near_frontier(points, tol: float = 0.02) -> list[tuple[float, float]]:
    """Points within `tol` accuracy of the frontier: the most accurate frontier point that forgets
    no more than the point is at most `tol` more accurate (paper Appendix B.1)."""
    pts = [(float(a), float(f)) for a, f in points]
    front = pareto_frontier(pts)
    out = []
    for a, f in pts:
        reach = max(fa for fa, ff in front if ff >= f)  # exists: the point itself is dominated or on it
        if reach - a <= tol + 1e-12:
            out.append((a, f))
    return sorted(set(out))


class Fit:
    """A fitted frontier curve: callable acc -> forgetting, with its kind and RMSE on the fitted points."""

    def __init__(self, f: Callable[[float], float], kind: str, xs, ys):
        self._f, self.kind = f, kind
        self.rmse = float(np.sqrt(np.mean([(f(x) - y) ** 2 for x, y in zip(xs, ys)])))

    def __call__(self, x: float) -> float:
        return float(self._f(x))


def _interp(xs, ys) -> Callable[[float], float]:
    if len(xs) == 1:
        y0 = float(ys[0])
        return lambda x: y0
    return lambda x: float(np.interp(x, xs, ys))


def fit_exponential(points) -> Fit:
    """Fit forgetting = c + a·exp(b·(acc − acc_min)) to (near-)frontier points; b may take either sign,
    so the curve can saturate (b < 0) or accelerate (b > 0).

    b is found by a grid search over ±[0.01, 100] (log-spaced) refined with a bounded scalar minimiser;
    for each b, (a, c) are the least-squares solution. If the least-squares line fits at least as well
    (the b→0 limit), the line is used. With fewer than MIN_EXP_POINTS distinct accuracies the curve is
    the piecewise-linear interpolation of the points. `Fit.kind` says which; `Fit.rmse` is the error on
    the fitted points.
    """
    pts = sorted((float(a), float(f)) for a, f in points)
    if not pts:
        raise ValueError("fit_exponential needs at least one point")
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    if len(np.unique(xs)) < MIN_EXP_POINTS:
        ux = np.unique(xs)
        uy = np.array([ys[xs == x].max() for x in ux])
        return Fit(_interp(ux, uy), "interpolation", xs, ys)
    x0 = xs.min()
    m, c = np.polyfit(xs, ys, 1)
    line_sse = float(np.sum((m * xs + c - ys) ** 2))

    def solve(b):  # for fixed b the model is linear in (a, c): profile them out by least squares
        design = np.column_stack([np.exp(b * (xs - x0)), np.ones_like(xs)])
        coef, *_ = np.linalg.lstsq(design, ys, rcond=None)
        return coef, float(np.sum((design @ coef - ys) ** 2))

    pos = np.logspace(-2, 2, 81)
    grid = np.concatenate([-pos[::-1], pos])  # b in ±[0.01, 100]
    sse = [solve(b)[1] for b in grid]
    i = int(np.argmin(sse))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = minimize_scalar(lambda b: solve(b)[1], bounds=(lo, hi), method="bounded")
    b = float(res.x) if res.fun <= sse[i] else float(grid[i])
    (a, cc), exp_sse = solve(b)
    if exp_sse < line_sse * (1 - 1e-6):
        return Fit(lambda x: cc + a * np.exp(b * (x - x0)), "exponential", xs, ys)
    return Fit(lambda x: m * x + c, "line", xs, ys)


def _in_range(pts, t) -> bool:
    return min(a for a, _ in pts) - 1e-9 <= t <= max(a for a, _ in pts) + 1e-9


def forgetting_at_accuracy(points, targets: list[float], tol: float = 0.02) -> list[float | None]:
    """Forgetting at each target accuracy, read off the fit to the near-frontier points.
    None for targets outside the observed accuracy range of `points`."""
    pts = [(float(a), float(f)) for a, f in points]
    if not pts:
        return [None for _ in targets]
    fit = fit_exponential(near_frontier(pts, tol))
    return [fit(t) if _in_range(pts, t) else None for t in targets]


def frontier_interp(points, targets: list[float]) -> list[float | None]:
    """Forgetting at each target by piecewise-linear interpolation along the observed Pareto-frontier
    points (no model). Below the frontier's lowest accuracy it holds that point's value; None outside
    the observed accuracy range of `points`."""
    pts = [(float(a), float(f)) for a, f in points]
    if not pts:
        return [None for _ in targets]
    front = pareto_frontier(pts)
    f = _interp([a for a, _ in front], [y for _, y in front])
    return [f(t) if _in_range(pts, t) else None for t in targets]


def quadratic_r2(x, y) -> tuple[np.ndarray, float]:
    """Least-squares y = c0·x² + c1·x + c2. Returns (coefficients highest power first, R²)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    coef = np.polyfit(x, y, 2)
    ss_res = float(np.sum((np.polyval(coef, x) - y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return coef, (1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0


def matched_targets(base_acc: float, reach: float) -> list[float]:
    """base+3, base+6, ... accuracy points, up to `reach` (the largest accuracy both methods reach)."""
    out, k = [], 1
    while base_acc + k * TARGET_STEP <= reach + 1e-9:
        out.append(round(base_acc + k * TARGET_STEP, 6))
        k += 1
    return out


def mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and sample std (ddof=1); std is 0.0 for a single value."""
    v = np.asarray(values, float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


# ---------- rows ----------


def is_base(row: dict) -> bool:
    return row.get("run_id") == "base" or row.get("arm") == "base"


def filter_rows(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Drop rows from unknown arms, and smoke rows when real rows exist. Returns (rows, notes)."""
    notes = []
    unknown = sorted({str(r.get("arm")) for r in rows if r.get("arm") not in KNOWN_ARMS and not is_base(r)})
    if unknown:
        notes.append(f"ignoring rows from unknown arm(s): {', '.join(unknown)}")
    rows = [r for r in rows if r.get("arm") in KNOWN_ARMS or is_base(r)]
    if any(r.get("smoke") for r in rows) and not all(r.get("smoke") for r in rows):
        notes.append(f"dropping {sum(bool(r.get('smoke')) for r in rows)} smoke row(s)")
        rows = [r for r in rows if not r.get("smoke")]
    return rows, notes


DIVERGED_ACC = 0.01  # a run whose final checkpoint scores below 1% GSM8K has collapsed (degenerate outputs)


def diverged_run_ids(rows: list[dict]) -> set[str]:
    """Runs that diverged (R47/R50): any row flagged `diverged` (GRPO dead-run stop), or, for rows written before
    the flag existed, a final checkpoint below DIVERGED_ACC GSM8K accuracy. Excluded from frontiers (in the spirit
    of the paper's App. B.1 step 4 trade-off-curve filter) and from the KL fit (our choice: the paper does not say
    its KL fit was filtered); reported in an appendix with the pooled R² including them."""
    out = set()
    for rid, rs in by_run([r for r in rows if not is_base(r)]).items():
        if any(r.get("diverged") for r in rs) or float(rs[-1]["gsm8k_acc"]) < DIVERGED_ACC:
            out.add(rid)
    return out


def is_clean(row: dict) -> bool:
    """A plain training run: no noise, no mixing, no LoRA, not a WiSE-FT interpolation (R30, Q0 selection)."""
    return not (row.get("noise_p") or row.get("mix_f") or row.get("lora") or row.get("wise_alpha"))


def source_run_id(init_from: str | None) -> str | None:
    """`init_from` is "<source run_id>/step-<n>" or "base" (R30): the source run_id, exactly."""
    return None if init_from is None else str(init_from).split("/")[0]


def find_base(rows: list[dict]) -> dict | None:
    return next((r for r in rows if is_base(r)), None)


def point(row: dict) -> tuple[float, float]:
    return (float(row["gsm8k_acc"]), float(row["forgetting_mean"]))


def by_run(rows: list[dict]) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for r in rows:
        out[r["run_id"]].append(r)
    return {k: sorted(v, key=lambda r: r["step"]) for k, v in sorted(out.items())}


def final_rows(rows: list[dict]) -> list[dict]:
    """The last checkpoint of each run."""
    return [v[-1] for v in by_run(rows).values()]


def setting(row: dict) -> tuple:
    """Hyperparameter setting of a run, excluding the seed."""
    return (row["method"], row["lr"], row.get("noise_p", 0.0), row.get("mix_f", 0.0), row.get("lora", 0),
            row.get("wise_alpha", 0.0), row.get("init_from"))


def best_setting_rows(rows: list[dict], method: str) -> list[dict]:
    """Rows of `method`'s best setting: the one rerun with the most seeds (spec §4: the best LR gets
    2 extra seeds); ties go to the setting whose runs reach the highest GSM8K accuracy."""
    groups = defaultdict(list)
    for r in rows:
        if r["method"] == method and not is_base(r):
            groups[setting(r)].append(r)
    if not groups:
        return []

    def score(item):
        _, rs = item
        return (len({r["seed"] for r in rs}), max(r["gsm8k_acc"] for r in rs))

    return max(sorted(groups.items(), key=lambda kv: repr(kv[0])), key=score)[1]


@dataclass(frozen=True)
class SeedBand:
    """Forgetting at `target` over the best setting's seeds. `k` of `n` seeds reach the target.
    `std` (ddof=1) is None unless every seed contributes and there are at least two."""

    mean: float
    std: float | None
    k: int
    n: int
    target: float

    @property
    def partial(self) -> bool:
        return self.k < self.n


def seed_band(rows: list[dict], method: str, targets=None,
              base_point: tuple[float, float] | None = None) -> SeedBand | None:
    """Seed band of forgetting at matched accuracy for `method`'s best setting.

    Each seed's curve is its own run's checkpoints plus `base_point`. The band is taken at the highest
    of `targets` (a float or a list) that every seed reaches, so no seed is silently dropped. If no
    target is reached by all seeds the band is partial: it is taken at the target reached by the most
    seeds (highest such target), with `std=None`. `targets` defaults to the highest accuracy every seed
    reaches. None if no seed reaches any target.
    """
    runs = list(by_run(best_setting_rows(rows, method)).values())
    if not runs:
        return None
    if targets is None:
        targets = [min(max(r["gsm8k_acc"] for r in rs) for rs in runs)]
    elif isinstance(targets, (int, float)):
        targets = [float(targets)]
    curves = [[point(r) for r in rs] + ([base_point] if base_point else []) for rs in runs]
    best = None
    for t in sorted(targets):
        vals = [v for v in (forgetting_at_accuracy(c, [t])[0] for c in curves) if v is not None]
        if vals and (best is None or len(vals) >= len(best[1])):
            best = (t, vals)
    if best is None:
        return None
    t, vals = best
    k, n = len(vals), len(runs)
    std = float(np.std(vals, ddof=1)) if k == n and k > 1 else None
    return SeedBand(float(np.mean(vals)), std, k, n, t)


def mixing_crossover(mix_rows: Mapping[float, float | None] | Iterable[tuple[float, float | None]],
                     grpo_band: tuple[float, float]) -> float | None:
    """Smallest mix fraction f whose forgetting is ≥ GRPO mean − std; None if no f qualifies.
    `mix_rows` maps f -> forgetting at matched accuracy (None = f did not reach the target)."""
    items = mix_rows.items() if isinstance(mix_rows, Mapping) else mix_rows
    mean, std = grpo_band
    ok = [f for f, forg in items if forg is not None and forg >= mean - std - 1e-12]
    return min(ok) if ok else None
