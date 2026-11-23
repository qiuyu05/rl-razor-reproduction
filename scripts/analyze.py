"""Turn results rows into the write-up's figures and tables, and guard the write-up's numbers.

    uv run python scripts/analyze.py --results results.jsonl --out figures/
    uv run python scripts/analyze.py --check-writeup WRITEUP.tex --out figures/ [--results results.jsonl]

The first form writes six figures (PNG at 200 dpi + PDF) into --out, plus `tables.md` and `tables.tex`
(same cells, same numbers) and `numbers.json`, which records every number in the tables split into
RESULTS (measured values) and CONSTANTS (hyperparameters, setting labels, reference constants).

The second form extracts every decimal number (regex -?\\d+\\.\\d+) from any text file and checks it
against --out/numbers.json:
  * a number is backed if it is a result, or is the unsigned magnitude of a NEGATIVE result
    (prose "forgets 4.5 pp" for a table entry -4.5). A sign is never added: -x is not backed by +x;
  * a number that matches only a constant is accepted but listed as "matched constant" for a human;
  * anything else is listed and the exit code is 1. Tables built from smoke data exit 3; missing
    tables exit 2. With --results the tables are regenerated from it first, so they cannot be stale.

Number formats (fixed, so the write-up can quote tables verbatim):
  accuracy and benchmark scores: percent, 1 decimal       e.g. 45.2
  forgetting / benchmark deltas: percentage points, 1 dec. e.g. -1.3 (checkpoint − base)
  forward KL: nats, 2 decimals; R² and fit RMSE: 2 / 1 decimals; hyperparameters (p, f, α, lr) as `:g`.
Missing arms never crash the script: their sections say "No data yet" and their figures are placeholders.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rlrazor.analysis import figures as F
from rlrazor.analysis.stats import (
    SeedBand, best_setting_rows, by_run, diverged_run_ids, filter_rows, final_rows, find_base, fit_exponential,
    forgetting_at_accuracy, frontier_interp, is_base, is_clean, matched_targets, mean_std, mixing_crossover,
    near_frontier, point, quadratic_r2, seed_band, setting, source_run_id,
)

NUMBER = re.compile(r"-?\d+\.\d+")
BENCH = ("hellaswag", "mmlu", "truthfulqa", "winogrande", "ifeval", "humaneval")
NA = "n/a"
NO_DATA = "No data yet (not run)."
SMOKE_BANNER = "SMOKE DATA: not results."
PAPER = {"arxiv": "2509.04259", "model": "Qwen2.5-0.5B-Instruct", "kl_r2": 0.71, "kl_r2_toy": 0.96}


def sft_ref(methods) -> str:
    """The study's SFT reference method: SFT on teacher answers (R42/R43) when present, else SFT (gold)."""
    return "teacher_sft" if "teacher_sft" in set(methods) else "sft"


# ---------- number formats ----------


class Const(str):
    """A table cell or text segment whose numbers are constants (hyperparameters, labels, references),
    not measured results. check-writeup reports write-up numbers that match only these."""


def _fixed(x, nd):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return NA
    s = f"{x:.{nd}f}"
    return s[1:] if re.fullmatch(r"-0\.0+", s) else s  # never print -0.0


def pct(x):  # accuracy / scores (fraction -> percent) and forgetting (fraction -> pp)
    return _fixed(None if x is None else 100 * x, 1)


def kl2(x):
    return _fixed(x, 2)


def par(x):
    return Const(f"{x:g}")


# ---------- tables ----------


@dataclass
class Table:
    title: str
    header: list[str]
    rows: list[list[str]]
    tex: bool = True  # False: in tables.md and numbers.json only (repeated in the write-up's body, or page-long)


@dataclass
class Section:
    title: str
    blocks: list = field(default_factory=list)  # Table, str, or tuple of str/Const segments (a paragraph)


def _segments(block) -> tuple:
    return block if isinstance(block, tuple) else (block,)


_TEX = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$", "±": r"$\pm$", "²": r"$^2$",
        "α": r"$\alpha$", "−": "--", "—": "---", "≥": r"$\geq$", "→": r"$\to$",
        "θ": r"$\theta$", "₀": r"$_0$", "·": r"$\cdot$", "≤": r"$\leq$", "∈": r"$\in$", "×": r"$\times$",
        "ρ": r"$\rho$", "π": r"$\pi$", "–": "--", "^": r"\^{}", "§": r"\S{}"}


def _tex(s: str) -> str:
    return "".join(_TEX.get(ch, ch) for ch in s)


def render_md(sections, preamble) -> str:
    out = ["# Results tables", "", *preamble, ""]
    for sec in sections:
        out += [f"## {sec.title}", ""]
        for b in sec.blocks:
            if isinstance(b, Table):
                out += [f"**{b.title}**", "", "| " + " | ".join(b.header) + " |",
                        "|" + "|".join("---" for _ in b.header) + "|"]
                out += ["| " + " | ".join(r) + " |" for r in b.rows]
            else:
                out.append("".join(_segments(b)))
            out.append("")
    return "\n".join(out)


MD_ONLY = ("Headline: final checkpoints", "Per-checkpoint forward KL", "Final checkpoint vs noise rate",
           "Quadratic fit of mean forgetting on forward KL, by subset", "Forgetting at ")  # see analyze()
TEX_MAX_ROWS = 55  # rows per table float in tables.tex; longer tables are split so none runs off the page
LANDSCAPE_COLS = 9  # landscape only for tables with more columns than this; narrower ones shrink to the text width
LANDSCAPE_CHARS = 134  # estimated row width (characters) above which a table goes on a landscape page


def _tex_width(t: "Table") -> int:
    """Rough rendered width of a table in characters: widest cell (or header) per column plus padding."""
    cols = list(zip(t.header, *t.rows)) if t.rows else [(h,) for h in t.header]
    return sum(max(len(str(c)) for c in col) + 3 for col in cols)


def render_tex(sections, preamble) -> str:
    out = ["% Generated by scripts/analyze.py -- do not edit. Requires \\usepackage{booktabs,adjustbox,float,pdflscape}.", ""]
    out += [_tex(p) + "\n" for p in preamble]
    for sec in sections:
        blocks = []
        for b in sec.blocks:  # a table longer than a page is split into page-sized parts (same cells)
            if isinstance(b, Table) and not b.tex:
                continue
            if isinstance(b, Table) and len(b.rows) > TEX_MAX_ROWS:
                parts = range(0, len(b.rows), TEX_MAX_ROWS)
                blocks += [dataclasses.replace(b, title=b.title + (f" (continued, part {k + 1})" if k else ""),
                                               rows=b.rows[i:i + TEX_MAX_ROWS]) for k, i in enumerate(parts)]
            else:
                blocks.append(b)
        # every table shrinks to the text width only when it is wider (adjustbox "max width"); very wide tables get
        # a landscape page so they stay readable (needs pdflscape). Consecutive landscape tables share one landscape
        # environment, and a heading followed by a landscape table goes inside it (no stranded headings).
        land = [isinstance(b, Table) and len(b.header) > LANDSCAPE_COLS and _tex_width(b) > LANDSCAPE_CHARS
                and len(b.rows) <= 25 for b in blocks]
        heading = [f"\\subsection*{{{_tex(sec.title)}}}", ""]
        in_land = False
        if not (land and land[0]):
            out += heading
        for k, b in enumerate(blocks):
            if land[k] and not in_land:
                out.append("\\begin{landscape}")
                in_land = True
                if k == 0:
                    out += heading
            elif not land[k] and in_land:
                out += ["\\end{landscape}", ""]
                in_land = False
            if isinstance(b, Table):
                spec = "l" + "r" * (len(b.header) - 1)
                out += ["\\begin{table}[H]", "\\centering", "\\small", f"\\caption{{{_tex(b.title)}}}",
                        "\\begin{adjustbox}{max width=\\linewidth}",
                        f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
                        " & ".join(_tex(h) for h in b.header) + " \\\\", "\\midrule"]
                out += [" & ".join(_tex(c) for c in r) + " \\\\" for r in b.rows]
                out += ["\\bottomrule", "\\end{tabular}", "\\end{adjustbox}", "\\end{table}"]
            else:
                out.append(_tex("".join(_segments(b))))
            out.append("")
        if in_land:
            out += ["\\end{landscape}", ""]
    return "\n".join(out)


def collect_numbers(sections, preamble) -> tuple[set[str], set[str]]:
    """(results, constants): numbers in Const cells/segments, headers and the preamble are constants;
    every other cell, title and paragraph segment holds results."""
    results, consts = set(), set()

    def add(s, const):
        (consts if const or isinstance(s, Const) else results).update(NUMBER.findall(s))

    for p in preamble:
        add(p, True)
    for sec in sections:
        add(sec.title, True)
        for b in sec.blocks:
            if isinstance(b, Table):
                add(b.title, False)
                for h in b.header:
                    add(h, True)
                for r in b.rows:
                    for c in r:
                        add(c, False)
            else:
                for s in _segments(b):
                    add(s, False)
    return results, consts


# ---------- helpers ----------


def label(row) -> Const:
    m, lr, p, f, lora, alpha, src = setting(row)
    parts = [m, f"lr={par(lr)}"]
    if p:
        parts.append(f"p={par(p)}")
    if f:
        parts.append(f"f={par(f)}")
    if lora:
        parts.append(f"r={lora}")
    if alpha:
        parts += [f"α={par(alpha)}", f"from {source_run_id(src)}"]
    return Const(" ".join(parts))


def _methods(rows):
    present = {r["method"] for r in rows}
    return [m for m in F.METHOD_STYLE if m in present] + sorted(present - set(F.METHOD_STYLE))


def _curve(rows, base_pt):
    pts = [point(r) for r in rows] + ([base_pt] if base_pt else [])
    near = near_frontier(pts)
    return {"points": pts, "near": near, "fit": fit_exponential(near),
            "lo": min(a for a, _ in pts), "hi": max(a for a, _ in pts)}


def _per_run_at(rows, target, base_pt) -> tuple[list[float], int]:
    """Forgetting at `target` along each run's own checkpoints (+ base): (values of runs reaching it, n runs)."""
    runs = by_run(rows)
    vals = []
    for rs in runs.values():
        pts = [point(r) for r in rs] + ([base_pt] if base_pt else [])
        v = forgetting_at_accuracy(pts, [target])[0] if target is not None else None
        if v is not None:
            vals.append(v)
    return vals, len(runs)


def _status(k, n):
    return "complete" if k == n else f"partial ({k}/{n} seeds)"


BAND_HEADER = ["Method", "Best setting", "Seeds used", "Target acc. (%)", "Forgetting mean (pp)", "Std (pp)",
               "Status"]


def _band_row(method, rows, band: SeedBand | None):
    best = best_setting_rows(rows, method)
    n = len(by_run(best))
    if band is None:
        return [method, label(best[0]) if best else NA, f"0/{n}", NA, NA, NA, "no seed reaches a target"]
    return [method, label(best[0]), f"{band.k}/{band.n}", pct(band.target), pct(band.mean), pct(band.std),
            _status(band.k, band.n)]


def _fit_table(curves) -> Table:
    return Table("Frontier fits (near-frontier points; exponential with either sign of b, or a line)",
                 ["Method", "Fit", "Points fitted", "Fit RMSE (pp)"],
                 [[m, c["fit"].kind, str(len(c["near"])), pct(c["fit"].rmse)] for m, c in curves.items()])


def bench_table(title, rows, base) -> Table:
    groups = defaultdict(list)
    for r in final_rows(rows):
        groups[label(r)].append(r)
    header = ["Setting (final ckpt)", "Seeds", "GSM8K (%)", *BENCH, "Mean forgetting (pp)", "KL"]
    body = []
    for lab, rs in sorted(groups.items()):
        cells = [Const(lab), str(len(rs)), pct(np.mean([r["gsm8k_acc"] for r in rs]))]
        for b in BENCH:
            ok = base is not None and all(b in r.get("bench", {}) for r in rs) and b in base.get("bench", {})
            cells.append(pct(np.mean([r["bench"][b] - base["bench"][b] for r in rs])) if ok else NA)
        cells.append(pct(np.mean([r["forgetting_mean"] for r in rs])))
        kls = [r["kl_fwd"] for r in rs if r.get("kl_fwd") is not None]
        cells.append(kl2(np.mean(kls)) if kls else NA)
        body.append(cells)
    return Table(f"{title}: per-benchmark change vs base (pp), mean over seeds", header, body)


def best_bench(rows, base) -> tuple[Table | None, dict]:
    """Per-benchmark change vs base at each method's best setting (the setting with the most seeds), final
    checkpoints, mean ± sd over seeds, plus an uncorrected Welch p vs GRPO (post hoc)."""
    from scipy.stats import ttest_ind

    if base is None:
        return None, {}
    keys = ["gsm8k", *BENCH, "mean"]

    def val(r, k):
        if k == "gsm8k":
            return r["gsm8k_acc"] - base["gsm8k_acc"]
        if k == "mean":
            return r["forgetting_mean"]
        return r["bench"][k] - base["bench"][k]

    methods = sorted(_methods(rows), key=lambda m: (m != "grpo", m))
    finals, out = {}, {"p_vs_grpo": {}}
    for m in methods:
        fs = final_rows(best_setting_rows(rows, m))
        if fs and all(b in f.get("bench", {}) for f in fs for b in BENCH):
            finals[m] = fs
    body = []
    for m, fs in finals.items():
        out[m] = {}
        cells = [Const(label(fs[0])), str(len(fs))]
        for k in keys:
            v = [val(f, k) for f in fs]
            sd = float(np.std(v, ddof=1)) if len(v) > 1 else None
            out[m][k] = (float(np.mean(v)), sd, len(v))
            cells.append(pct(np.mean(v)) + (f" ± {pct(sd)}" if sd is not None else ""))
        body.append(cells)
        if m != "grpo" and "grpo" in finals and len(fs) > 1 and len(finals["grpo"]) > 1:
            out["p_vs_grpo"][m] = {k: float(ttest_ind([val(f, k) for f in fs], [val(f, k) for f in finals["grpo"]],
                                                       equal_var=False).pvalue) for k in keys}
    if not body:
        return None, out
    header = ["Best setting (final ckpt)", "Seeds", "GSM8K (pp)", *BENCH, "Mean (pp)"]
    tables = [Table("Per-benchmark change at each method's best setting (final checkpoint, change vs base in pp, "
                    "mean ± sd over seeds)", header, body)]
    if out["p_vs_grpo"]:
        tables.append(Table("Welch t-test p vs GRPO at the best settings (post hoc, uncorrected; n = seeds)",
                            ["Method", "GSM8K", *BENCH, "Mean"],
                            [[Const(m), *[_fixed(ps[k], 3) for k in keys]] for m, ps in out["p_vs_grpo"].items()]
                            + [[Const(f"{m} (Bonferroni, × 6 benchmarks)"), NA,
                                *[_fixed(min(1.0, 6 * ps[k]), 3) for k in BENCH], NA]
                               for m, ps in out["p_vs_grpo"].items()]))
    return tables, out


# ---------- headline, KL detail, data, cost (R46/R50/R51) ----------


def pp2(x):  # forgetting / changes in pp with 2 decimals (seeded headline only)
    return _fixed(None if x is None else 100 * x, 2)


def _f3(x):
    return _fixed(x, 3)


def _welch(a, b) -> tuple[float | None, float | None]:
    from scipy.stats import ttest_ind

    if len(a) < 2 or len(b) < 2:
        return None, None
    r = ttest_ind(a, b, equal_var=False)
    return float(r.statistic), float(r.pvalue)


def headline_table(rows, base) -> tuple[Table | None, dict]:
    """Final checkpoints at each method's best setting (most seeds, then accuracy): GSM8K and mean forgetting as
    mean ± sd over seeds with 2-decimal pp, KL, and Welch t / p of mean forgetting vs GRPO. The seeded comparison
    is the primary Q0/Q1 result; pooled-frontier values mix settings and seeds."""
    if base is None:
        return None, {}
    finals = {m: final_rows(best_setting_rows(rows, m)) for m in sorted(_methods(rows), key=lambda m: (m != "grpo", m))}
    finals = {m: fs for m, fs in finals.items() if fs}
    g = [f["forgetting_mean"] for f in finals.get("grpo", [])]
    body, out = [], {}
    for m, fs in finals.items():
        acc = [f["gsm8k_acc"] for f in fs]
        fg = [f["forgetting_mean"] for f in fs]
        kl = [f["kl_fwd"] for f in fs if f.get("kl_fwd") is not None]
        klt = [kl_detail(f)["per_token"] for f in fs if kl_detail(f)]
        sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else None  # noqa: E731
        t, p = _welch(fg, g) if m != "grpo" else (None, None)
        out[m] = {"n": len(fs), "acc": (float(np.mean(acc)), sd(acc)), "forgetting": (float(np.mean(fg)), sd(fg)),
                  "kl": float(np.mean(kl)) if kl else None, "t": t, "p": p,
                  "diff": float(np.mean(fg) - np.mean(g)) if g and m != "grpo" else None}
        body.append([Const(label(fs[0])), str(len(fs)),
                     pct(np.mean(acc)) + (f" ± {pct(sd(acc))}" if sd(acc) is not None else ""),
                     pct(np.mean(acc) - base["gsm8k_acc"]),
                     pp2(np.mean(fg)) + (f" ± {pp2(sd(fg))}" if sd(fg) is not None else ""),
                     kl2(np.mean(kl)) if kl else NA, _f3(np.mean(klt)) if klt else NA,
                     pp2(out[m]["diff"]) if out[m]["diff"] is not None else NA,
                     _fixed(t, 2) if t is not None else NA, _fixed(p, 3) if p is not None else NA])
    if not body:
        return None, out
    return Table("Headline: final checkpoints at each method's best setting (mean ± sd over seeds; forgetting in pp "
                 "with 2 decimals; Welch t-test of mean forgetting vs GRPO, uncorrected)",
                 ["Best setting (final ckpt)", "Seeds", "GSM8K (%)", "GSM8K change (pp)", "Forgetting (pp)",
                  "KL (nats/seq)", "KL (nats/token)", "Forgetting − GRPO (pp)", "Welch t", "p"], body), out


def held_notes(curves: dict, targets, rows) -> list:
    """Matched-accuracy caveat: a target below a method's lowest Pareto-frontier point gets
    that point's value held downward, not an interpolation. Names the run that supplies the held value."""
    from rlrazor.analysis.stats import pareto_frontier

    notes = []
    for m, c in curves.items():
        front = pareto_frontier(c["points"])
        if not front:
            continue
        a0, f0 = front[0]
        for t in targets:
            if t < a0:
                src = [r for r in rows if r["method"] == m and point(r) == (a0, f0)]
                who = Const(f"{src[0]['run_id']} step {src[0]['step']}") if src else Const("the base model")
                notes.append((f"Caveat: {pct(t)}% lies below {m}'s lowest frontier point (", pct(a0), "% GSM8K, ",
                              pct(f0), " pp, from ", who, "); its interp. value there is that one point held "
                              "downward, not an interpolation, and its fit is anchored by it."))
    return notes


def kl_detail(row) -> dict | None:
    """Per-sequence forward-KL statistics from a row's saved log-ratio sums (log π₀ − log π over each base sample's
    response, y ~ π₀) and token counts (R46/R50). The base row has KL 0 by definition (all statistics 0)."""
    s, n = row.get("kl_logratio_sums"), row.get("kl_n_tokens")
    if not s or not n:
        if is_base(row) and row.get("kl_fwd") == 0:
            return {k: 0.0 for k in ("mean", "per_token", "median", "median_per_token", "p90", "top10_share",
                                     "frac_neg", "rho_len")} | {"n": 0}
        return None
    from scipy.stats import spearmanr

    s, n = np.asarray(s, float), np.asarray(n, float)
    k = max(1, int(round(0.1 * len(s))))
    top = np.sort(s)[::-1][:k].sum()
    return {"mean": float(s.mean()), "per_token": float(s.sum() / n.sum()), "median": float(np.median(s)),
            "median_per_token": float(np.median(s / n)), "p90": float(np.percentile(s, 90)),
            "top10_share": float(top / s.sum()) if s.sum() else float("nan"), "frac_neg": float((s < 0).mean()),
            "rho_len": float(spearmanr(s, n).statistic), "n": len(s)}


def top_kl_samples(row, other, k: int = 5) -> list[dict]:
    """The k base samples with the largest log-ratio under `row`, with token counts, per-token values and the
    log-ratio of the same sample under `other` (sample indices are shared: every checkpoint scores the same
    cached base samples)."""
    s, n = np.asarray(row["kl_logratio_sums"], float), np.asarray(row["kl_n_tokens"], float)
    o = np.asarray(other["kl_logratio_sums"], float)
    if len(o) != len(s) or list(other["kl_n_tokens"]) != list(row["kl_n_tokens"]):
        raise ValueError("rows were not scored on the same base samples")
    return [{"index": int(i), "logratio": float(s[i]), "tokens": int(n[i]), "per_token": float(s[i] / n[i]),
             "other": float(o[i]), "rank_other": int((o > o[i]).sum()) + 1} for i in np.argsort(-s, kind="stable")[:k]]


def kl_rank_agreement(a, b) -> float:
    """Spearman correlation of two checkpoints' per-sample log-ratios: do they agree on WHICH samples drift."""
    from scipy.stats import spearmanr

    return float(spearmanr(a["kl_logratio_sums"], b["kl_logratio_sums"]).statistic)


SFT_LIKE = ("teacher_sft", "sft", "rft_offline", "rft_iter", "self_rewrite", "rl_distill_sft")


def kl_fit_subsets(rows) -> list[tuple[str, int, float, float, bool]]:
    """Quadratic R² and Spearman ρ of mean forgetting on KL for named subsets (R46: the Q0-only fit is the
    headline compared with the paper's 0.71, which covers SFT and RL models only). `rows`: non-diverged, non-smoke
    rows incl. the base row. Returns (label, n, R², ρ, headline)."""
    from scipy.stats import spearmanr

    base = [r for r in rows if is_base(r)]
    nb = [r for r in rows if not is_base(r)]
    q0 = [r for r in nb if r.get("arm") == "q0" and is_clean(r)]
    q1 = [r for r in nb if r.get("arm") == "q1"]
    sl = [r for r in nb if r["method"] in SFT_LIKE and r.get("arm") in ("q0", "q1")]
    low = lambda rs, lr: [r for r in rs if r["lr"] <= lr * (1 + 1e-9)]  # noqa: E731
    pt = lambda r: (kl_detail(r) or {}).get("per_token")  # noqa: E731
    med = lambda r: (kl_detail(r) or {}).get("median")  # noqa: E731
    seq = lambda r: r.get("kl_fwd")  # noqa: E731
    subsets = [
        ("Q0: teacher SFT + GRPO + base, sequence KL (headline)", q0 + base, seq, True),
        ("Q0, per-token KL", q0 + base, pt, False),
        ("Q0, median per-sequence KL", q0 + base, med, False),
        ("Q0 + Q1, all methods + base, sequence KL", q0 + q1 + base, seq, False),
        ("SFT-like (teacher SFT, RFT, self-rewrite, RL-distill), all LRs", sl, seq, False),
        ("SFT-like, LR ≤ 1e-5", low(sl, 1e-5), seq, False),
        ("SFT-like, LR ≤ 3e-6", low(sl, 3e-6), seq, False),
        ("All methods, LR ≤ 3e-6", low([r for r in q0 + q1], 3e-6), seq, False),
        ("GRPO only", [r for r in q0 if r["method"] == "grpo"], seq, False),
    ]
    out = []
    for lab, rs, key, head in subsets:
        pts = [(key(r), r["forgetting_mean"]) for r in rs if key(r) is not None]
        if len(pts) < 4:
            continue
        x, y = zip(*pts)
        _, r2 = quadratic_r2(list(x), list(y))
        out.append((lab, len(pts), float(r2), float(spearmanr(x, y).statistic), head))
    return out


def kl_near_target(rows, target: float, tol: float = 0.01) -> list[list]:
    """Checkpoints of each method's best setting (all its seeds and checkpoints) within ±tol accuracy of `target`:
    count, mean accuracy, mean forgetting, mean KL (sequence and per token). Compares KL at (nearly) equal new-task
    accuracy instead of at the final checkpoint, where the methods' accuracies differ."""
    body = []
    for m in _methods(rows):
        rs = [r for r in best_setting_rows(rows, m) if abs(r["gsm8k_acc"] - target) <= tol and kl_detail(r)]
        if not rs:
            continue
        body.append([Const(m), str(len(rs)), pct(np.mean([r["gsm8k_acc"] for r in rs])),
                     pp2(np.mean([r["forgetting_mean"] for r in rs])), kl2(np.mean([r["kl_fwd"] for r in rs])),
                     _f3(np.mean([kl_detail(r)["per_token"] for r in rs])),
                     Const(", ".join(f"{r['run_id'].rsplit('-', 1)[0].split('-', 1)[1]}@{r['step']}" for r in rs))])
    return body


_LOG_DICT = re.compile(r"^\{'(loss|train_runtime)'.*\}$")


def train_log_summary(text: str, n: int = 10) -> dict | None:
    """Mean of the first and last `n` logged optimizer steps of a TRL training log (lines printed as Python dicts):
    entropy, reward, completion length, clipped ratio (GRPO) — whatever the log has. None if nothing is logged."""
    import ast

    steps = []
    for line in text.splitlines():
        line = line.strip()
        if not _LOG_DICT.match(line) or "'loss'" not in line:
            continue
        try:
            d = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            continue
        steps.append({k: float(v) for k, v in d.items() if isinstance(v, (str, int, float))
                      and re.fullmatch(r"-?[\d.]+(e[-+]?\d+)?", str(v))})
    if not steps:
        return None
    keys = {"entropy": "entropy", "reward": "reward", "length": "completions/mean_length",
            "clipped": "completions/clipped_ratio"}

    def avg(part, k):
        v = [s[k] for s in part if k in s]
        return float(np.mean(v)) if v else None

    return {"steps": len(steps), "first": {a: avg(steps[:n], k) for a, k in keys.items()},
            "last": {a: avg(steps[-n:], k) for a, k in keys.items()}}


_CHARGED = re.compile(r"^finished (\S+): exit (\d+), charged \$([\d.]+)")


def parse_launch_log(text: str, log: str = "") -> list[dict]:
    """Per-run charges a launch.py log reports ("finished <run_id>: exit <code>, charged $<x>"). The charge is the
    launcher's measured GPU-seconds × list price (or the projection for a failed run), not a billing figure."""
    return [{"run_id": m.group(1), "exit": int(m.group(2)), "charged": float(m.group(3)), "log": log}
            for m in (_CHARGED.match(line.strip()) for line in text.splitlines()) if m]


def logs_to_results(logs_dir: Path, results_dir: Path) -> list[Path]:
    """Commit-able summaries of the (git-ignored) launcher and training logs: launcher_charges.jsonl and
    train_log_summary.jsonl in `results_dir`."""
    charges = []
    for p in sorted(logs_dir.glob("launch-*.log")):
        charges += parse_launch_log(p.read_text(errors="replace"), p.name)
    summaries = []
    for p in sorted((logs_dir / "logs").glob("*.log")):
        s = train_log_summary(p.read_text(errors="replace"))
        if s:
            summaries.append({"run_id": p.stem, **s})
    committed = [float(m) for p in sorted(logs_dir.glob("launch-*.log"))
                 for m in re.findall(r"summary \(committed \$([\d.]+) of budget", p.read_text(errors="replace"))]
    # Stage 2: outputs/launch/stage2_*.log, a separate guard ($96, R54)
    committed2 = [float(m) for p in sorted((logs_dir / "launch").glob("stage2_*.log"))
                  for m in re.findall(r"summary \(committed \$([\d.]+) of budget", p.read_text(errors="replace"))]
    out = []
    if committed:  # the guard's end state: the largest final figure of any launch (launches chain their --spent)
        path = results_dir / "launcher_committed.json"
        state = {"committed": max(committed)}
        if committed2:
            state["stage2_committed"] = max(committed2)
        path.write_text(json.dumps(state) + "\n")
        out.append(path)
    for name, data in (("launcher_charges.jsonl", charges), ("train_log_summary.jsonl", summaries)):
        path = results_dir / name
        path.write_text("".join(json.dumps(d) + "\n" for d in data))
        out.append(path)
    return out


BILLING_PHASES = (  # (label, first hour, last hour) in UTC, from the launcher logs' launch times
    ("Engineering: smoke tests, deterministic evaluation, pilot runs", "2026-11-21T00", "2026-11-22T05"),
    ("Teacher data, Q0 sweep at the paper's LRs, diagnostics", "2026-11-22T06", "2026-11-22T09"),
    ("Q0 at lower LRs, extra seeds, Q1 arms", "2026-11-22T10", "2026-11-22T14"),
    ("Stage 2: Q2, Q3, Q1 remainder, Q4", "2026-11-22T15", "2026-11-22T23"),
)
STAGE1_PHASES = 3  # the first three phases are stage 1 (budget $41.50); stage 2 runs under a $96 guard (R54)
BENCH_ITEMS = {"hellaswag": 500, "mmlu": 570, "truthfulqa": 500, "winogrande": 500, "ifeval": 541, "humaneval": 164}
RUNTIME_PKGS = ("torch", "transformers", "trl", "vllm", "lm-eval", "datasets", "peft", "accelerate")


def billing_tables(billing: dict, charges: list[dict], launcher_committed: float | None,
                   stage2_committed: float | None = None) -> tuple[list, dict]:
    """Actual dollars from Modal's billing report (hourly rows), by phase and by app, plus launcher-measured
    per-run charges grouped by method (successful runs only; failed attempts were rate-limited before any
    container ran and were relaunched)."""
    rows = billing.get("rows", [])
    total = sum(float(r["cost"]) for r in rows)
    phase = []
    for lab, lo, hi in BILLING_PHASES:
        v = sum(float(r["cost"]) for r in rows if lo <= r["interval_start"][:13] <= hi)
        phase.append([Const(lab), Const(f"{lo[5:]}–{hi[5:]} UTC"), f"{v:.2f}"])
    stage1 = sum(float(r["cost"]) for r in rows if r["interval_start"][:13] <= BILLING_PHASES[STAGE1_PHASES - 1][2])
    phase.insert(STAGE1_PHASES, [Const("Stage 1 subtotal"), Const(""), f"{stage1:.2f}"])
    phase.append([Const("Total (Modal billing)"), Const(""), f"{total:.2f}"])
    apps = defaultdict(float)
    for r in rows:
        apps[r["description"]] += float(r["cost"])
    main = apps.pop("rlrazor", 0.0)
    app_body = [[Const("rlrazor (all training and evaluation)"), f"{main:.2f}"],
                [Const("one-off diagnostic apps (" + ", ".join(sorted(apps)) + ")"), f"{sum(apps.values()):.2f}"]]
    by_m = defaultdict(list)
    for c in charges:
        if c["exit"] == 0:
            by_m[c["run_id"].split("-lr")[0]].append(c["charged"])
    run_body = [[Const(m), str(len(v)), f"{sum(v):.2f}", f"{np.mean(v):.2f}"] for m, v in sorted(by_m.items())]
    launched = sum(sum(v) for v in by_m.values())
    run_body.append([Const("all launcher-measured runs"), str(sum(len(v) for v in by_m.values())), f"{launched:.2f}",
                     NA])
    tables = [Table("Actual spend from Modal billing, by phase (USD)", ["Phase", "Billing hours", "USD"], phase),
              Table("Actual spend from Modal billing, by app (USD)", ["App", "USD"], app_body),
              Table("Stage 1: launcher-measured cost per grid run (GPU-seconds × list price; USD; stage 2 is in "
                    "the per-arm table below)",
                    ["Method", "Runs", "Total", "Mean per run"], run_body)]
    if launcher_committed is not None:
        guard = [[Const("Guard ceiling"), Const("41.50")],
                 [Const("Launcher committed at the end (conservative; billing baseline + measured + projections)"),
                  f"{launcher_committed:.2f}"]]
        if stage2_committed is not None:
            guard += [[Const("Stage 2 guard ceiling (R54; $100 total ceiling)"), Const("96.00")],
                      [Const("Stage 2: launcher committed at the end (failed runs charged their projection)"),
                       f"{stage2_committed:.2f}"]]
        guard.append([Const("Modal billing, month total"), f"{total:.2f}"])
        tables.append(Table("Budget guard", ["Item", "USD"], guard))
    return tables, {"total": total, "stage1": stage1, "by_method": {m: sum(v) for m, v in by_m.items()}}


def stage2_cost_table(rows, train_times: list[dict]) -> Table | None:
    """Stage 2 measured cost per arm and method: eval GPU-seconds of its result rows and training
    wall-seconds (train_times), priced as the launcher prices them (scripts/cost_report.py list prices). Stage 2's
    launchers were replaced mid-flight (GPU cap), so their logs do not list every run; this table does."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import cost_report as cr

    p = cr.Prices.from_env()
    usd, runs = defaultdict(float), defaultdict(set)
    key = {}
    for r in rows:
        if not is_base(r) and not is_stage1(r):
            k = (r["arm"], r["method"], "WiSE-FT" if r.get("wise_alpha") else "LoRA" if r.get("lora") else "")
            key[r["run_id"]] = k
            usd[k] += p.eval_usd(r.get("gpu_seconds") or 0.0)
            runs[k].add(r["run_id"])
    for t in train_times:
        k = key.get(t["run_id"])
        if k is not None:
            usd[k] += p.train_usd(t["wall_seconds"])
    if not usd:
        return None
    body = [[Const(a), Const(" ".join(x for x in (m, v) if x)), str(len(runs[(a, m, v)])), f"{usd[(a, m, v)]:.2f}"]
            for a, m, v in sorted(usd)]
    body.append([Const("all stage-2 runs with results"), Const(""), str(sum(len(v) for v in runs.values())),
                 f"{sum(usd.values()):.2f}"])
    return Table("Stage 2: measured cost per arm and method (GPU-seconds × list price; USD)",
                 ["Arm", "Method", "Runs", "USD"], body)


def runtime_table(req_text: str, train_gpu: str = "A100-40GB", eval_gpu: str = "L4") -> Table:
    pins = dict(re.findall(r"^([A-Za-z0-9_.-]+)==([^\s;]+)", req_text, flags=re.M))
    body = [[Const(p), Const(pins.get(p, NA))] for p in RUNTIME_PKGS]
    body += [[Const("Python (Modal image)"), Const("3.11")], [Const("Training GPU"), Const(train_gpu)],
             [Const("Evaluation GPU"), Const(eval_gpu)]]
    return Table("Runtime on Modal (requirements-modal.txt; authoritative for every GPU result)",
                 ["Component", "Version"], body)


def runtime_cost_tables(rows, train_times: list[dict], all_rows=None) -> list[Table]:
    """Wall-clock training time per run (train_times.jsonl) and eval GPU-seconds per checkpoint (results rows)."""
    by_m = defaultdict(list)
    for t in train_times:
        c = t.get("config") or {}
        if not c.get("smoke") and c.get("arm") != "pilot":
            by_m[c.get("method")].append(t["wall_seconds"])
    body = [[Const(m), str(len(v)), f"{np.mean(v) / 3600:.2f}", f"{sum(v) / 3600:.2f}"] for m, v in sorted(by_m.items())]
    tables = [Table("Training wall-clock per run (A100-40GB; hours)", ["Method", "Runs", "Mean (h)", "Total (h)"],
                    body)]
    ev_rows = []
    for lab, rs in (("All evaluated grid checkpoints (incl. diverged runs)", all_rows), ("Non-diverged checkpoints", rows)):
        ev = [r["gpu_seconds"] for r in (rs or []) if r.get("gpu_seconds") and not is_base(r)]
        if ev:
            ev_rows.append([Const(lab), str(len(ev)), f"{np.mean(ev):.0f}", f"{sum(ev) / 3600:.2f}"])
    if ev_rows:
        tables.append(Table("Evaluation per checkpoint (L4; GSM8K, 6 benchmarks, KL)",
                            ["Checkpoints", "Count", "Mean GPU-seconds", "Total GPU-hours"], ev_rows))
    return tables


def is_stage1(row: dict) -> bool:
    """Rows of stage 1 (pilot, Q0, Q1 at its stage-1 seeds), i.e. not added by stage 2 (R54):
    Q2/Q3/Q4 arms, iterative RFT and the self-rewrite seeds 1-2."""
    c = row.get("config") or row
    if c.get("arm") in ("q2", "q3", "q4"):
        return False
    return not (c.get("method") == "rft_iter" or (c.get("method") == "self_rewrite" and (c.get("seed") or 0) > 0))


def plan_costs(rows, train_times: list[dict], only_unrun: bool = True,
               title: str | None = None) -> tuple[Table, float]:
    """Projected cost of the stages that were not run (Q1 remainder, Q2, Q3, Q4, staleness), from the launcher's
    cost model (scripts/cost_report.py, units measured from this study's rows) and the configs make_configs.py
    would generate at the chosen best LRs and final checkpoints."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import cost_report as cr
    import make_configs as mc

    u, p = cr.measure(rows, train_times), cr.Prices.from_env()
    clean = [r for r in rows if r.get("arm") in ("q0", "q1") and is_clean(r)]

    def best(m):
        rs = best_setting_rows(clean, m)
        fs = [f for f in final_rows(rs) if f["seed"] == 0] or final_rows(rs)
        return (rs[0]["lr"], f"{fs[0]['run_id']}/step-{fs[0]['step']}") if rs else (None, None)

    (sft_lr, sft_fin), (grpo_lr, grpo_fin), (rft_lr, rft_fin) = best("teacher_sft"), best("grpo"), best("rft_offline")
    kw = dict(best_sft_lr=sft_lr, best_grpo_lr=grpo_lr, best_rft_lr=rft_lr, grpo_final=grpo_fin, sft_final=sft_fin,
              rft_final=rft_fin, crossover=("0.1", "0.25"))
    groups = [("Q1: iterative RFT, 3 seeds (s0 refused by the guard)", "q1", lambda c: c.method == "rft_iter"),
              ("Q1: self-rewrite, 2 more seeds", "q1", lambda c: c.method == "self_rewrite" and c.seed > 0),
              ("Q2: label / reward noise p ∈ {0.2, 0.4}, 3 seeds each", "q2", lambda c: True),
              ("Q3: general-data mixing, 5 fractions (lower bound: the model omits the added rows)", "q3",
               lambda c: True),
              ("Q3: 2 extra seeds at the 2 crossover fractions (assumed f = 0.1, 0.25; lower bound)", "q3_seeds",
               lambda c: True),
              ("Q4: LoRA r=64 (2 LRs) and GKD restore (100 steps)", "q4", lambda c: c.wise_alpha == 0),
              ("Q4: WiSE-FT, 2 sources × 3 α (CPU interpolation + 1 eval)", "q4", lambda c: c.wise_alpha > 0)]
    from rlrazor.config import run_id as _rid

    done = {r["run_id"] for r in rows} if only_unrun else set()
    q3_seeded = only_unrun and any(r.get("arm") == "q3" and r.get("seed", 0) > 0 for r in rows)
    body, total = [], 0.0
    for lab, stage, keep in groups:
        if stage == "q3_seeds" and q3_seeded:  # run at the measured crossover's bracket, not the assumed one
            continue
        cfgs = [c for _, c in mc.stage_configs(stage, **kw) if keep(c) and _rid(c) not in done]
        if not cfgs:  # every config of this group has results (stage 2)
            continue
        usd = 0.0
        for c in cfgs:
            if not only_unrun and c.mix_f > 0:  # the stage-1 table's lower bound: the added Tulu rows not costed
                c = dataclasses.replace(c, mix_f=0.0)
            if c.wise_alpha > 0:
                usd += cr.run_cost(c, u, p, train=False, n_evals=1)[0] + p.cpu_usd(p.wise_cpu_s)
            else:
                usd += cr.run_cost(c, u, p)[0]
        total += usd
        body.append([Const(lab), str(len(cfgs)), f"{usd:.2f}"])
    stale = cr.staleness(u, p, mc.GRPO_STEPS)[0]
    st_train, st_eval, _ = cr.group_usd(stale, p, u.eval_s, 3)
    st_usd = st_train + st_eval
    body.append([Const("Optional: async staleness, GRPO k ∈ {4, 16, 64} (needs a lagged sampler)"), str(stale.count),
                 f"{st_usd:.2f}"])
    body.append([Const("Total of the configs above not yet run (without staleness)" if only_unrun else
                       "Total, Q1 remainder + Q2–Q4 (without staleness)"), "", f"{total:.2f}"])
    title = title or ("Not run: projected cost of the remaining stages (launcher cost model; GSM8K-sized runs at "
                      "the chosen LRs; 3 GRPO / 2 SFT-like checkpoints per run)")
    return Table(title, ["Stage", "Configs", "Projected USD"], body), total


def data_tables(teacher: dict | None, dstats: list[dict]) -> list[Table]:
    tables = []
    if teacher:
        tables.append(Table("Teacher SFT data (Qwen2.5-7B-Instruct, T = 1, up to 16 samples per prompt, keep 1 "
                            "correct)", ["Item", "Value"], [
                                [Const("Prompts"), Const(str(teacher["n_prompts"]))],
                                [Const("Rows kept"), Const(str(teacher["n_rows"]))],
                                [Const("Keep rate (%)"), pct(teacher["keep_rate"])],
                                [Const("Solved in the first 4 samples (%)"), pct(teacher["first_round_solved_frac"])],
                                [Const("Samples per prompt"), _fixed(teacher["samples_per_prompt"], 2)],
                                [Const("Correct rate over all samples (%)"), pct(teacher["correct_rate"])],
                                [Const("Boxed-answer rate (%)"), pct(teacher["boxed_rate"])]]))
    seen, body, iters = set(), [], []
    for d in dstats:
        m = d["run_id"].split("-lr")[0]
        rounds = sorted(k for k in d if k.startswith("round_"))
        if rounds:  # iterative RFT: one data set per round, sampled from the current model
            iters.append([Const(f"{m} s{d['run_id'].split('-s')[-1].split('-')[0]}"),
                          *[pct(d[k]["kept_frac"]) for k in rounds]])
            continue
        key = (m, d.get("kept_frac"), d.get("fallback_rate"), d.get("source"))
        if m == "teacher_data" or key in seen:  # teacher set: table above; the same data set for several LRs: once
            continue
        seen.add(key)
        src = d.get("source") or ("own rewrites of the gold answers" if m == "self_rewrite" else NA)
        src = "/".join(src.split("/")[-2:]) if src.startswith("/vol/") else src.split("/")[-1]
        body.append([Const(f"{m} s{d['run_id'].split('-s')[-1].split('-')[0]}"), Const(src),
                     Const(str(d.get("n_rows", NA))),
                     pct(d["kept_frac"]) if d.get("kept_frac") is not None else NA,
                     _fixed(d["mean_correct_per_prompt"], 2) if d.get("mean_correct_per_prompt") is not None else NA,
                     pct(d["fallback_rate"]) if d.get("fallback_rate") is not None else NA])
    if body:
        tables.append(Table("Self-generated SFT data sets", ["Method", "Sampled from", "Rows",
                                                                     "Prompts kept (%)", "Correct per prompt",
                                                                     "Gold fallback (%)"], body))
    if iters:
        n = max(len(r) for r in iters) - 1
        tables.append(Table("Iterative RFT: prompts with a kept correct sample, per round (%)",
                            ["Run", *[f"Round {i + 1}" for i in range(n)]], iters))
    return tables


def bench_se_table(base) -> Table:
    """Item-sampling standard error of each benchmark subset at the base score, binomial sqrt(p(1−p)/n); a
    difference of two independent evaluations has √2 × that. TruthfulQA-MC2 is a per-item probability, for which
    the binomial value is an upper bound. Paired (same-item) differences have smaller SE; the per-item outputs
    needed to estimate it were not saved (lm-eval ran without --log_samples)."""
    body = []
    for b in BENCH:
        p, n = base["bench"][b], BENCH_ITEMS[b]
        se = np.sqrt(p * (1 - p) / n)
        body.append([Const(b), Const(str(n)), pct(p), pct(se), pct(np.sqrt(2) * se)])
    if base.get("gsm8k_acc") is not None:  # the new-task metric, for scale (full 1,319-item test set)
        p, n = base["gsm8k_acc"], 1319
        se = np.sqrt(p * (1 - p) / n)
        body.append([Const("GSM8K (new task, full test set)"), Const(str(n)), pct(p), pct(se), pct(np.sqrt(2) * se)])
    return Table("Benchmark subsets and item-sampling SE at the base score", ["Benchmark", "Items", "Base (%)",
                                                                             "SE (pp)", "SE of an unpaired difference (pp)"],
                 body)


def diag_table(diag: dict) -> Table:
    body = [[Const(k), pct(v["acc"]), pct(v["boxed_rate"]), _fixed(v["mean_tokens"], 1), pct(v["frac_512"])]
            for k, v in diag.items()]
    return Table("Greedy diagnostic on the first 200 GSM8K test prompts (results/q0_diag_samples.json)",
                 ["Checkpoint", "Accuracy (%)", "Boxed (%)", "Mean tokens", "Hit 512 tokens (%)"], body)


def kl_sections(rows, base, headline, extras) -> tuple[list[Section], dict]:
    """The KL analysis (R46/R50): subset fits with the Q0-only headline vs the paper's 0.71,
    per-token and median KL per checkpoint, KL at near-equal accuracy, the top samples, rank agreement between
    checkpoints, and the GRPO training dynamics (entropy) behind the sharpening reading."""
    summary, sec = {}, Section("Forward KL: subsets, per-token and per-sequence detail")
    fits = kl_fit_subsets(rows)
    if fits:
        sec.blocks.append(Table("Quadratic fit of mean forgetting on forward KL, by subset (headline: Q0 only, the "
                                "population of the paper's 0.71)", ["Subset", "Checkpoints", "R²", "Spearman ρ",
                                                                    "R² (paper, LLMs)"],
                                [[Const(lab), str(n), f"{r2:.2f}", f"{rho:.2f}",
                                  Const(f"{PAPER['kl_r2']:.2f}") if head else Const("")]
                                 for lab, n, r2, rho, head in fits]))
        summary["fits"] = fits
    nb = [r for r in rows if not is_base(r) and kl_detail(r)]
    body = []
    for r in sorted(nb, key=lambda r: (r["method"], r["lr"], r["seed"], r["step"])):
        d = kl_detail(r)
        body.append([Const(f"{label(r)} s{r['seed']}"), Const(str(r["step"])), pct(r["gsm8k_acc"]),
                     pct(r["forgetting_mean"]), kl2(d["mean"]), kl2(d["median"]), _f3(d["per_token"]),
                     _f3(d["median_per_token"]), _fixed(d["top10_share"], 2), _fixed(d["rho_len"], 2)])
    if body:
        sec.blocks.append(Table("Per-checkpoint forward KL on the 256 cached base samples: mean and median per "
                                "sequence (nats), per token (total log-ratio / total tokens) and median per token, "
                                "share of the total from the top 10% of samples, Spearman ρ with response length",
                                ["Setting", "Step", "GSM8K (%)", "Forgetting (pp)", "KL mean", "KL median",
                                 "KL/token", "Median KL/token", "Top-10% share", "ρ(len)"], body))
    if headline is not None:
        near = kl_near_target([r for r in rows if not is_base(r) and r.get("arm") in ("q0", "q1") and is_clean(r)],
                              headline)
        if near:
            sec.blocks.append(Table(f"Forward KL at near-equal accuracy: checkpoints of each method's best setting "
                                    f"within 1 point of {pct(headline)}% GSM8K", ["Method", "Checkpoints", "Mean GSM8K (%)", "Mean forgetting (pp)",
                                              "Mean KL (nats/seq)", "Mean KL (nats/token)", "Checkpoints used"], near))
    clean = [r for r in rows if r.get("arm") in ("q0", "q1") and is_clean(r) and not is_base(r)]

    def seed_final(m, seed):
        fs = [f for f in final_rows(best_setting_rows(clean, m)) if f["seed"] == seed and kl_detail(f)]
        return fs[0] if fs else None

    g0, s0 = seed_final("grpo", 0), seed_final("teacher_sft", 0)
    if g0 and s0:
        tops = []
        for a, b, an, bn in ((g0, s0, "grpo", "teacher_sft"), (s0, g0, "teacher_sft", "grpo")):
            for t in top_kl_samples(a, b):
                tops.append([Const(an), Const(str(t["index"])), Const(str(t["tokens"])), kl2(t["logratio"]),
                             _f3(t["per_token"]), kl2(t["other"]), Const(str(t["rank_other"]))])
        sec.blocks.append(Table(f"Top-5 base samples by log-ratio at the seed-0 final checkpoints of the best settings "
                                f"({label(g0)} vs {label(s0)}); the other method's value and rank (of 256) on the "
                                "same sample", ["Ranked by", "Sample", "Tokens", "Log-ratio (nats)", "Per token",
                                                "Other method (nats)", "Other's rank"], tops))
        summary["top"] = tops
        pairs = [("grpo s0", "teacher_sft s0", g0, s0)]
        for m, ref in (("grpo", g0), ("teacher_sft", s0)):
            for sd in (1, 2):
                o = seed_final(m, sd)
                if o:
                    pairs.append((f"{m} s0", f"{m} s{sd}", ref, o))
        rd = [f for f in final_rows([r for r in clean if r["method"] == "rl_distill_sft"]) if kl_detail(f)]
        if rd:
            pairs.append(("grpo s0", "rl_distill_sft s0", g0, rd[0]))
        sec.blocks.append(Table("Do checkpoints drift on the same samples? Spearman ρ of per-sample log-ratios "
                                "(final checkpoints, best settings)", ["Checkpoint A", "Checkpoint B", "Spearman ρ"],
                                [[Const(a), Const(b), _fixed(kl_rank_agreement(x, y), 2)] for a, b, x, y in pairs]))
    # figure: KL vs accuracy (best-LR and other low-LR checkpoints of SFT and GRPO) and per-token distributions
    low = [r for r in clean if r["lr"] <= 3e-6 * (1 + 1e-9) and r["method"] in ("teacher_sft", "grpo") and kl_detail(r)]
    acc_kl = {m: [(r["gsm8k_acc"], r["kl_fwd"], r["lr"]) for r in low if r["method"] == m]
              for m in ("teacher_sft", "grpo")}
    per_tok = {}
    for m in ("teacher_sft", "rft_offline", "grpo", "rl_distill_sft"):
        fs = [f for f in final_rows(best_setting_rows(clean, m)) if f.get("kl_logratio_sums")]
        if fs:
            per_tok[m] = list(np.concatenate([np.asarray(f["kl_logratio_sums"]) / np.asarray(f["kl_n_tokens"])
                                              for f in fs]))
    if any(acc_kl.values()) and per_tok:
        F.kl_detail_figure(extras.get("_out"), "kl_detail", {m: v for m, v in acc_kl.items() if v}, per_tok,
                           base["gsm8k_acc"] if base else None)
    tls = extras.get("train_log_summary") or []
    kept = {r["run_id"] for r in rows}  # non-diverged grid runs only
    body = []
    for t in tls:
        if t["run_id"] not in kept or not t["run_id"].startswith("grpo-") or t["first"].get("reward") is None:
            continue
        f, l_ = t["first"], t["last"]
        body.append([Const(t["run_id"]), Const(str(t["steps"])), _f3(f["entropy"]), _f3(l_["entropy"]),
                     _f3(f["reward"]), _f3(l_["reward"]), _fixed(f["length"], 1), _fixed(l_["length"], 1)])
    if body:
        sec.blocks.append(Table("GRPO training dynamics: mean over the first and last 10 logged steps (policy entropy "
                                "per token on its own rollouts, reward, completion length)",
                                ["Run", "Steps logged", "Entropy first", "Entropy last", "Reward first", "Reward last",
                                 "Length first", "Length last"], body))
    if extras.get("diag"):
        sec.blocks.append(diag_table(extras["diag"]))
    return [sec], summary


def extra_sections(rows, base, headline, extras, all_rows=None) -> tuple[list[Section], dict]:
    """Sections that need inputs beyond the result rows (data stats, billing, logs, runtime pins). Each is skipped
    when its input is absent, so synthetic-data tests keep working."""
    secs, summary = [], {}
    data = data_tables(extras.get("teacher_stats"), extras.get("dataset_stats") or [])
    if base is not None:
        data.append(bench_se_table(base))
    if data:
        secs.append(Section("Data and benchmark subsets", data))
    cost = Section("Cost, run time and runtime")
    if extras.get("billing"):
        t, summary["billing"] = billing_tables(extras["billing"], extras.get("launcher_charges") or [],
                                               extras.get("launcher_committed"), extras.get("stage2_committed"))
        cost.blocks += t
    if extras.get("train_times"):
        t2 = stage2_cost_table(all_rows or rows, extras["train_times"])
        if t2:
            cost.blocks.append(t2)
    if extras.get("train_times"):
        cost.blocks += runtime_cost_tables(rows, extras["train_times"], all_rows)
        try:
            t, summary["plan_total"] = plan_costs(rows, extras["train_times"])
            cost.blocks.append(t)
        except (SystemExit, ValueError, ImportError) as e:  # missing units or best settings: no plan table
            cost.blocks.append(f"Plan cost unavailable: {e}")
    if extras.get("requirements"):
        cost.blocks.append(runtime_table(extras["requirements"]))
    if cost.blocks:
        secs.append(cost)
    return secs, summary


def load_extras(results_dir: Path, root: Path) -> dict:
    """Optional inputs next to results.jsonl (all committed under results/) and the Modal pins."""
    def js(name):
        p = results_dir / name
        return json.loads(p.read_text()) if p.exists() else None

    def jl(name):
        p = results_dir / name
        return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else None

    req = root / "requirements-modal.txt"
    committed = js("launcher_committed.json")
    return {"billing": js("billing_hourly.json"), "launcher_charges": jl("launcher_charges.jsonl") or [],
            "launcher_committed": committed["committed"] if committed else None,
            "stage2_committed": (committed or {}).get("stage2_committed"),
            "train_times": jl("train_times.jsonl"), "dataset_stats": jl("dataset_stats.jsonl"),
            "teacher_stats": js("teacher_data_stats.json"), "diag": js("q0_diag_samples.json"),
            "train_log_summary": jl("train_log_summary.jsonl"),
            "requirements": req.read_text() if req.exists() else None}


# ---------- sections ----------


def analyze(rows, out: Path, extras: dict | None = None) -> dict:
    """Write figures, tables.md, tables.tex and numbers.json into `out`. Returns a summary of the
    computed numbers (unrounded) and the notes."""
    rows, notes = filter_rows(rows)
    div_ids = diverged_run_ids(rows)
    all_rows, div_rows = rows, [r for r in rows if r["run_id"] in div_ids]
    if div_ids:
        notes.append(f"excluded {len(div_ids)} diverged run(s) ({len(div_rows)} checkpoint row(s)) from frontiers "
                     "and fits; listed in the appendix")
        rows = [r for r in rows if r["run_id"] not in div_ids]
    out.mkdir(parents=True, exist_ok=True)
    base = find_base(rows)
    base_pt = point(base) if base else None
    arm = {a: [r for r in rows if r.get("arm") == a and not is_base(r)] for a in ("q0", "q1", "q2", "q3", "q4")}
    dirty = [r for r in arm["q0"] if not is_clean(r)]
    if dirty:
        notes.append(f"excluded {len(dirty)} q0 row(s) with noise, mixing, LoRA or WiSE-FT from Q0")
        arm["q0"] = [r for r in arm["q0"] if is_clean(r)]
    if base is None:
        notes.append("no base row (run_id 'base') yet: matched-accuracy targets and benchmark deltas unavailable")
    finals = {rid: rs[-1] for rid, rs in by_run(rows).items()}
    summary = {"notes": notes}
    sections = []
    SFT = sft_ref(r["method"] for r in rows if not is_base(r))

    # reference constants the write-up may quote
    ref = Section("Reference constants", [Table("Constants quoted in the write-up", ["Item", "Value"], [
        [Const("Paper (RL's Razor), arXiv"), Const(PAPER["arxiv"])],
        [Const("Model"), Const(PAPER["model"])],
        [Const("Paper: R² of forgetting vs forward KL (LLMs)"), Const(f"{PAPER['kl_r2']:.2f}")],
        [Const("Paper: R² of forgetting vs forward KL (ParityMNIST toy, §4 p6)"), Const(f"{PAPER['kl_r2_toy']:.2f}")],
        [Const("Frontier tolerance (acc. points)"), Const("2")],
        [Const("Matched-accuracy target step (acc. points)"), Const("3")],
        [Const("Paper sections cited"), Const("§1 (p2), §3.1 (p4, p5), §4 (p6), §5.1 (p7, p8), §5.2 (p8), §7 (p9, p10)")],
        [Const("Paper sections cited, continued"), Const("Abstract (p1), App. A (p18), App. B.1 (p18), Table 2 (p19), Fig. 9 (p22), Fig. 11 (p23)")],
    ])])
    if base:
        ref.blocks.append(Table("Base model scores (%)", ["Metric", "Score (%)"],
                                [["GSM8K", pct(base["gsm8k_acc"])]]
                                + [[b, pct(base["bench"][b])] for b in BENCH if b in base.get("bench", {})]))
    sections.append(ref)

    # ----- Q0 -----
    q0 = arm["q0"]
    sec = Section("Q0: SFT vs GRPO")
    head_tab, summary["headline"] = headline_table([r for r in arm["q0"] + arm["q1"] if is_clean(r)], base)
    if head_tab:
        sec.blocks.append(head_tab)
        sec.blocks.append("The seeded comparison above is the primary result. The pooled-frontier tables below mix "
                          "settings and seeds: a Pareto frontier is an upper envelope of noisy points and a method "
                          "with more checkpoints gets a more optimistic envelope.")
    curves, targets, fitted, interp, bands, headline = {}, [], {}, {}, {}, None
    if q0:
        q0_methods = _methods(q0)
        curves = {m: _curve([r for r in q0 if r["method"] == m], base_pt) for m in q0_methods}
        if base_pt:
            reach = min(max(r["gsm8k_acc"] for r in q0 if r["method"] == m) for m in q0_methods)
            targets = matched_targets(base_pt[0], reach)
        headline = targets[-1] if targets else None
        fitted = {m: forgetting_at_accuracy(curves[m]["points"], targets) for m in q0_methods}
        interp = {m: frontier_interp(curves[m]["points"], targets) for m in q0_methods}
        bands = {m: seed_band(q0, m, targets, base_pt) if targets else None for m in q0_methods}
        if targets:
            gap = SFT in fitted and "grpo" in fitted
            header = ["Target acc. (%)"]
            for m in q0_methods:
                header += [f"{m} fit (pp)", f"{m} interp. (pp)"]
            header += [f"{SFT} − grpo fit (pp)", f"{SFT} − grpo interp. (pp)"] if gap else []
            body = []
            for i, t in enumerate(targets):
                r = [pct(t)]
                for m in q0_methods:
                    r += [pct(fitted[m][i]), pct(interp[m][i])]
                if gap:
                    for d in (fitted, interp):
                        s, g = d[SFT][i], d["grpo"][i]
                        r.append(pct(s - g) if s is not None and g is not None else NA)
                body.append(r)
            sec.blocks.append(Table("Matched accuracy: forgetting along each pooled frontier (fit, and "
                                    "piecewise-linear interpolation of the observed frontier points)", header, body))
            sec.blocks.append(_fit_table(curves))
            sec.blocks += held_notes(curves, targets, q0)
            sec.blocks.append(Table("Matched accuracy: seed variance at the highest target every seed reaches",
                                    BAND_HEADER, [_band_row(m, q0, bands[m]) for m in q0_methods]))
            sec.blocks.append("A seed band's target is the highest target every seed of the best setting reaches; it "
                              "can be lower than the highest target above and can differ between methods. Near-"
                              "frontier filtering is by accuracy distance, so dominated checkpoints of other seeds "
                              "can enter a fit. The base model is a point of every method's set.")
        else:
            sec.blocks.append("Matched accuracy: no targets yet (needs a base row and accuracy ≥ base + 3 points).")
        sec.blocks.append(bench_table("Q0", q0, base))
        F.frontier_figure(out, "frontier_q0", "Q0: SFT vs GRPO", curves, base_pt, targets, fitted,
                          {m: b for m, b in bands.items() if b})
    else:
        sec.blocks.append(NO_DATA)
        F.placeholder(out, "frontier_q0", "Q0: SFT vs GRPO")
    sections.append(sec)
    summary["q0"] = {"targets": targets, "fit": fitted, "interp": interp, "bands": bands,
                     "rmse": {m: c["fit"].rmse for m, c in curves.items()},
                     "points": {m: c["points"] for m, c in curves.items()}}

    # ----- KL vs forgetting -----
    sec = Section("Forgetting vs forward KL (all checkpoints)")
    kl_rows = [r for r in rows if r.get("kl_fwd") is not None]
    if len(kl_rows) >= 3:
        coef, r2 = quadratic_r2([r["kl_fwd"] for r in kl_rows], [r["forgetting_mean"] for r in kl_rows])
        # the figure shows the arms the write-up reports; the q4 runs (WiSE-FT, LoRA, GKD) stay in the tables and
        # in the pooled fit above, but are not drawn (the write-up does not report them)
        fig_rows = [r for r in kl_rows if r.get("arm") != "q4"]
        if not any(not is_base(r) for r in fig_rows):  # q4-only data: nothing else to draw, so draw it
            fig_rows = kl_rows
        pts = defaultdict(list)
        for r in fig_rows:
            pts[r["method"] if not is_base(r) else "base"].append((r["kl_fwd"], r["forgetting_mean"]))
        pts = {m: pts[m] for m in [*_methods([r for r in fig_rows if not is_base(r)]), "base"] if m in pts}
        sec.blocks.append(Table("Quadratic fit of mean forgetting on forward KL, all non-diverged checkpoints (the "
                                "headline Q0-only fit is in the next section)",
                                ["Checkpoints", "R² (this study)", "R² (paper, LLMs)"],
                                [[str(len(kl_rows)), f"{r2:.2f}", Const(f"{PAPER['kl_r2']:.2f}")]]))
        q0kl = [r for r in kl_rows if is_base(r) or (r.get("arm") == "q0" and is_clean(r))]
        if len(q0kl) >= 3:
            coef0, r2_0 = quadratic_r2([r["kl_fwd"] for r in q0kl], [r["forgetting_mean"] for r in q0kl])
            F.kl_figure(out, "kl_vs_forgetting", pts, coef0, r2_0, fit_label="Q0-only quadratic fit")
        else:  # no Q0 data yet: show the pooled fit, labelled as such
            r2_0 = None
            F.kl_figure(out, "kl_vs_forgetting", pts, coef, r2, fit_label="Pooled quadratic fit")
        summary["kl_r2"], summary["kl_r2_q0"] = r2, r2_0
    else:
        sec.blocks.append(NO_DATA)
        F.placeholder(out, "kl_vs_forgetting", "Forgetting vs forward KL")
    sections.append(sec)
    if len(kl_rows) >= 3:
        ks, summary["kl"] = kl_sections(rows, base, headline, {**(extras or {}), "_out": out})
        sections += ks

    # ----- Q1 -----
    q1 = arm["q1"]
    sec = Section("Q1: SFT on self-generated data")
    if q1:
        q1_methods = _methods(q1)
        q1_curves = {m: _curve([r for r in q1 if r["method"] == m], base_pt) for m in q1_methods}
        t1 = targets
        if not t1 and base_pt:
            t1 = matched_targets(base_pt[0], min(q1_curves[m]["hi"] for m in q1_methods))
        all_curves = {**{m: curves[m] for m in (SFT, "grpo") if m in curves}, **q1_curves}
        items, band_rows, q1_bands = [], [], {}
        if t1:
            body = []
            for m, c in all_curves.items():
                fv, iv = forgetting_at_accuracy(c["points"], t1), frontier_interp(c["points"], t1)
                body += [[m, pct(t), pct(f), pct(i)] for t, f, i in zip(t1, fv, iv)]
            sec.blocks.append(Table("Matched accuracy: forgetting along each pooled frontier",
                                    ["Method", "Target acc. (%)", "Fit (pp)", "Interp. (pp)"], body))
            sec.blocks.append(_fit_table(all_curves))
            for m in (SFT, "grpo"):
                if m in curves:
                    q1_bands[m] = bands.get(m)
                    band_rows.append(_band_row(m, q0, bands.get(m)))
            for m in q1_methods:
                q1_bands[m] = seed_band(q1, m, t1, base_pt)
                band_rows.append(_band_row(m, q1, q1_bands[m]))
            items = [(m, F.style(m)[2], b) for m, b in q1_bands.items() if b]
            sec.blocks.append(Table("Matched accuracy: seed variance at the highest target every seed reaches",
                                    BAND_HEADER, band_rows))
        else:
            sec.blocks.append("Matched accuracy: no targets yet.")
        sec.blocks.append(bench_table("Q1", q1, base))
        F.selfgen_figure(out, "q1_selfgen", {m: curves[m] for m in (SFT, "grpo") if m in curves}, q1_curves,
                         base_pt, items)
        summary["q1"] = {"bands": q1_bands}
    else:
        sec.blocks.append(NO_DATA)
        F.placeholder(out, "q1_selfgen", "Q1: SFT on self-generated data")
    sections.append(sec)

    # ----- per-benchmark at the best settings (Q0 + Q1, seeded) -----
    sec = Section("Per-benchmark change at the best settings (seeded)")
    tables, bb = best_bench([r for r in arm["q0"] + arm["q1"] if is_clean(r)], base)
    if tables:
        sec.blocks += tables
        sec.blocks.append("The six-benchmark mean can hide benchmark-specific changes; p-values are post hoc and "
                          "uncorrected for the six benchmarks, with n = seeds per method.")
    else:
        sec.blocks.append(NO_DATA)
    summary["best_bench"] = bb
    sections.append(sec)

    # ----- Q2 -----
    q2 = arm["q2"]
    sec = Section("Q2: noise robustness")
    if q2:
        series, body, q2sum = {}, [], {}
        for m in _methods(q2):
            mrows = [r for r in q2 if r["method"] == m]
            lrs = {r["lr"] for r in mrows}
            clean = best_setting_rows([r for r in q0 if r["method"] == m and r["lr"] in lrs] or q0, m)
            by_p = defaultdict(list)
            for r in final_rows(mrows) + final_rows(clean):
                by_p[r.get("noise_p", 0.0)].append(r)
            series[m], q2sum[m] = [], {}
            for p, rs in sorted(by_p.items()):
                am, asd = mean_std([r["gsm8k_acc"] for r in rs])
                fm, fsd = mean_std([r["forgetting_mean"] for r in rs])
                kls = [r["kl_fwd"] for r in rs if r.get("kl_fwd") is not None]
                ifv = ([r["bench"]["ifeval"] - base["bench"]["ifeval"] for r in rs]
                       if base and all("ifeval" in r.get("bench", {}) for r in rs) else [])
                im, isd = mean_std(ifv) if ifv else (None, None)
                series[m].append((p, am, asd, fm, fsd))
                q2sum[m][p] = {"n": len(rs), "acc": (am, asd), "forgetting": (fm, fsd), "ifeval": (im, isd),
                               "accs": [r["gsm8k_acc"] for r in rs], "run_ids": sorted(r["run_id"] for r in rs)}
                body.append([m, par(p), str(len(rs)), pct(am), pct(asd), pp2(fm), pp2(fsd), pct(im), pct(isd),
                             kl2(np.mean(kls)) if kls else NA])
        sec.blocks.append("SFT corrupts a fraction p of labels; GRPO flips each reward with probability p. These "
                          "are different kinds of corruption: the comparison is about practical robustness, "
                          "not equal information content. p = 0 is the method's Q0 best setting at the same LR.")
        sec.blocks.append(Table("Final checkpoint vs noise rate, mean and std over seeds (forgetting with 2 "
                                "decimals)", ["Method", "p", "Seeds", "GSM8K (%)", "Std", "Forgetting (pp)", "Std",
                                              "IFEval change (pp)", "Std", "KL"], body))
        if "grpo" in q2sum and SFT in q2sum:
            gap = []
            for p in sorted(set(q2sum["grpo"]) & set(q2sum[SFT])):
                g, s_ = q2sum["grpo"][p], q2sum[SFT][p]
                t, pv = _welch(g["accs"], s_["accs"])
                g0, s0 = q2sum["grpo"].get(0.0), q2sum[SFT].get(0.0)
                gap.append([par(p), pct(g["acc"][0] - base["gsm8k_acc"]) if base else NA,
                            pct(s_["acc"][0] - base["gsm8k_acc"]) if base else NA,
                            pct(g["acc"][0] - g0["acc"][0]) if g0 else NA,
                            pct(s_["acc"][0] - s0["acc"][0]) if s0 else NA,
                            pct(g["acc"][0] - s_["acc"][0]), _fixed(t, 2), _fixed(pv, 3)])
            sec.blocks.append(Table(f"GSM8K at each noise rate: gain over base and GRPO − {SFT} (final checkpoints; "
                                    "Welch t-test over seeds, uncorrected)",
                                    ["p", "GRPO gain (pp)", f"{SFT} gain (pp)", "GRPO change vs p = 0 (pp)",
                                     f"{SFT} change vs p = 0 (pp)", f"GRPO − {SFT} (pp)", "Welch t", "p-value"],
                                    gap))
        sec.blocks.append(bench_table("Q2", q2, base))
        F.noise_figure(out, "q2_noise", series)
        summary["q2"] = q2sum
    else:
        sec.blocks.append(NO_DATA)
        F.placeholder(out, "q2_noise", "Q2: noise robustness")
    sections.append(sec)

    # ----- Q3 -----
    q3 = arm["q3"]
    sec = Section("Q3: general-data mixing into SFT")
    if q3:
        lrs = {r["lr"] for r in q3}
        clean = best_setting_rows([r for r in q0 if r["method"] == SFT and r["lr"] in lrs] or q0, SFT)
        band = bands.get("grpo")
        tg = band.target if band else headline  # compare at the GRPO band's own target
        by_f = defaultdict(list)
        for r in q3 + clean:
            by_f[r.get("mix_f", 0.0)].append(r)
        pts, body, complete, q3sum, seed_body = [], [], {}, {}, []
        for f, rs in sorted(by_f.items()):
            vals, n = _per_run_at(rs, tg, base_pt)
            k = len(vals)
            fm = float(np.mean(vals)) if vals else None
            fsd = float(np.std(vals, ddof=1)) if k == n and k > 1 else None
            fin = final_rows(rs)
            am, asd = mean_std([r["gsm8k_acc"] for r in fin])
            fgm, fgsd = mean_std([r["forgetting_mean"] for r in fin])
            ifv = ([r["bench"]["ifeval"] - base["bench"]["ifeval"] for r in fin]
                   if base and all("ifeval" in r.get("bench", {}) for r in fin) else [])
            im, isd = mean_std(ifv) if ifv else (None, None)
            if len(fin) < 2:  # one seed: no spread
                asd = fgsd = isd = None
            if k == n:
                complete[f] = fm
            q3sum[f] = {"mean": fm, "std": fsd, "k": k, "n": n, "acc": (am, asd), "final_forgetting": (fgm, fgsd),
                        "final_ifeval": (im, isd), "ifeval_vals": ifv}
            pts.append((f, np.nan if fm is None else fm, fsd or 0.0, am, asd or 0.0, len(fin)))
            body.append([par(f), f"{k}/{n}", pct(fm), pct(fsd), _status(k, n), pct(am), pct(asd), pp2(fgm),
                         pp2(fgsd), pct(im), pct(isd)])
            for rid, rr in sorted(by_run(rs).items(), key=lambda kv: kv[1][0].get("seed", 0)):
                v = (forgetting_at_accuracy([point(r) for r in rr] + ([base_pt] if base_pt else []), [tg])[0]
                     if tg is not None else None)
                seed_body.append([par(f), Const(f"s{rr[0].get('seed', 0)}"), pct(max(r["gsm8k_acc"] for r in rr)),
                                  _fixed(None if v is None else 100 * v, 3) if v is not None else "not reached"])
        sec.blocks.append(Table(f"Forgetting at {pct(tg)}% GSM8K (the GRPO band's target), and final checkpoints "
                                "(mean ± sd over seeds; f = 0 is the Q0 best setting)",
                                ["f", "Seeds used", "Forgetting at target (pp)", "Std", "Status", "Final GSM8K (%)",
                                 "Std", "Final forgetting (pp)", "Std", "Final IFEval change (pp)", "Std"], body))
        grpo_final = final_rows(best_setting_rows(q0, "grpo"))
        grpo_acc = mean_std([r["gsm8k_acc"] for r in grpo_final]) if grpo_final else None
        cross = None
        if band is None:
            sec.blocks.append("Crossover: GRPO seed band not available yet.")
        elif band.std is None:
            sec.blocks.append(("Crossover: not computed, the GRPO band has no std (", Const(f"{band.k}/{band.n}"),
                               " seeds reach a matched target)."))
        else:
            cross = mixing_crossover(complete, (band.mean, band.std))
            lim = pct(band.mean - band.std)
            thr = band.mean - band.std
            seed_body.append([Const("GRPO band"), Const(f"{band.k}/{band.n} seeds"), NA,
                              f"{_fixed(100 * band.mean, 3)} ± {_fixed(100 * band.std, 3)}"])
            sec.blocks.append(Table(f"Primary criterion in detail: forgetting at {pct(tg)}% GSM8K per seed (pp, 3 "
                                    "decimals; each run's own checkpoints and the base)",
                                    ["f", "Seed", "Best GSM8K reached (%)", "Forgetting at target (pp)"], seed_body))
            margin = []
            for f, q in sorted(q3sum.items()):
                if q["mean"] is not None:
                    margin.append([par(f), f"{q['k']}/{q['n']}", _fixed(100 * q["mean"], 3), _fixed(100 * thr, 3),
                                   _fixed(100 * (q["mean"] - thr), 3)])
            sec.blocks.append(Table("Primary criterion margin: mean forgetting at the target minus the threshold "
                                    f"(GRPO mean − sd = {_fixed(100 * band.mean, 3)} − {_fixed(100 * band.std, 3)} pp; "
                                    f"GRPO band sd {_fixed(100 * band.std, 2)} pp); negative = misses",
                                    ["f", "Seeds reaching", "Mean (pp)", "Threshold (pp)", "Margin (pp)"], margin))
            summary.setdefault("q3_detail", {})["threshold"] = thr
            if cross is None:
                sec.blocks.append(("Crossover: no fully-seeded mix fraction reaches GRPO mean − std (", lim,
                                   " pp)."))
            else:
                sec.blocks.append(("Crossover: smallest f with forgetting ≥ GRPO mean − std (", lim, " pp) is f = ",
                                   par(cross), "."))
        cross_if = None
        if base and base_pt and band is not None:
            # Pre-registered secondary criterion (R54, fixed before any Q3 result): the same rule on IFEval alone,
            # the one benchmark where teacher SFT forgot on every Q0 seed. Per run: IFEval change vs base at the
            # GRPO band's target, read off the same exponential near-frontier fit (+ base point at 0).
            b_if = base["bench"]["ifeval"]

            def if_at(rows_):
                runs = by_run(rows_)
                vals = [v for v in (forgetting_at_accuracy(
                    [(r["gsm8k_acc"], r["bench"]["ifeval"] - b_if) for r in rs] + [(base_pt[0], 0.0)], [tg])[0]
                    for rs in runs.values()) if v is not None]
                return vals, len(runs)

            gv, gn = if_at(best_setting_rows(q0, "grpo"))
            g_mean = float(np.mean(gv)) if gv else None
            g_std = float(np.std(gv, ddof=1)) if len(gv) == gn and gn > 1 else None
            ibody, icomplete = [], {}
            for f, rs in sorted(by_f.items()):
                vals, n = if_at(rs)
                k = len(vals)
                fm = float(np.mean(vals)) if vals else None
                if k == n and vals:
                    icomplete[f] = fm
                ibody.append([par(f), f"{k}/{n}", pct(fm),
                              pct(float(np.std(vals, ddof=1)) if k == n and k > 1 else None), _status(k, n)])
            ibody.append([Const("GRPO (best LR, seeds)"), f"{len(gv)}/{gn}", pct(g_mean), pct(g_std),
                          _status(len(gv), gn)])
            sec.blocks.append(Table(f"Secondary criterion (pre-registered, R54): IFEval change at {pct(tg)}% GSM8K",
                                    ["f", "Seeds used", "IFEval change (pp)", "Std", "Status"], ibody))
            if g_mean is None or g_std is None:
                sec.blocks.append("IFEval crossover: not computed, the GRPO IFEval band is incomplete.")
            else:
                cross_if = mixing_crossover(icomplete, (g_mean, g_std))
                lim = pct(g_mean - g_std)
                if cross_if is None:
                    sec.blocks.append(("IFEval crossover: no fully-seeded mix fraction reaches GRPO mean − std (",
                                       lim, " pp)."))
                else:
                    sec.blocks.append(("IFEval crossover: smallest f with IFEval change ≥ GRPO mean − std (", lim,
                                       " pp) is f = ", par(cross_if), "."))
        partial = [f for f in by_f if f not in complete]
        if partial:
            sec.blocks.append(("Excluded from the crossover (not every seed reaches the target): f = ",
                               Const(", ".join(par(f) for f in partial)), "."))
        if grpo_acc:
            low = [q[0] for q in pts if q[3] < grpo_acc[0] - grpo_acc[1]]
            if low:
                sec.blocks.append(("Final GSM8K is below GRPO's range (mean − std = ",
                                   pct(grpo_acc[0] - grpo_acc[1]), "%) at f = ",
                                   Const(", ".join(par(f) for f in low)),
                                   " (f = 0, the unmixed Q0 setting, included)." if 0.0 in low else "."))
        f0 = q3sum.get(0.0, {}).get("ifeval_vals") or []
        wb = []
        for f, q in sorted(q3sum.items()):
            if f > 0 and len(q["ifeval_vals"]) > 1 and len(f0) > 1:
                t, pv = _welch(q["ifeval_vals"], f0)
                wb.append([par(f), str(len(q["ifeval_vals"])), pct(float(np.mean(q["ifeval_vals"])) - float(np.mean(f0))),
                           _fixed(t, 2), _fixed(pv, 3)])
        if wb:
            sec.blocks.append(Table("Final IFEval change: each seeded mix vs f = 0 (Welch t-test over seeds, "
                                    "uncorrected)", ["f", "Seeds", "Difference vs f = 0 (pp)", "Welch t", "p-value"],
                                    wb))
        sec.blocks.append(bench_table("Q3", q3, base))
        if_pts = [(f, q["final_ifeval"][0], q["final_ifeval"][1] or 0.0, len(q["ifeval_vals"]))
                  for f, q in sorted(q3sum.items())
                  if q["final_ifeval"][0] is not None]
        g_if = ([r["bench"]["ifeval"] - base["bench"]["ifeval"] for r in grpo_final]
                if base and grpo_final and all("ifeval" in r.get("bench", {}) for r in grpo_final) else [])
        F.mixing_figure(out, "q3_mixing", pts, (band.mean, band.std or 0.0) if band else None, grpo_acc, cross, tg,
                        if_pts or None, mean_std(g_if) if len(g_if) > 1 else None)
        summary["q3"] = {"target": tg, "by_f": q3sum, "crossover": cross, "crossover_ifeval": cross_if}
    else:
        sec.blocks.append(NO_DATA)
        F.placeholder(out, "q3_mixing", "Q3: mixing general data into SFT")
    sections.append(sec)

    # ----- Q4 -----
    q4 = arm["q4"]
    sec = Section("Q4: cheap mitigations")
    if q4:
        wise, wbody = {}, []
        for r in sorted([r for r in q4 if (r.get("wise_alpha") or 0) > 0],
                        key=lambda r: (str(r.get("init_from")), r["wise_alpha"])):
            src = source_run_id(r.get("init_from"))
            path = wise.setdefault(src, (r["method"], []))[1]
            if not path:
                if base_pt:
                    path.append((0.0, *base_pt))
                if src in finals:
                    path.append((1.0, *point(finals[src])))
                    s_ = finals[src]
                    wbody.append([Const(str(src)), Const("1 (source)"), pct(s_["gsm8k_acc"]),
                                  pct(s_["forgetting_mean"]), kl2(s_.get("kl_fwd"))])
                else:
                    notes.append(f"WiSE-FT source run {src!r} not in results")
            path.append((r["wise_alpha"], *point(r)))
            wbody.append([Const(str(src)), par(r["wise_alpha"]), pct(r["gsm8k_acc"]), pct(r["forgetting_mean"]),
                          kl2(r.get("kl_fwd"))])
        if wbody:
            sec.blocks.append(Table("WiSE-FT, 1 seed: θ = (1 − α)·θ₀ + α·θ_ft (α = 1 is the source run's final "
                                    "checkpoint)",
                                    ["Source run", "α", "GSM8K (%)", "Forgetting (pp)", "KL"], wbody))
        lora, lbody = [], []
        lora_rows = [r for r in q4 if (r["method"] == "lora_sft" or r.get("lora")) and not r.get("wise_alpha")]
        for rid, rs in by_run(lora_rows).items():
            lab = Const(label(rs[-1]).removeprefix("lora_sft "))
            lora.append((lab, [point(r) for r in rs]))
            at, _ = _per_run_at(rs, headline, base_pt)
            lbody.append([lab, pct(rs[-1]["gsm8k_acc"]), pct(rs[-1]["forgetting_mean"]),
                          kl2(rs[-1].get("kl_fwd")), pct(at[0] if at else None)])
        if lbody:
            sec.blocks.append(Table(f"LoRA SFT, 1 seed: final checkpoint and forgetting at {pct(headline)}% GSM8K",
                                    ["Setting", "GSM8K (%)", "Forgetting (pp)", "KL", "Forgetting at target (pp)"],
                                    lbody))
        distill, dbody = [], []
        for rid, rs in by_run([r for r in q4 if r["method"] == "distill"]).items():
            src = source_run_id(rs[-1].get("init_from"))
            s = finals.get(src)
            distill.append((point(s) if s else None, [point(r) for r in rs]))
            end = rs[-1]
            dbody.append([Const(str(src)), pct(s["gsm8k_acc"]) if s else NA, pct(s["forgetting_mean"]) if s else NA,
                          pct(end["gsm8k_acc"]), pct(end["forgetting_mean"]),
                          pct(end["forgetting_mean"] - s["forgetting_mean"]) if s else NA,
                          pct(end["gsm8k_acc"] - s["gsm8k_acc"]) if s else NA])
        if dbody:
            sec.blocks.append(Table("On-policy distillation from the base model (recovery), 1 seed",
                                    ["Start run", "Start GSM8K (%)", "Start forgetting (pp)", "End GSM8K (%)",
                                     "End forgetting (pp)", "Forgetting recovered (pp)", "GSM8K change (pp)"],
                                    dbody))
            ckb = []
            for rid, rs in by_run([r for r in q4 if r["method"] == "distill"]).items():
                s_ = finals.get(source_run_id(rs[-1].get("init_from")))
                if s_:
                    ckb.append([Const("start (source final)"), Const("0"), pct(s_["gsm8k_acc"]),
                                pp2(s_["forgetting_mean"]), kl2(s_.get("kl_fwd"))])
                ckb += [[Const(label(r)), Const(str(r["step"])), pct(r["gsm8k_acc"]), pp2(r["forgetting_mean"]),
                         kl2(r.get("kl_fwd"))] for r in rs]
            sec.blocks.append(Table("Distillation checkpoints (forgetting with 2 decimals)",
                                    ["Run", "Step", "GSM8K (%)", "Forgetting (pp)", "KL"], ckb))
        sec.blocks.append(bench_table("Q4", q4, base))
        F.mitigations_figure(out, "q4_mitigations", {m: curves[m] for m in (SFT, "grpo") if m in curves},
                             wise, lora, distill, base_pt)
        summary["q4"] = {"wise": wise, "distill": distill}
    else:
        sec.blocks.append(NO_DATA)
        F.placeholder(out, "q4_mitigations", "Q4: cheap mitigations")
    sections.append(sec)

    # ----- data, cost, runtime (inputs beyond the rows) -----
    es, summary["extra"] = extra_sections(rows, base, headline, extras or {}, all_rows)
    sections += es

    # ----- Appendix: diverged runs -----
    if div_rows:
        sec = Section("Appendix: diverged runs (excluded from frontiers and fits)")
        sec.blocks.append(Table("Checkpoints of diverged runs", ["Run", "Step", "GSM8K (%)", "Forgetting (pp)", "KL"],
                                [[Const(r["run_id"]), Const(str(r["step"])), pct(r["gsm8k_acc"]),
                                  pct(r["forgetting_mean"]), kl2(r.get("kl_fwd"))]
                                 for rs in by_run(div_rows).values() for r in rs]))
        pooled = [r for r in all_rows if r.get("kl_fwd") is not None]
        if len(pooled) >= 3:
            _, r2_all = quadratic_r2([r["kl_fwd"] for r in pooled], [r["forgetting_mean"] for r in pooled])
            sec.blocks.append(Table("Quadratic fit of forgetting on forward KL including diverged runs",
                                    ["Checkpoints", "R² (incl. diverged)"], [[str(len(pooled)), f"{r2_all:.2f}"]]))
            summary["kl_r2_with_diverged"] = r2_all
        summary["diverged"] = sorted(div_ids)
        sections.append(sec)

    smoke = bool(rows) and all(r.get("smoke") for r in rows)
    preamble = ([f"**{SMOKE_BANNER}**"] if smoke else []) + [
        "Formats: accuracy and benchmark scores in percent (1 decimal); forgetting and benchmark changes in "
        "percentage points, checkpoint − base (1 decimal, negative = forgetting); forward KL in nats (2 decimals); "
        "R² 2 decimals. Matched-accuracy targets are base + 3, + 6, ... accuracy points up to the largest "
        "accuracy every compared method reaches. Seed bands use the highest target every seed reaches; "
        "a band marked partial is missing seeds and has no std.",
        *[f"Note: {n}." for n in notes],
    ]
    for sec in sections:  # tables the write-up's body repeats cell for cell, or page-long: tables.md only
        sec.blocks = [dataclasses.replace(b, tex=False) if isinstance(b, Table) and b.title.startswith(MD_ONLY)
                      else b for b in sec.blocks]
    (out / "tables.md").write_text(render_md(sections, preamble))
    (out / "tables.tex").write_text(render_tex(sections, preamble))
    results, consts = collect_numbers(sections, preamble)
    (out / "numbers.json").write_text(json.dumps(
        {"smoke": smoke, "results": sorted(results), "constants": sorted(consts - results)}, indent=1))
    summary["smoke"] = smoke
    return summary


def _backed(n: str, known: set[str]) -> bool:
    """n is in `known`, or n is the unsigned magnitude of a negative entry. A sign is never added."""
    return n in known or (not n.startswith("-") and "-" + n in known)


def check_writeup(path: Path, tables_dir: Path) -> int:
    nf = tables_dir / "numbers.json"
    if not nf.exists():
        print(f"no numbers.json in {tables_dir}; run analyze.py --results first", file=sys.stderr)
        return 2
    data = json.loads(nf.read_text())
    if data.get("smoke"):
        print(f"tables in {tables_dir} were built from smoke data; refusing to back a write-up with them")
        return 3
    results, consts = set(data["results"]), set(data["constants"])
    found = list(dict.fromkeys(NUMBER.findall(path.read_text())))
    missing = [n for n in found if not _backed(n, results) and not _backed(n, consts)]
    const_only = [n for n in found if not _backed(n, results) and _backed(n, consts)]
    if const_only:
        print(f"{len(const_only)} number(s) matched constants only (hyperparameters/reference values), check them:")
        for n in const_only:
            print(f"  matched constant: {n}")
    if missing:
        print(f"{len(missing)} number(s) in {path} not found in {tables_dir} tables:")
        for n in missing:
            print(f"  {n}")
        return 1
    print(f"all numbers in {path} are backed by the tables")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, help="results.jsonl (from rlrazor.results.export_jsonl)")
    ap.add_argument("--out", type=Path, default=Path("figures"),
                    help="figures and tables directory; --check-writeup reads tables from here (default: figures)")
    ap.add_argument("--from-logs", type=Path, metavar="DIR",
                    help="summarise DIR/launch-*.log and DIR/logs/*.log into the --results directory first")
    ap.add_argument("--check-writeup", type=Path, metavar="PATH",
                    help="write-up (.tex, .md, any text) to check; with --results, tables are regenerated first")
    args = ap.parse_args(argv)
    if not args.results and not args.check_writeup:
        ap.error("--results or --check-writeup is required")
    if args.results and args.from_logs:
        for p in logs_to_results(args.from_logs, args.results.parent):
            print(f"wrote {p}")
    if args.results:
        rows = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
        extras = load_extras(args.results.parent, Path(__file__).resolve().parent.parent)
        for n in analyze(rows, args.out, extras)["notes"]:
            print(f"note: {n}")
        print(f"wrote figures and tables to {args.out}")
    if args.check_writeup:
        return check_writeup(args.check_writeup, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
