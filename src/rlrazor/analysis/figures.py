"""Matplotlib figures for the write-up (Agg backend, light background, PNG at 200 dpi plus a PDF copy).

Every method keeps one colour and one marker in every figure (METHOD_STYLE, the dataviz reference
palette in its fixed slot order). Scatter panels show at most three methods at once; larger
comparisons are faceted. Inputs are fractions; axes show percentage points.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DPI = 200
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

# method -> (colour, marker, label); slot order of the reference categorical palette
METHOD_STYLE = {
    "sft": ("#2a78d6", "o", "SFT (gold)"),
    "teacher_sft": ("#2a78d6", "o", "SFT (teacher)"),  # the study's SFT arm (R42/R43); same slot as SFT (gold)
    "grpo": ("#eb6834", "s", "GRPO"),
    "rft_offline": ("#1baf7a", "^", "Offline RFT"),
    "rft_iter": ("#eda100", "D", "Iterative RFT"),
    "self_rewrite": ("#e87ba4", "v", "Self-rewrite SFT"),
    "rl_distill_sft": ("#008300", "P", "SFT on GRPO outputs"),
    "lora_sft": ("#6250d6", "X", "LoRA SFT"),
    "distill": ("#e34948", "*", "On-policy distillation"),
    "base": (INK, "o", "Base model"),
}
FALLBACK_STYLE = (MUTED, "o", None)


def style(method: str) -> tuple[str, str, str]:
    c, m, label = METHOD_STYLE.get(method, FALLBACK_STYLE)
    return c, m, label or method


def _rc():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2, "lines.markersize": 6,
    })


def _fig(ncols=1, nrows=1, w=4.2, h=3.4):
    _rc()
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)
    return fig, axes


def save(fig, out: Path, name: str) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=DPI)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def placeholder(out: Path, name: str, title: str, message: str = "No data yet") -> None:
    """A labelled empty figure, so LaTeX includes still compile while the grid is incomplete."""
    fig, axes = _fig()
    ax = axes[0, 0]
    ax.set_axis_off()
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", color=INK2, fontsize=11, transform=ax.transAxes)
    save(fig, out, name)


def _pct(v):
    return 100 * np.asarray(v, float)


def _acc_forg_axes(ax):
    ax.set_xlabel("GSM8K accuracy (%)")
    ax.set_ylabel("Mean forgetting (pp, ckpt − base)")
    ax.axhline(0, color=AXIS, linewidth=1, zorder=0)


def _frontier(ax, method, curve, label=True, thin=False):
    """curve: dict(points, near, fit, lo, hi); draws all points faint, near-frontier points solid, fit line."""
    c, m, lab = style(method)
    pts = np.array(curve["points"])
    near = np.array(curve["near"])
    if not thin:
        ax.scatter(_pct(pts[:, 0]), _pct(pts[:, 1]), s=18, color=c, marker=m, alpha=0.25, linewidths=0)
        ax.scatter(_pct(near[:, 0]), _pct(near[:, 1]), s=30, color=c, marker=m, edgecolors=SURFACE,
                   linewidths=1, label=lab if label else None, zorder=3)
    xs = np.linspace(curve["lo"], curve["hi"], 100)
    ax.plot(_pct(xs), _pct([curve["fit"](x) for x in xs]), color=c, linewidth=1.2 if thin else 2,
            alpha=0.8 if thin else 1, linestyle="--" if thin else "-",
            label=f"{lab} frontier fit" if thin and label else None)


def zoom_limits(curves: dict, base_pt, pad_x: float = 0.01, pad_y: float = 0.003):
    """Axis limits (fractions) around the near-frontier points and the base model, so the frontiers being
    compared are visible; returns (xlim, ylim, n_points_outside). Points far off the frontier (e.g. high-LR
    checkpoints that forget 10+ pp) are left outside and counted."""
    near = [p for c in curves.values() for p in c["near"]] + ([tuple(base_pt)] if base_pt else [])
    if not near:
        return None, None, 0
    xs, ys = [a for a, _ in near], [f for _, f in near]
    xlim = (min(xs) - pad_x, max(xs) + pad_x)
    ylim = (min(ys) - pad_y, max(ys) + pad_y)
    outside = sum(1 for c in curves.values() for a, f in c["points"]
                  if not (xlim[0] <= a <= xlim[1] and ylim[0] <= f <= ylim[1]))
    return xlim, ylim, outside


def frontier_figure(out, name, title, curves: dict, base_pt, targets, matched: dict, bands: dict):
    """Left: points, near-frontier points and exponential fits. Right: forgetting at matched accuracy
    along each pooled frontier, with the best setting's SeedBand (mean ± std at its target)."""
    fig, axes = _fig(ncols=2, h=3.9)
    ax = axes[0, 0]
    for method, curve in curves.items():
        _frontier(ax, method, curve)
    if base_pt:
        ax.scatter(*_pct(base_pt), s=50, color=INK, marker="o", zorder=4, label="Base model")
    _acc_forg_axes(ax)
    xlim, ylim, outside = zoom_limits(curves, base_pt)
    if xlim and outside:  # zoom to the frontiers; say how many checkpoints fall outside the view
        ax.set_xlim(*_pct(xlim))
        ax.set_ylim(*_pct(ylim))
        ax.text(0.98, 0.98, f"{outside} checkpoint(s) outside this view\n(far from the frontier; see tables)",
                transform=ax.transAxes, color=INK2, fontsize=7, va="top", ha="right")
    ax.set_title("Near-frontier checkpoints and fits")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=3, fontsize=7)
    ax = axes[0, 1]
    if targets:
        for method, vals in matched.items():
            c, m, lab = style(method)
            xs = [t for t, v in zip(targets, vals) if v is not None]
            ys = [v for v in vals if v is not None]
            ax.plot(_pct(xs), _pct(ys), color=c, marker=m, label=f"{lab} (pooled frontier)")
        for i, (method, band) in enumerate(bands.items()):  # bands drawn beside the pooled points, not on them
            c, m, lab = style(method)
            ax.errorbar(100 * band.target + 0.3 * (i + 1), 100 * band.mean, yerr=100 * (band.std or 0.0), color=c, marker=m,
                        markersize=8, capsize=4, markerfacecolor=SURFACE, linewidth=1.5, zorder=4,
                        label=f"{lab} best setting, {_band_label(band)}")
        ax.set_xlim(100 * min(targets) - 1.5, 100 * max(targets) + 1.5)  # fixed range: the band dodge is visual only
        ax.set_xticks([100 * t for t in targets], [f"{100 * t:.1f}" for t in targets])
        ax.set_xlabel("Target GSM8K accuracy (%)")
        ax.set_ylabel("Forgetting at matched accuracy (pp)")
        ax.axhline(0, color=AXIS, linewidth=1, zorder=0)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=1, fontsize=7)  # off the data
    else:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No matched-accuracy targets yet", ha="center", va="center", color=INK2,
                transform=ax.transAxes)
    ax.set_title("Forgetting at matched accuracy")
    fig.suptitle(title, color=INK)
    save(fig, out, name)


def kl_figure(out, name, pts_by_method: dict, coef, r2, fit_label: str = "Quadratic fit"):
    """Forgetting vs forward KL over all checkpoints, with a quadratic fit and its R² (the fit's population is
    named by `fit_label`; analyze.py passes the Q0-only fit, the population of the paper's 0.71)."""
    fig, axes = _fig(w=5.2, h=3.8)
    ax = axes[0, 0]
    xs_all = []
    for method, pts in pts_by_method.items():
        c, m, lab = style(method)
        p = np.array(pts)
        xs_all.extend(p[:, 0])
        ax.scatter(p[:, 0], _pct(p[:, 1]), s=22, color=c, marker=m, edgecolors=SURFACE, linewidths=0.8,
                   label=lab, alpha=0.9)
    if coef is not None:
        xs = np.linspace(0, max(xs_all), 100)
        ax.plot(xs, _pct(np.polyval(coef, xs)), color=INK, linewidth=1.5,
                label=f"{fit_label}, R² = {r2:.2f}")
    ax.set_xlabel("Forward KL on GSM8K, E[KL(π₀‖π)] (nats)")
    ax.set_ylabel("Mean forgetting (pp, ckpt − base)")
    ax.axhline(0, color=AXIS, linewidth=1, zorder=0)
    ax.set_title("Forgetting vs forward KL, non-diverged checkpoints")
    ax.legend(loc="lower right", ncols=1)
    save(fig, out, name)


def _band_label(band) -> str:
    """Partial bands (not every seed reaches the target) are labelled as such and carry no error bar."""
    if band.partial:
        return f"partial: {band.k}/{band.n} seeds, no std"
    return f"seed mean ± std ({band.n} seeds)" if band.std is not None else f"{band.n} seed"


def _dot_panel(ax, items: list, xlabel):
    """items: (method, label, SeedBand) -> horizontal dot plot with seed error bars; partial bands hollow."""
    for i, (method, label, band) in enumerate(items):
        c, m, _ = style(method)
        ax.errorbar(100 * band.mean, i, xerr=100 * (band.std or 0.0), color=c, marker=m, markersize=7, capsize=3,
                    linewidth=1.5, markerfacecolor=SURFACE if band.partial else c)
    targets = {round(it[2].target, 6) for it in items}
    one = len(targets) == 1  # common target: name it once in the axis label, not on every tick
    ax.set_yticks(range(len(items)), [it[1] + ("" if one else f" @ {100 * it[2].target:.1f}%")
                                      + (f" ({it[2].k}/{it[2].n})" if it[2].partial else "") for it in items])
    if one:
        xlabel = f"{xlabel} at {100 * targets.pop():.1f}% GSM8K"
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.axvline(0, color=AXIS, linewidth=1, zorder=0)
    ax.grid(axis="y", visible=False)


def selfgen_figure(out, name, ref_curves: dict, curves: dict, base_pt, band_items):
    """One panel per Q1 method against the SFT/GRPO reference fits, two columns, sized for a full text width
    (about 6.5 in) so every font prints at 7 pt or more. The seed-band dot plot fills a free grid slot only when
    the method count is odd; the band values are in the tables either way."""
    n_methods = len(curves)
    with_bands = bool(band_items) and n_methods % 2 == 1
    n = max(1, n_methods + with_bands)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = _fig(ncols=ncols, nrows=nrows, w=6.5 / ncols, h=2.6)
    flat = axes.ravel()
    for ax, (method, curve) in zip(flat, curves.items()):
        for ref, rc in ref_curves.items():
            _frontier(ax, ref, rc, thin=True)
        _frontier(ax, method, curve)
        if base_pt:
            ax.scatter(*_pct(base_pt), s=30, color=INK, zorder=4)
        _acc_forg_axes(ax)
        ax.set_title(style(method)[2])
        ax.legend(loc="lower left", fontsize=7.5)
    if with_bands:
        ax = flat[n_methods]
        _dot_panel(ax, band_items, "Forgetting (pp)")
        ax.set_title("Seed mean ± std, best setting; hollow = partial")
    if not curves:
        flat[0].set_axis_off()
        flat[0].text(0.5, 0.5, "No Q1 runs yet", ha="center", va="center", color=INK2, transform=flat[0].transAxes)
    for ax in flat[n:]:
        ax.set_axis_off()
    save(fig, out, name)


def noise_figure(out, name, series: dict):
    """series: method -> list of (p, acc_mean, acc_std, forg_mean, forg_std), final checkpoints."""
    fig, axes = _fig(ncols=2)
    for method, pts in series.items():
        c, m, lab = style(method)
        p = np.array(sorted(pts))
        axes[0, 0].errorbar(p[:, 0], _pct(p[:, 1]), yerr=_pct(p[:, 2]), color=c, marker=m, capsize=3, label=lab)
        axes[0, 1].errorbar(p[:, 0], _pct(p[:, 3]), yerr=_pct(p[:, 4]), color=c, marker=m, capsize=3, label=lab)
    axes[0, 0].set_ylabel("Final GSM8K accuracy (%)")
    axes[0, 1].set_ylabel("Final mean forgetting (pp)")
    axes[0, 1].axhline(0, color=AXIS, linewidth=1, zorder=0)
    for ax in axes[0]:
        ax.set_xlabel("Noise rate p (SFT: label, GRPO: reward flip)")
        ax.legend(loc="lower left")
    axes[0, 0].set_title("Accuracy vs noise")
    axes[0, 1].set_title("Forgetting vs noise")
    fig.suptitle("Q2: label noise (SFT) vs reward noise (GRPO), seed mean ± std", color=INK)
    save(fig, out, name)


def mixing_figure(out, name, pts, grpo_band, grpo_acc, crossover, target, ifeval=None, grpo_ifeval=None):
    """pts: list of (f, forg_mean, forg_std, acc_mean, acc_std[, n_seeds]); grpo_band / grpo_acc / grpo_ifeval:
    (mean, std) or None. ifeval: optional list of (f, final IFEval change mean, std[, n_seeds]) for a third panel.
    Fractions with several seeds are joined by a line; single-seed fractions are hollow, unjoined markers."""
    fig, axes = _fig(ncols=3 if ifeval else 2, w=2.9 if ifeval else 4.2, h=3.6)

    def series(ax, x, y, e, n, label):
        many = n > 1
        ax.errorbar(x[many], y[many], yerr=e[many], color=c, marker=m, capsize=3, label=f"{label} (3 seeds)")
        if (~many).any():
            ax.plot(x[~many], y[~many], linestyle="none", marker=m, markerfacecolor="none", markeredgecolor=c,
                    markersize=7, label=f"{label} (1 seed)")
    c, m, lab = style("teacher_sft")
    gc = style("grpo")[0]
    p = np.array([(*t, 1)[:6] if len(t) == 5 else t for t in sorted(pts, key=lambda t: t[0])], dtype=float)
    ok = ~np.isnan(p[:, 1])
    ax = axes[0, 0]
    if grpo_band:
        ax.axhspan(100 * (grpo_band[0] - grpo_band[1]), 100 * (grpo_band[0] + grpo_band[1]), color=gc, alpha=0.18,
                   linewidth=0, label="GRPO seed mean ± std")
    series(ax, p[ok, 0], _pct(p[ok, 1]), _pct(p[ok, 2]), p[ok, 5], f"{lab} + general")
    if crossover is not None:
        ax.axvline(crossover, color=INK2, linewidth=1, linestyle=":", label=f"Crossover f = {crossover:g}")
    ax.set_ylabel(f"Forgetting at {100 * target:.1f}% GSM8K (pp)" if target else "Forgetting at matched acc. (pp)")
    ax.set_title("Forgetting at matched accuracy\n(mean of the seeds that reach it)", fontsize=9)
    ax = axes[0, 1]
    if grpo_acc:
        ax.axhspan(100 * (grpo_acc[0] - grpo_acc[1]), 100 * (grpo_acc[0] + grpo_acc[1]), color=gc, alpha=0.18,
                   linewidth=0, label="GRPO final acc. mean ± std")
    series(ax, p[:, 0], _pct(p[:, 3]), _pct(p[:, 4]), p[:, 5], lab)
    ax.set_ylabel("Final GSM8K accuracy (%)")
    ax.set_title("Accuracy vs mix fraction")
    if ifeval:
        ax = axes[0, 2]
        q = np.array([(*t, 1)[:4] if len(t) == 3 else t for t in sorted(ifeval)], dtype=float)
        if grpo_ifeval:
            ax.axhspan(100 * (grpo_ifeval[0] - grpo_ifeval[1]), 100 * (grpo_ifeval[0] + grpo_ifeval[1]), color=gc,
                       alpha=0.18, linewidth=0, label="GRPO final mean ± std")
        ax.axhline(0, color=INK2, linewidth=0.8)
        series(ax, q[:, 0], _pct(q[:, 1]), _pct(q[:, 2]), q[:, 3], lab)
        ax.set_ylabel("Final IFEval change vs base (pp)")
        ax.set_title("IFEval vs mix fraction")
    for ax in axes[0]:
        ax.set_xscale("symlog", linthresh=0.02)
        ax.set_xlim(-0.002, p[:, 0].max() * 1.15)
        ax.set_xticks(p[:, 0], [f"{f:g}" for f in p[:, 0]])
        ax.minorticks_off()
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_xlabel("General-data fraction f")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, fontsize=8, frameon=False)
    fig.suptitle("Q3: mixing general data into SFT", color=INK)
    save(fig, out, name)


def mitigations_figure(out, name, ref_curves: dict, wise: dict, lora: list, distill: list, base_pt):
    """Three facets against the SFT/GRPO reference fits.
    wise: source label -> (method, [(alpha, acc, forg)]) including α=0 (base) and α=1 (source final).
    lora: [(label, [(acc, forg)])]. distill: [((src_acc, src_forg) | None, [(acc, forg)])]."""
    fig, axes = _fig(ncols=3, w=3.0, h=4.0)
    panels = [("WiSE-FT interpolation", wise), ("LoRA SFT", lora), ("On-policy distillation from base", distill)]
    # each panel is zoomed to its own data (plus base); reference fits are drawn only inside that range
    spans = [[q[1] for _, path in wise.values() for q in path],
             [a for _, pts in lora for a, _ in pts], [a for src, pts in distill for a, _ in pts + ([src] if src else [])]]
    for ax, (title, data), xs in zip(axes[0], panels, spans):
        lo = min(xs + ([base_pt[0]] if base_pt else [])) - 0.01 if xs else None
        hi = max(xs + ([base_pt[0]] if base_pt else [])) + 0.01 if xs else None
        for ref, rc in ref_curves.items():
            rc = dict(rc, lo=max(rc["lo"], lo), hi=min(rc["hi"], hi)) if xs else rc
            if rc["lo"] < rc["hi"]:
                _frontier(ax, ref, rc, thin=True)
        if xs:
            ax.set_xlim(100 * lo, 100 * hi)
        if base_pt:
            ax.scatter(*_pct(base_pt), s=40, color=INK, zorder=4, label="Base")
        _acc_forg_axes(ax)
        ax.set_title(title)
        if not data:
            ax.text(0.5, 0.5, "No data yet", ha="center", va="center", color=INK2, transform=ax.transAxes)
    ax = axes[0, 0]
    for k, (src, (method, path)) in enumerate(wise.items()):
        c, m, lab = style(method)
        path = sorted(path)
        ax.plot(_pct([q[1] for q in path]), _pct([q[2] for q in path]), color=c, marker=m, linewidth=1.5,
                markeredgecolor=SURFACE, label=f"{lab} source, s0")
        for a, x, y in path:
            if 0 < a < 1:
                ax.annotate(f"α={a:g}", (100 * x, 100 * y), textcoords="offset points", xytext=(5, 5 if k % 2 else -11), fontsize=8,
                            color=INK2)
    ax = axes[0, 1]
    c, m, lab = style("lora_sft")
    for label, pts in lora:
        p = np.array(sorted(pts))
        ax.plot(_pct(p[:, 0]), _pct(p[:, 1]), color=c, marker=m, linewidth=1, markeredgecolor=SURFACE,
                label=label)
    ax = axes[0, 2]
    c, m, lab = style("distill")
    for i, (src, pts) in enumerate(distill):
        p = np.array(pts)
        ax.plot(_pct(p[:, 0]), _pct(p[:, 1]), color=c, marker=m, markersize=9, linewidth=1,
                markeredgecolor=SURFACE, label=lab if i == 0 else None)
        if src:
            ax.scatter(*_pct(src), s=40, color=style("sft")[0], marker="o", zorder=4,
                       label="Start (SFT final)" if i == 0 else None)
            ax.annotate("", xy=_pct(p[-1]), xytext=_pct(src), arrowprops=dict(arrowstyle="->", color=INK2))
    for ax in axes[0]:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, fontsize=8, frameon=False)
    fig.suptitle("Q4: cheap mitigations (1 seed each)", color=INK)
    save(fig, out, name)


def kl_detail_figure(out, name, acc_kl: dict, per_token: dict, base_acc=None):
    """KL analysis. Left: forward KL (nats/seq) vs GSM8K accuracy per checkpoint, one series per method,
    so KL is compared at equal accuracy rather than at the final checkpoint. Right: distribution of per-sample
    per-token log-ratio at the final checkpoints of each method's best setting (seeds pooled), box = quartiles,
    whiskers = 5th/95th percentiles."""
    fig, axes = _fig(ncols=2, w=4.0, h=3.4)
    ax = axes[0, 0]
    for method, pts in acc_kl.items():
        c, m, lab = style(method)
        p = np.array(pts)
        ax.scatter(_pct(p[:, 0]), p[:, 1], s=26, color=c, marker=m, edgecolors=SURFACE, linewidths=0.8, label=lab)
        if p.shape[1] > 2 and method == "grpo":  # direct-label each GRPO LR at its highest-KL point
            for lr in sorted(set(p[:, 2])):
                q = p[p[:, 2] == lr]
                top = q[np.argmax(q[:, 1])]
                ax.annotate(f"LR {lr:.0e}".replace("e-0", "e-"), (100 * top[0], top[1]), xytext=(6, -3),
                            textcoords="offset points", color=INK2, fontsize=8)
    if base_acc is not None:
        ax.axvline(100 * base_acc, color=AXIS, linewidth=1, zorder=0)
        ax.annotate("base", (100 * base_acc, 1), xycoords=ax.get_xaxis_transform(), xytext=(3, -10),
                    textcoords="offset points", color=INK2, fontsize=8)
    ax.set_xlabel("GSM8K accuracy (%)")
    ax.set_ylabel("Forward KL (nats per sequence)")
    ax.set_title("KL vs accuracy, low-LR checkpoints")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=2)  # below the axes, off the data
    ax = axes[0, 1]
    labels, data, colors = [], [], []
    for method, vals in per_token.items():
        c, _, lab = style(method)
        labels.append(lab.replace(" ", "\n", 1) if len(lab) > 6 else lab)
        data.append(vals)
        colors.append(c)
    bp = ax.boxplot(data, whis=(5, 95), showfliers=False, patch_artist=True, widths=0.5,
                    medianprops={"color": INK, "linewidth": 1.5})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
        patch.set_edgecolor(c)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_ylabel("Per-sample KL per token (nats)")
    ax.set_title("Per-sample KL, final ckpts, best LR")
    save(fig, out, name)
