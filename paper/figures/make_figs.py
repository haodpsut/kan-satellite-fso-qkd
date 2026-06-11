"""Generate paper-quality result figures (PDF) from the result CSVs in the
code/ subtree. Run from anywhere; figures land next to this script.

  python paper/figures/make_figs.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch
import numpy as np

HERE = Path(__file__).resolve().parent
# paper/ lives at <repo>/paper, results at <repo>/results (repo root = code/)
CODE_RESULTS = HERE.parents[1] / "results"

# ------- Matplotlib style for publication figures -------
plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "legend.fontsize": 8.5,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "0.6",
    "legend.borderpad": 0.3,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
})

# Consistent colour palette: Okabe-Ito colourblind-safe, print-safe.
# Series are ALSO separated by linestyle/marker so colour is never the only cue.
_OI_BLUE   = "#0072B2"   # blue
_OI_ORANGE = "#E69F00"   # orange
_OI_GREEN  = "#009E73"   # bluish green
_OI_VERM   = "#D55E00"   # vermillion
_OI_PURPLE = "#CC79A7"   # reddish purple
_OI_SKY    = "#56B4E9"   # sky blue
_OI_GREY   = "#7F7F7F"   # neutral grey

C_STATIC   = _OI_GREY
C_ADAPTIVE = _OI_ORANGE
C_ANALYTIC = _OI_BLUE
C_TLE      = _OI_GREEN
C_THIRD    = _OI_PURPLE
C_KAN      = _OI_BLUE
C_MLP      = _OI_ORANGE
C_KNN      = _OI_GREEN
C_LIN      = _OI_VERM


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out = {}
    for k in rows[0].keys():
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except (ValueError, TypeError):
                vals.append(np.nan)
        out[k] = np.array(vals)
    return out


# ============================================================
# Figure 1: single-link adaptive vs static over a pass
# ============================================================
def fig_zenith_pass():
    d = read_csv(CODE_RESULTS / "optimize_pass.csv")
    if not d:
        return

    # Focus on a single representative LEO pass (the first satellite track),
    # so the per-step controller decisions are legible at single-column width.
    sat = d.get("sat_id")
    m = (sat == sat[0]) if sat is not None else np.ones_like(d["t"], dtype=bool)
    t0 = d["t"][m][0]
    t = d["t"][m] - t0                       # pass-relative time [s]
    zen = d["zenith_deg"][m]
    mu_s, beta_s = d["mu_s"][m], d["beta_s"][m]
    mu_o, beta_o = d["mu_opt"][m], d["beta_opt"][m]
    sec_o = d["secure_opt"][m] if "secure_opt" in d else np.zeros_like(t)

    # Secure-key window: contiguous span where the adaptive link yields key
    # (Eve-binding region near zenith). Shade it as context across all panels.
    sec_mask = sec_o > 0.5
    if sec_mask.any():
        idx = np.where(sec_mask)[0]
        win = (t[idx[0]], t[idx[-1]])
    else:
        win = None

    fig, ax = plt.subplots(3, 1, figsize=(3.45, 3.7), sharex=True,
                           gridspec_kw={"hspace": 0.14,
                                        "height_ratios": [0.85, 1.0, 1.0]})

    def _shade(a):
        if win is not None:
            a.axvspan(win[0], win[1], color=_OI_GREEN, alpha=0.10,
                      lw=0, zorder=0)

    def _tag(a, s):
        a.text(0.985, 0.90, s, transform=a.transAxes, fontsize=8,
               va="top", ha="right", color="0.45")

    # ---- (a) zenith trajectory ----
    a = ax[0]
    _shade(a)
    a.plot(t, zen, color=C_ANALYTIC, lw=1.5, zorder=3)
    a.axhline(60, ls=":", color="0.55", lw=0.8, zorder=1)
    a.set_ylabel("Zenith\n[deg]", linespacing=1.0)
    a.set_ylim(-3, 66)
    a.set_yticks([0, 30, 60])
    _tag(a, "(a)")

    # ---- (b) mu: pulse intensity ----
    a = ax[1]
    _shade(a)
    a.fill_between(t, 0, mu_o, step="post", color=C_ADAPTIVE,
                   alpha=0.07, lw=0, zorder=1)
    a.step(t, mu_o, where="post", color=C_ADAPTIVE, lw=1.4, zorder=3)
    a.step(t, mu_s, where="post", color=C_STATIC, lw=1.1, ls=(0, (5, 2)),
           zorder=2)
    a.set_ylabel(r"$\mu$")
    a.set_ylim(0, 1.18)
    a.set_yticks([0, 0.5, 1.0])
    _tag(a, "(b)")

    # ---- (c) beta: decoy/signal ratio ----
    a = ax[2]
    _shade(a)
    ymax = max(beta_o.max(), beta_s[0]) * 1.15
    a.fill_between(t, 0, beta_o, step="post", color=C_ADAPTIVE,
                   alpha=0.13, lw=0, zorder=1)
    a.step(t, beta_o, where="post", color=C_ADAPTIVE, lw=1.4, zorder=3)
    a.step(t, beta_s, where="post", color=C_STATIC, lw=1.1, ls=(0, (5, 2)),
           zorder=2)
    a.set_ylabel(r"$\beta$")
    a.set_xlabel("pass time [s]")
    a.set_ylim(0, ymax)
    a.set_yticks([0, 1, 2, 3])
    a.set_xlim(t[0], t[-1])
    _tag(a, "(c)")

    fig.align_ylabels(ax)

    # ---- single shared legend on top ----
    handles = [
        plt.Line2D([0], [0], color=C_ADAPTIVE, lw=1.4,
                   label=r"adaptive $(\mu^\star,\beta^\star)$"),
        plt.Line2D([0], [0], color=C_STATIC, lw=1.1, ls=(0, (5, 2)),
                   label="static"),
        plt.Rectangle((0, 0), 1, 1, fc=_OI_GREEN, alpha=0.18, ec="none",
                      label="secure window"),
    ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.965), ncol=3,
               columnspacing=1.1, handlelength=1.6,
               fontsize=8, frameon=False)

    fig.savefig(HERE / "fig_zenith_pass.pdf")
    plt.close(fig)


# ============================================================
# Figure 2: cluster pass — analytic vs TLE-Walker n=500 long
# ============================================================
def fig_cluster_compare():
    da = read_csv(CODE_RESULTS / "cluster_pass_analytic.csv")
    dt_long  = CODE_RESULTS / "cluster_pass_tle_walker_long.csv"
    dt_short = CODE_RESULTS / "cluster_pass_tle_walker.csv"
    dt = read_csv(dt_long) if dt_long.exists() else read_csv(dt_short)
    if not da or not dt:
        return

    fig, axes = plt.subplots(
        2, 2, figsize=(7.16, 4.25),
        gridspec_kw={"height_ratios": [0.82, 1.18],
                     "hspace": 0.34, "wspace": 0.14})

    panels = [
        (0, da, C_ANALYTIC,
         "(a)  Analytic constellation",
         r"$n{=}10$ overhead, $T{=}50$ min",
         r"$1.64\times$"),
        (1, dt, C_TLE,
         "(b)  TLE-Walker shell, $n{=}500$",
         r"PTIT/Hanoi, Starlink-density, $T{=}90$ min",
         r"$2.10\times$"),
    ]

    EL_MIN = 60.0  # min serving elevation marker (60-deg horizon)

    # ---------- TOP ROW: zenith trajectory ----------
    for col, d, color, title, sub, gain in panels:
        ax = axes[0, col]
        t = d["t"] / 60.0  # minutes
        zen = d["zenith"]
        ax.axhspan(0, EL_MIN, color=color, alpha=0.06, zorder=0)
        ax.plot(t, zen, color=color, lw=1.6, zorder=3,
                solid_capstyle="round")
        ax.scatter(t, zen, s=8, color=color, zorder=4,
                   edgecolor="white", linewidth=0.3)
        ax.axhline(EL_MIN, ls=(0, (4, 3)), color="0.45", lw=0.9, zorder=2)
        ax.text(t.max(), EL_MIN + 1.4, r"$60^{\circ}$ horizon",
                ha="right", va="bottom", fontsize=7.0, color="0.4")
        ax.set_ylim(0, 70)
        ax.set_yticks([0, 30, 60])
        ax.set_xlim(t.min() - 0.5, t.max() + 0.5)
        ax.set_title(title + "\n", fontsize=8.7, pad=12,
                     fontweight="semibold", linespacing=1.0)
        ax.text(0.5, 1.045, sub, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=7.3, color="0.4")
        ax.tick_params(axis="x", labelbottom=False)

    axes[0, 0].set_ylabel("Zenith angle [deg]")
    axes[0, 1].tick_params(axis="y", labelleft=False)

    # ---------- BOTTOM ROW: secure pairs / step ----------
    YMAX = 3.7
    for col, d, color, title, sub, gain in panels:
        ax = axes[1, col]
        t = d["t"] / 60.0
        span = t.max() - t.min()
        bw = span * 0.0125            # visible but non-overlapping paired bars
        ss, sa = d["sec_s"], d["sec_a"]

        ax.bar(t - bw/2, ss, width=bw, color=C_STATIC,
               edgecolor="white", linewidth=0.3, zorder=3,
               label="best static")
        ax.bar(t + bw/2, sa, width=bw, color=C_ADAPTIVE,
               hatch="////", edgecolor="white", linewidth=0.4,
               zorder=4, label="adaptive")

        ax.set_ylim(0, YMAX)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_xlim(t.min() - 0.5, t.max() + 0.5)
        ax.set_xlabel("Time since AOS [min]")

        # ---- elegant totals box (top-left, away from the bars) ----
        ns = int(round(ss.sum()))
        na = int(round(sa.sum()))
        bx, by, bw_box, bh = 0.025, 0.475, 0.42, 0.485
        box = FancyBboxPatch(
            (bx, by), bw_box, bh, transform=ax.transAxes,
            boxstyle="round,pad=0.010,rounding_size=0.04",
            facecolor="white", edgecolor="0.70", linewidth=0.7,
            alpha=0.97, zorder=10, mutation_aspect=0.85)
        ax.add_patch(box)
        cx = bx + 0.032
        ax.text(cx, by + bh - 0.055, "secure pairs / pass",
                transform=ax.transAxes, fontsize=6.6, color="0.45",
                va="top", zorder=11)
        ax.text(cx, by + bh - 0.150, fr"$\Sigma$ static $= {ns}$",
                transform=ax.transAxes, fontsize=8.2, color=C_STATIC,
                va="top", zorder=11)
        ax.text(cx, by + bh - 0.250, fr"$\Sigma$ adapt. $= {na}$",
                transform=ax.transAxes, fontsize=8.2, color="#8A6000",
                va="top", zorder=11, fontweight="bold")
        ydiv = by + 0.150
        ax.plot([bx + 0.028, bx + bw_box - 0.028], [ydiv, ydiv],
                transform=ax.transAxes, color="0.82", lw=0.6, zorder=11)
        ax.text(cx, by + 0.075, "key gain", transform=ax.transAxes,
                ha="left", va="center", fontsize=6.8, color="0.45",
                zorder=11)
        ax.text(bx + bw_box - 0.030, by + 0.072, gain,
                transform=ax.transAxes, ha="right", va="center",
                fontsize=12.0, color=_OI_VERM, fontweight="bold", zorder=11)

    axes[1, 0].set_ylabel("Secure pairs / step")
    axes[1, 1].tick_params(axis="y", labelleft=False)

    # ---------- figure-level legend ----------
    handles = [
        Patch(facecolor=C_STATIC, edgecolor="white", linewidth=0.4,
              label=r"best static (single global $\mathbf{p}^{\star}$)"),
        Patch(facecolor=C_ADAPTIVE, hatch="////", edgecolor="white",
              linewidth=0.4, label="per-step adaptive (oracle)"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.015), ncol=2, fontsize=9,
               frameon=False, handlelength=1.5, columnspacing=2.2,
               handletextpad=0.5)

    fig.savefig(HERE / "fig_cluster_compare.pdf")
    plt.close(fig)


# ============================================================
# Figure 3: TLE-Walker coverage vs constellation size
# ============================================================
def fig_tle_walker_coverage():
    n_vals     = np.array([50,   100,  200,  500])
    served_pct = np.array([6.9, 20.6, 49.7, 62.9])
    sats_vis   = np.array([5,    12,   28,   43])
    handovers  = np.array([8,    19,   42,   55])

    baseline = 46.8
    xpos = np.arange(len(n_vals))   # clean categorical spacing (log-like, no crowding)

    fig, ax = plt.subplots(1, 1, figsize=(3.45, 2.75))

    # --- Right axis FIRST (drawn behind): integer counts, visually subordinate ---
    ax2 = ax.twinx()
    ax2.set_zorder(1)
    ax.set_zorder(2)
    ax.patch.set_visible(False)            # hero axis floats transparently on top
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)  # global rcParams hides it; re-enable on twin
    ax2.spines["right"].set_color("0.45")
    ax2.grid(False)
    l2, = ax2.plot(xpos, sats_vis, "s--", color=C_ADAPTIVE, markersize=4.5,
                   lw=1.1, markeredgecolor="white", markeredgewidth=0.5,
                   label="sats. visible", zorder=3)
    l3, = ax2.plot(xpos, handovers, "^:", color=C_TLE, markersize=4.5,
                   lw=1.1, markeredgecolor="white", markeredgewidth=0.5,
                   label="handover events", zorder=3)
    ax2.set_ylabel("count", color="0.40")
    ax2.tick_params(axis="y", labelcolor="0.40", colors="0.45")
    ax2.set_ylim(0, 72)
    ax2.set_yticks([0, 20, 40, 60])

    # --- Left axis (hero): served-time % as filled area + bold line+markers ---
    ax.fill_between(xpos, 0, served_pct, color=C_ANALYTIC, alpha=0.12, zorder=2)
    l1, = ax.plot(xpos, served_pct, "o-", color=C_ANALYTIC, lw=2.0,
                  markersize=5.5, markeredgecolor="white", markeredgewidth=0.6,
                  label="served-time %", zorder=5)

    # Analytic baseline reference (subtle, labelled in clear space)
    ax.axhline(baseline, ls=(0, (4, 3)), color="0.45", lw=0.9, zorder=1)
    ax.text(0.04, baseline - 1.2, "analytic baseline 46.8%", fontsize=6.8,
            color="0.40", ha="left", va="top")

    ax.set_xlabel(r"Walker shell size $n$ ($P{\times}S$ satellites)")
    ax.set_ylabel("served-time [%]", color=C_ANALYTIC)
    ax.tick_params(axis="y", labelcolor=C_ANALYTIC)
    ax.spines["left"].set_color(C_ANALYTIC)
    ax.set_ylim(0, 72)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_xlim(-0.28, len(n_vals) - 1 + 0.28)
    ax.set_xticks(xpos)
    ax.set_xticklabels([str(n) for n in n_vals])

    # Peak value label at n=500
    ax.annotate(f"{served_pct[-1]:.1f}%", xy=(xpos[-1], served_pct[-1]),
                xytext=(xpos[-1] - 0.02, served_pct[-1] + 3.6), fontsize=7.5,
                color=C_ANALYTIC, ha="center", fontweight="bold")

    # n=200 approximately matches the analytic baseline (callout in whitespace)
    ax.annotate(r"$n{=}200$: 49.7%" "\n" r"($\approx$ baseline)",
                xy=(xpos[2], served_pct[2]), xytext=(2.62, 28.0),
                fontsize=7.2, color=C_ANALYTIC, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                          edgecolor=C_ANALYTIC, lw=0.5, alpha=0.92),
                arrowprops=dict(arrowstyle="-|>", color=C_ANALYTIC, lw=0.8,
                                shrinkA=3, shrinkB=4,
                                connectionstyle="arc3,rad=-0.28"))

    leg = ax.legend(handles=[l1, l2, l3], loc="upper left",
                    fontsize=7.5, framealpha=0.95, borderpad=0.35,
                    handlelength=2.0, labelspacing=0.3)
    leg.set_zorder(10)

    fig.savefig(HERE / "fig_tle_walker_coverage.pdf")
    plt.close(fig)


# ============================================================
# Figure 4: single-link controller 4-axis trade-off
# ============================================================
def fig_controller_tradeoff():
    models  = ["KAN", "MLP", "KNN", "Linear"]
    colors  = [C_KAN, C_MLP, C_KNN, C_LIN]
    hatches = ["", "//", "..", "xx"]   # redundant cue for grayscale / colourblind
    keyret  = [0.716, 0.683, 0.562, 0.614]
    keyret_e= [0.123, 0.086, 0.000, 0.000]
    params  = [392, 1282, 1188, 10]
    latency = [116.7, 5.0, 25.3, 0.1]
    r2_b    = [1.00, 1.00, 0.99, 0.96]

    fig, ax = plt.subplots(1, 4, figsize=(7.0, 2.3),
                           gridspec_kw={"wspace": 0.55})
    x = np.arange(len(models))

    def _style_xaxis(a):
        a.set_xticks(x)
        a.set_xticklabels(models, fontsize=7.5, rotation=30, ha="right")

    bars0 = ax[0].bar(x, keyret, yerr=keyret_e, color=colors, hatch=hatches,
                      capsize=2.5, ecolor="0.3", edgecolor="white", linewidth=0.4)
    _style_xaxis(ax[0])
    ax[0].set_ylabel("Key retention")
    ax[0].set_ylim(0, 1.05)
    ax[0].set_title(r"(a) $M_2$: closed-loop key", fontsize=8.5)
    for b, v in zip(bars0, keyret):
        ax[0].text(b.get_x() + b.get_width()/2, v + 0.04, f"{v:.2f}",
                   ha="center", va="bottom", fontsize=7)

    bars1 = ax[1].bar(x, params, color=colors, hatch=hatches,
                      edgecolor="white", linewidth=0.4)
    _style_xaxis(ax[1])
    ax[1].set_ylabel("# params")
    ax[1].set_yscale("log")
    ax[1].set_ylim(1, 8000)
    ax[1].set_title(r"(b) $M_3$: parameters", fontsize=8.5)
    for b, v in zip(bars1, params):
        ax[1].text(b.get_x() + b.get_width()/2, v*1.4, f"{v}",
                   ha="center", va="bottom", fontsize=7)

    bars2 = ax[2].bar(x, latency, color=colors, hatch=hatches,
                      edgecolor="white", linewidth=0.4)
    _style_xaxis(ax[2])
    ax[2].set_ylabel(r"$\mu$s / prediction")
    ax[2].set_yscale("log")
    ax[2].set_ylim(0.05, 800)
    ax[2].set_title(r"(c) $M_4$: latency", fontsize=8.5)
    for b, v in zip(bars2, latency):
        ax[2].text(b.get_x() + b.get_width()/2, v*1.6, f"{v:g}",
                   ha="center", va="bottom", fontsize=7)

    bars3 = ax[3].bar(x, r2_b, color=colors, hatch=hatches,
                      edgecolor="white", linewidth=0.4)
    _style_xaxis(ax[3])
    ax[3].set_ylabel(r"$R^2$ on $\beta^\star$")
    ax[3].set_ylim(0.90, 1.025)
    ax[3].set_title(r"(d) $M_5$: $\beta$-rule fit", fontsize=8.5)
    for b, v in zip(bars3, r2_b):
        ax[3].text(b.get_x() + b.get_width()/2, v + 0.004, f"{v:.2f}",
                   ha="center", va="bottom", fontsize=7)

    fig.savefig(HERE / "fig_controller_tradeoff.pdf")
    plt.close(fig)


# ============================================================
# Figure 5: beta safety-margin sweep
# ============================================================
def fig_margin_sweep():
    d = read_csv(CODE_RESULTS / "margin_sweep.csv")
    if not d:
        return
    delta = d["delta"]
    mask = delta <= 0.5  # trim trailing decay; the interesting region is [0, 0.3]
    delta = delta[mask]
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.6))
    ax.plot(delta, d["kan_keyret"][mask],    "o-", color=C_KAN, label="KAN")
    ax.plot(delta, d["mlp_keyret"][mask],    "s-", color=C_MLP, label="MLP")
    ax.plot(delta, d["knn_keyret"][mask],    "^-", color=C_KNN, label="KNN")
    ax.plot(delta, d["linear_keyret"][mask], "d-", color=C_LIN, label="Linear")
    ax.axvspan(0.03, 0.10, color="gold", alpha=0.18,
               label=r"$\delta^\star$ band")
    ax.set_xlabel(r"safety margin $\delta$")
    ax.set_ylabel("closed-loop key retention")
    ax.set_xlim(-0.005, 0.50)
    ax.set_ylim(0.45, 0.95)
    ax.legend(loc="lower left", fontsize=7.5, ncol=2)
    fig.savefig(HERE / "fig_margin_sweep.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_zenith_pass()
    fig_cluster_compare()
    fig_tle_walker_coverage()
    fig_controller_tradeoff()
    fig_margin_sweep()
    print("wrote 5 PDF figures to", HERE)
