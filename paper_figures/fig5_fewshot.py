"""Figure 5 -- few-shot efficiency (one-column wide).

  (a) task-space error E_task vs number of demonstrations N, four models.
      Pdiag finite is lowest and flattest (sample-efficient); the full operator
      and RBF start higher and converge more slowly, TP-GMM SE(2) starts worst.
      A shaded left region carries the N=3 "minimum-excitation stress test"
      (hollow markers), kept visually apart from the N>=5 matched regime.
  (b) generator-law recovery rate vs N.  Pdiag finite recovers the intervention
      law from N=3; the full operator needs N=5; the generic RBF recovers
      unreliably.

A narrow strip above both panels reports complete-law recovery for ours:
92% (qualified) / 88% (random) at N=3, then 100% for N>=5.

Data: tpgmm_fewshot/tpgmm_fewshot_summary.json (matched 4-model sweep, N>=5),
fewshot_results/fewshot_law_recovery_rates.csv (random protocol; TP-GMM SE(2)
was not run in that repeated sweep, hence three lines in (b)), and the N=3
stress-test values pinned from EXPERIMENTS_RECORD.md / VALIDATION.md.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

from common import (
    GRAY_DARK,
    GRID,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_LINESTYLES,
    MODEL_MARKERS,
    MULTISEED,
    OURS,
    TEXT,
    apply_style,
    save_pub,
    style_axes,
)

HERE = Path(__file__).resolve().parent

MODELS_4 = ["Pdiag finite", "TP-GMM SE(2)", "Full operator", "Generic RBF"]
MODELS_3 = ["Pdiag finite", "Full operator", "Generic RBF"]

# N=3 "minimum-excitation stress test" (random protocol) task errors.  These live
# outside tpgmm_fewshot_summary.json (whose matched sweep starts at N=5) and
# outside fewshot_aggregate_summary.csv (which omits TP-GMM SE(2) at N=3), so they
# are pinned here as documented constants -- see EXPERIMENTS_RECORD.md (Random
# rows, N=3) and VALIDATION.md.
N3_TASK_ERROR = {
    "Pdiag finite": (3.27, 0.82),
    "TP-GMM SE(2)": (10.48, 0.56),
    "Full operator": (9.50, 1.17),
    "Generic RBF": (24.12, 0.96),
}


def load_task_error():
    d = json.loads((MULTISEED / "tpgmm_fewshot" / "tpgmm_fewshot_summary.json").read_text())
    out = {}
    for model in MODELS_4:
        ns, vals, errs = [], [], []
        for r in d["summary"]:
            if r["model"] != model:
                continue
            ns.append(int(r["sample_size"]))
            vals.append(float(r["task_error_mean_over_subset_means"]))
            sd = r["between_subset_sd"]
            errs.append(float(sd) if sd == sd else 0.0)  # NaN -> 0 (N=30 single subset)
        order = np.argsort(ns)
        out[model] = (np.array(ns)[order], np.array(vals)[order], np.array(errs)[order])
    return out


def load_law_recovery():
    rows = list(csv.DictReader(
        (MULTISEED / "fewshot_results" / "fewshot_law_recovery_rates.csv").open()))
    out = {}
    for model in MODELS_3:
        ns, vals = [], []
        for r in rows:
            if r["model"] == model and r["protocol"] == "random":
                ns.append(int(r["sample_size"]))
                vals.append(float(r["generator_law_pass_fraction"]))
        order = np.argsort(ns)
        out[model] = (np.array(ns)[order], np.array(vals)[order])
    return out


def panel_recovery_strip(ax):
    """Narrow strip above (a)/(b): complete-law recovery for ours (Pdiag finite),
    a row of small markers -- half-tone at N=3 (92% qualified), solid at N>=5."""
    ns = [3, 5, 10, 20, 30]
    rates = [0.92, 1.0, 1.0, 1.0, 1.0]
    ax.set_xscale("log")
    ax.set_xlim(2.6, 40)
    ax.set_ylim(0.0, 1.3)
    for n, r in zip(ns, rates):
        if n == 3:
            # half-tone: complete-law recovery is not yet saturated at N=3.
            ax.scatter([n], [0.62], s=44, marker="o", facecolor=OURS,
                       edgecolor="none", alpha=0.35, zorder=3)
        else:
            ax.scatter([n], [0.62], s=44, marker="o", facecolor=OURS,
                       edgecolor="none", zorder=3)
        ax.text(n, 0.12, f"{int(round(r * 100))}%", ha="center", va="center",
                fontsize=5.3, color=TEXT)
        ax.text(n, 1.14, f"N={n}", ha="center", va="top", fontsize=5.2,
                color=GRAY_DARK)
    # (N=3 shows qualified 92%; the random-protocol 88% is carried by the caption.)
    ax.set_title("Complete-law recovery (ours)", loc="left", fontsize=6.2,
                 fontweight="semibold", pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def panel_task_error(ax, data):
    for model in MODELS_4:
        ns, vals, errs = data[model]
        # Error bars are downweighted (faded ecolor, thin whiskers, small caps) so
        # the mean curve stays visually dominant over the uncertainty band.
        ax.errorbar(ns, vals, yerr=errs, marker=MODEL_MARKERS[model], ms=4,
                    linestyle=MODEL_LINESTYLES[model],
                    lw=1.6 if model == "Pdiag finite" else 1.2,
                    color=MODEL_COLORS[model], label=MODEL_LABELS[model],
                    ecolor=mpl.colors.to_rgba(MODEL_COLORS[model], 0.45),
                    elinewidth=0.7, capsize=1.6, zorder=3)

    # N=3 minimum-excitation stress test: hollow markers in a shaded left region.
    ax.axvspan(2.5, 4.0, color="#F3F3F3", zorder=0)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(2.7, 0.5, "minimum-excitation stress test", transform=trans,
            rotation=90, ha="center", va="center", fontsize=5.2, color=GRAY_DARK)
    for model in MODELS_4:
        val, err = N3_TASK_ERROR[model]
        ax.errorbar(3, val, yerr=err, marker=MODEL_MARKERS[model], ms=4,
                    markerfacecolor="white", markeredgecolor=MODEL_COLORS[model],
                    linestyle="none", color=MODEL_COLORS[model],
                    ecolor=mpl.colors.to_rgba(MODEL_COLORS[model], 0.45),
                    elinewidth=0.7, capsize=1.6, zorder=3)

    # Direct labels: the ours-vs-TP-GMM gap at N=3 (wide) and N=30 (narrowed).
    ours30 = float(data["Pdiag finite"][1][-1])
    tpgmm30 = float(data["TP-GMM SE(2)"][1][-1])
    ax.annotate(f"{N3_TASK_ERROR['Pdiag finite'][0]:.2f}", (3, N3_TASK_ERROR['Pdiag finite'][0]),
                textcoords="offset points", xytext=(5, 5), fontsize=5.4,
                color=MODEL_COLORS["Pdiag finite"])
    ax.annotate(f"{N3_TASK_ERROR['TP-GMM SE(2)'][0]:.2f}", (3, N3_TASK_ERROR['TP-GMM SE(2)'][0]),
                textcoords="offset points", xytext=(5, 5), fontsize=5.4,
                color=MODEL_COLORS["TP-GMM SE(2)"])
    ax.annotate(f"{ours30:.2f}", (30, ours30), textcoords="offset points",
                xytext=(-14, 6), fontsize=5.4, color=MODEL_COLORS["Pdiag finite"])
    ax.annotate(f"{tpgmm30:.2f}", (30, tpgmm30), textcoords="offset points",
                xytext=(-14, 6), fontsize=5.4, color=MODEL_COLORS["TP-GMM SE(2)"])

    ax.set_xscale("log")
    ax.set_xlim(2.5, 40)
    ax.set_xticks([3, 5, 10, 20, 30])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("demonstrations $N$", fontsize=7)
    ax.set_ylabel("$E_\\mathrm{task}$ (mm-eq.)", fontsize=7)
    ax.set_title("a  Task-space error vs $N$", loc="left", fontsize=8.0, fontweight="semibold")
    style_axes(ax)


def panel_law_recovery(ax, data):
    for model in MODELS_3:
        ns, vals = data[model]
        ax.plot(ns, vals, marker=MODEL_MARKERS[model], ms=4,
                linestyle=MODEL_LINESTYLES[model],
                lw=1.6 if model == "Pdiag finite" else 1.2,
                color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    ax.axhline(1.0, color=GRID, lw=0.6, ls="--", zorder=0)
    ax.set_xscale("log")
    ax.set_xticks([3, 5, 8, 10, 15, 20, 30])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("demonstrations $N$", fontsize=7)
    ax.set_ylabel("law recovery rate", fontsize=7)
    ax.set_ylim(-0.05, 1.08)
    ax.set_title("b  Generator-law recovery vs $N$", loc="left", fontsize=8.0, fontweight="semibold")
    style_axes(ax)


def main():
    apply_style()
    # Unify typography with Fig. 3 (same base sizes; sans-serif math via apply_style).
    mpl.rcParams.update({
        "font.size": 7.5,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.5,
    })
    task_data = load_task_error()
    law_data = load_law_recovery()

    fig = plt.figure(figsize=(6.8, 3.8))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[0.62, 1.0],
                  hspace=0.55, wspace=0.32,
                  left=0.09, right=0.97, top=0.90, bottom=0.13)

    ax_strip = fig.add_subplot(gs[0, :])
    panel_recovery_strip(ax_strip)

    ax_a = fig.add_subplot(gs[1, 0])
    ax_b = fig.add_subplot(gs[1, 1])
    panel_task_error(ax_a, task_data)
    panel_law_recovery(ax_b, law_data)

    handles = [
        Line2D([], [], color=MODEL_COLORS[m], linestyle=MODEL_LINESTYLES[m],
               marker=MODEL_MARKERS[m], ms=4, lw=1.2, label=MODEL_LABELS[m])
        for m in MODELS_4
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.0,
               handlelength=1.6, bbox_to_anchor=(0.5, -0.03), frameon=False)

    save_pub(fig, HERE / "fig5_fewshot")


if __name__ == "__main__":
    main()
