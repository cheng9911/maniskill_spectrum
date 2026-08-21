from __future__ import annotations

"""TP-GMM matched few-shot task-error animation.

Visualizes tpgmm_fewshot_summary.csv + tpgmm_fewshot_win_fraction.csv: the
rotation-aware TP-GMM SE(2) baseline's task error converges toward Pdiag finite
as the number of demonstrations N grows, but never beats it (win fraction of
Pdiag stays 1.0 at every N).

Run from the repo root::

    python phase_switch_symmetry/make_tpgmm_fewshot_video.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "phase_switch_symmetry_multiseed/tpgmm_fewshot"
OUT = ROOT / "phase_switch_symmetry_videos/27_tpgmm_matched_fewshot.mp4"

MODEL_COLORS = {
    "Pdiag finite": "#009E73",
    "Full operator": "#D55E00",
    "Generic RBF": "#0072B2",
    "TP-GMM SE(2)": "#CC79A7",
}
MODEL_ORDER = ["Full operator", "Generic RBF", "TP-GMM SE(2)", "Pdiag finite"]


def main():
    summary = pd.read_csv(DATA / "tpgmm_fewshot_summary.csv")
    wins = pd.read_csv(DATA / "tpgmm_fewshot_win_fraction.csv")
    Ns = np.array(sorted(summary.sample_size.unique()), dtype=float)

    fig = plt.figure(figsize=(11.6, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], left=0.06, right=0.97,
                          top=0.84, bottom=0.12, wspace=0.16)

    ax_err = fig.add_subplot(gs[0, 0])
    ax_win = fig.add_subplot(gs[0, 1])

    # ---- left: task error vs N (lines) ----
    lines = {}
    for m in MODEL_ORDER:
        sub = summary[summary.model == m].sort_values("sample_size")
        lines[m] = ax_err.plot(
            sub.sample_size, sub.task_error_mean_over_subset_means,
            marker="o", ms=7, lw=2.5, color=MODEL_COLORS[m], label=m,
        )[0]

    # ---- right: Pdiag win fraction vs N (bars) ----
    bars = ax_win.bar(
        wins.sample_size.astype(float), wins.pdiag_win_fraction,
        width=2.2, color="#009E73", alpha=0.85, edgecolor="none",
    )

    # ---- style ----
    ax_err.set_xticks(Ns)
    ax_err.set_xlabel("demonstrations N")
    ax_err.set_ylabel("task error (mm equiv)")
    ax_err.set_title("Task error vs N", fontweight="bold")
    ax_err.grid(alpha=0.25, ls="--")
    ax_err.legend(fontsize=8.5, loc="upper right", frameon=False)

    ax_win.set_xticks(Ns)
    ax_win.set_ylim(0, 1.15)
    ax_win.set_xlabel("demonstrations N")
    ax_win.set_ylabel("Pdiag win fraction")
    ax_win.set_title("Pdiag beats TP-GMM SE(2)", fontweight="bold")
    ax_win.grid(alpha=0.25, ls="--", axis="y")

    fig.suptitle(
        "TP-GMM matched few-shot: rotation-aware task-frame baseline converges\n"
        "toward Pdiag finite as N grows, but Pdiag wins at every N",
        fontsize=12.5, fontweight="bold",
    )

    # animate: a vertical cursor sweeping N on both panels
    x_cur = 5.0
    cursor_err = ax_err.axvline(x_cur, color="0.35", lw=1.0, ls=":")
    cursor_win = ax_win.axvline(x_cur, color="0.35", lw=1.0, ls=":")
    txt = fig.text(0.5, 0.015, "", ha="center", fontsize=9.5, color="#333333")

    n = 120
    sweep = np.linspace(Ns[0], Ns[-1], n)

    def update(i):
        x = sweep[i]
        cursor_err.set_xdata([x, x])
        cursor_win.set_xdata([x, x])
        # label each line at the current N (nearest x)
        labels = []
        for m in MODEL_ORDER:
            sub = summary[summary.model == m].sort_values("sample_size")
            y = float(np.interp(x, sub.sample_size, sub.task_error_mean_over_subset_means))
            labels.append(f"{m} {y:.2f}")
        # win fraction at nearest N
        w = float(np.interp(x, wins.sample_size, wins.pdiag_win_fraction))
        txt.set_text(f"N = {x:.1f}   |   " + "   ·   ".join(labels) + f"   |   Pdiag win {w:.2f}")
        return [cursor_err, cursor_win, txt]

    anim = animation.FuncAnimation(fig, update, frames=n, interval=1000.0 / 12.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=12, codec="libx264", bitrate=2500,
                                    extra_args=["-pix_fmt", "yuv420p"])
    anim.save(str(OUT), writer=writer)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
