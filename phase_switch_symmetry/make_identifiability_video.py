from __future__ import annotations

"""Model-aware identifiability animation.

Scatter of the design information matrix's condition number (identifiability /
ill-conditioning) against task recovery error, colored by demonstration count N,
with points revealed progressively in sample-size order. A running Spearman rho
reports the honest (weak-to-moderate) association: better-conditioned designs
(lower condition number) tend to recover the task with lower error.

Run from the repo root::

    python phase_switch_symmetry/make_identifiability_video.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "phase_switch_symmetry_multiseed/identifiability/identifiability_merged.csv"
OUT = ROOT / "phase_switch_symmetry_videos/29_identifiability.mp4"

CMAP = plt.get_cmap("viridis")
X_COL = "condition_number"
Y_COL = "task_error_mean_mm_equiv"


def main():
    df = pd.read_csv(CSV).dropna(subset=[X_COL, Y_COL])
    Ns = sorted(df.sample_size.unique())
    norm = plt.Normalize(Ns[0], Ns[-1])

    groups = [df[df.sample_size == n] for n in Ns]
    sizes = [len(g) for g in groups]

    # fixed axis limits from the full data, with margin
    x = df[X_COL].to_numpy()
    y = df[Y_COL].to_numpy()
    xlim = (x.min() - 0.05 * (x.max() - x.min()), x.max() + 0.05 * (x.max() - x.min()))
    ylim = (y.min() - 0.08 * (y.max() - y.min()), y.max() + 0.08 * (y.max() - y.min()))

    fig, ax = plt.subplots(figsize=(10.6, 6.2))
    fig.subplots_adjust(left=0.09, right=0.86, top=0.85, bottom=0.13)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("design condition number (identifiability / ill-conditioning)")
    ax.set_ylabel("task error (mm equiv)")
    ax.set_title("Identifiability vs recovery, colored by N", fontweight="bold")
    ax.grid(alpha=0.25, ls="--")

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02)
    cb.set_label("demonstrations N")

    txt = fig.text(0.5, 0.02, "", ha="center", fontsize=10.5, color="#333333")

    scatter = ax.scatter([], [], s=34, alpha=0.85, edgecolor="white", linewidth=0.4)
    n_total = sum(sizes)
    per_group = [max(12, int(120 * sz / n_total)) for sz in sizes]
    frames = []
    for gi, cnt in enumerate(per_group):
        for k in range(cnt):
            frames.append((gi, int(np.ceil(sizes[gi] * (k + 1) / cnt))))

    def update(t):
        gi, upto = frames[t]
        sub = groups[gi].iloc[:upto]
        px = sub[X_COL].to_numpy()
        py = sub[Y_COL].to_numpy()
        c = [CMAP(norm(n)) for n in sub.sample_size]
        scatter.set_offsets(np.column_stack([px, py]))
        scatter.set_facecolors(np.asarray(c))
        rho = spearmanr(px, py).statistic if len(px) >= 5 else float("nan")
        txt.set_text(
            f"N = {Ns[gi]}   ·   {upto}/{sizes[gi]} subsets   ·   "
            f"Spearman rho(condition number, error) = {rho:+.3f}"
        )
        return [scatter, txt]

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000.0 / 12.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=12, codec="libx264", bitrate=2500,
                                    extra_args=["-pix_fmt", "yuv420p"])
    anim.save(str(OUT), writer=writer)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
