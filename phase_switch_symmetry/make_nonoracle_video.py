from __future__ import annotations

"""Non-oracle progress reparameterization animation.

Shows that the recovered axial-yaw generator law alpha_yaw(s) (a ~1 -> 0 step at
key clearance) survives refitting under five observable progress coordinates
(time, SE(3) arc length, keypoint descent, and two DTW alignments), compared to
the privileged oracle phase label. The oracle panel is the reference; the five
non-oracle panels overlay the oracle mean (dashed) to make the amplitude
smearing visible.

Run from the repo root::

    python phase_switch_symmetry/make_nonoracle_video.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NPZ = ROOT / "phase_switch_symmetry_multiseed/nonoracle_progress/nonoracle_profiles.npz"
OUT = ROOT / "phase_switch_symmetry_videos/28_nonoracle_progress.mp4"

PARAM_ORDER = ["oracle", "time", "arc", "keypoint", "dtw", "dtw-position"]
TITLES = {
    "oracle": "oracle (phases)",
    "time": "time (normalized step)",
    "arc": "arc (SE(3) arc length)",
    "keypoint": "keypoint (z descent)",
    "dtw": "dtw [x,y,ell*yaw]",
    "dtw-position": "dtw [x,y,z]",
}
SEEDS = [20260818, 20270818, 20280818, 20290818, 20300818]
ACCENT = "#2471a3"
REF = "0.45"


def main():
    d = np.load(NPZ)
    progress = d["progress"].astype(float)
    s = (progress - progress.min()) / (progress.max() - progress.min())

    curves = {}
    for c in PARAM_ORDER:
        vals = np.stack([d[f"seed_{sd}_param_{c}"] for sd in SEEDS])
        curves[c] = (vals.mean(0), vals.std(0))

    oracle_mean, _ = curves["oracle"]

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.2))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.86, bottom=0.10, hspace=0.34, wspace=0.14)

    handles = {}
    for ax, c in zip(axes.ravel(), PARAM_ORDER):
        mean, std = curves[c]
        if c != "oracle":
            ax.plot(s, oracle_mean, color=REF, lw=1.6, ls="--", label="oracle ref", zorder=3)
        ax.fill_between(s, mean - std, mean + std, color=ACCENT, alpha=0.18, lw=0)
        handles[c] = ax.plot(s, mean, color=ACCENT, lw=2.4, zorder=4)[0]
        ax.set_title(TITLES[c], fontsize=10.5, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.12, 1.22)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.grid(alpha=0.2, ls="--")
        if c == "oracle":
            ax.set_ylabel(r"$\alpha_{\psi}(s)$")

    fig.suptitle(
        "Non-oracle progress: recovered axial-yaw law $\\alpha_{\\psi}(s)$ (mean over 5 seeds, "
        "shaded = std)\nis preserved under every observable progress coordinate",
        fontsize=12.5, fontweight="bold",
    )
    cursors = [ax.axvline(0.0, color="0.25", lw=1.0, ls=":") for ax in axes.ravel()]
    txt = fig.text(0.5, 0.015, "", ha="center", fontsize=10, color="#333333")

    n = 130

    def update(i):
        x = i / (n - 1)
        for cur in cursors:
            cur.set_xdata([x, x])
        # report alpha at the cursor for oracle vs each coord
        j = int(np.clip(np.round(x * (len(s) - 1)), 0, len(s) - 1))
        parts = []
        for c in PARAM_ORDER:
            parts.append(f"{TITLES[c].split(' ')[0]} {curves[c][0][j]:.2f}")
        txt.set_text(f"s = {x:.2f}   |   " + "   ·   ".join(parts))
        return cursors + [txt]

    anim = animation.FuncAnimation(fig, update, frames=n, interval=1000.0 / 12.0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=12, codec="libx264", bitrate=2500,
                                    extra_args=["-pix_fmt", "yuv420p"])
    anim.save(str(OUT), writer=writer)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
