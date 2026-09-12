from __future__ import annotations

"""Segment 3 -- "What P_r(s) does": animate the α_pitch(s) law driving a trajectory.

Left panel: the identified α_pitch(s) curve (mean ± 1 s.d. over three seeds) with
a moving phase marker.  Right panel: a side-view schematic where the peg
progressively tilts by α_pitch(s)·c_tilt to match a hole tilted at 10°, so the
audience sees "α small → large" as a geometric fact, not an abstract curve.

Reads the frozen law from real_relation_consistency/tilt_sweep/frozen_law.npz.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.transforms import Affine2D  # noqa: E402

from video_common import W, H, FPS, write_mp4  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FROZEN = ROOT / "real_relation_consistency" / "tilt_sweep" / "frozen_law.npz"

TILT_DEG = 10.0
PHASE_NAMES = {3: "align", 4: "enter", 5: "unlock", 6: "insert"}
INDIGO = "#4437A8"
GRAY = "#9aa0a6"


def load_alpha():
    data = np.load(FROZEN)
    progress = data["progress"]
    phase = data["phase_codes"]
    curves = np.stack([data[f"seed_{s}_alpha_pitch"] for s in (20260910, 20270910, 20280910)])
    return progress, phase, curves.mean(axis=0), curves.std(axis=0)


def phase_boundaries(phase):
    bounds = []
    prev = None
    for i, p in enumerate(phase):
        if p != prev:
            bounds.append(i)
            prev = p
    bounds.append(len(phase))
    return bounds


def rect_rot(ax, cx, cy, w, h, deg, **kw):
    t = Affine2D().rotate_deg_around(cx, cy, deg) + ax.transData
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, transform=t, **kw))


def draw_schematic(ax, alpha):
    ax.clear()
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-0.5, 4.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    tilt = TILT_DEG  # fixed hole tilt
    # table surface
    ax.axhline(0.0, color="#bbbbbb", lw=3.0, zorder=1)
    # tilted socket (hole) outline, fixed at tilt
    rect_rot(ax, 0.0, 0.75, 1.5, 1.5, tilt, facecolor="#e8e8e8",
             edgecolor="#555555", lw=2.5, zorder=2)
    rect_rot(ax, 0.0, 0.75, 0.9, 1.5, tilt, facecolor="#cfe8e4",
             edgecolor="#1b9e77", lw=2.5, zorder=3)

    # nominal (0 deg) dashed reference peg
    rect_rot(ax, 0.0, 2.6, 0.5, 2.2, 0.0, facecolor="none",
             edgecolor=GRAY, lw=2.0, ls=(0, (6, 4)), zorder=4)

    # adapted peg: tilt = alpha * tilt, descending with alpha (0 -> 1 maps to z)
    peg_tilt = alpha * tilt
    z = 3.4 - 1.8 * alpha
    rect_rot(ax, 0.0, z, 0.5, 2.2, peg_tilt, facecolor=INDIGO,
             edgecolor=INDIGO, alpha=0.85, lw=1.0, zorder=5)

    ax.text(0.0, 3.9, f"peg tilt = {peg_tilt:5.2f}°", ha="center",
            fontsize=20, fontweight="bold", color=INDIGO)
    ax.text(-2.85, 0.35, "hole 10°", fontsize=15, color="#555555", ha="left")
    ax.text(1.15, 2.6, "nominal", fontsize=15, color=GRAY, ha="left")


def main() -> None:
    out_dir = HERE / "segments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "seg3_adapted_trajectory.mp4"

    progress, phase, alpha_mean, alpha_std = load_alpha()
    bounds = phase_boundaries(phase)
    centers = []
    for i in range(len(bounds) - 1):
        centers.append((bounds[i] + bounds[i + 1] - 1) / 2)

    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax_curve = fig.add_axes([0.07, 0.30, 0.48, 0.50])
    ax_schem = fig.add_axes([0.60, 0.12, 0.36, 0.76])
    fig.text(0.5, 0.945, "How  P$_r$(s)  shapes a trajectory",
             ha="center", va="center", fontsize=40, fontweight="bold", color="#222222")
    cap = fig.text(0.5, 0.065, "", ha="center", va="center",
                   fontsize=22, color="#222222")

    n_frames = 480
    frames = []
    for t in range(n_frames):
        s = min(1.0, max(0.0, (t - 60) / 380.0))  # marker sweeps 0..1 over main window
        i = int(round(s * (len(progress) - 1)))
        alpha = float(alpha_mean[i])
        phase_name = PHASE_NAMES.get(int(phase[i]), "?")

        # --- alpha curve panel ---
        ax_curve.clear()
        ax_curve.fill_between(progress, alpha_mean - alpha_std, alpha_mean + alpha_std,
                              color=INDIGO, alpha=0.15, linewidth=0)
        ax_curve.plot(progress, alpha_mean, color=INDIGO, lw=3.0)
        ax_curve.axvline(progress[i], color="#222222", lw=1.6, ls="--", alpha=0.7)
        ax_curve.axhline(0.0, color="#cccccc", lw=1.0)
        ax_curve.axhline(1.0, color="#cccccc", lw=1.0, ls=":")
        # phase shading + labels
        for j in range(len(bounds) - 1):
            x0 = bounds[j] / (len(progress) - 1)
            x1 = (bounds[j + 1] - 1) / (len(progress) - 1)
            ax_curve.axvspan(x0, x1, color="0.90", alpha=0.35, zorder=0)
            ax_curve.text((x0 + x1) / 2, 1.14, PHASE_NAMES.get(int(phase[bounds[j]]), ""),
                          ha="center", va="top", fontsize=16, color="#707070")
        ax_curve.set_xlim(0, 1)
        ax_curve.set_ylim(-0.1, 1.28)
        ax_curve.set_xticks([0, 0.5, 1.0])
        ax_curve.set_xticklabels(["0", "0.5", "1"], fontsize=15)
        ax_curve.set_yticks([0, 0.5, 1.0])
        ax_curve.set_yticklabels(["0", "0.5", "1"], fontsize=15)
        ax_curve.set_xlabel("phase progress  s", fontsize=18)
        ax_curve.set_ylabel("α$_{pitch}$(s)", fontsize=20)
        ax_curve.set_title("identified relation law", loc="left",
                           fontsize=20, fontweight="bold", color=INDIGO, pad=8)
        ax_curve.text(progress[i], alpha_mean[i] + 0.06, f"α={alpha:.2f}",
                      ha="center", fontsize=16, color="#222222", fontweight="bold")

        # --- schematic panel ---
        draw_schematic(ax_schem, alpha)

        cap.set_text(
            f"phase: {phase_name}    ·    α_pitch(s)={alpha:.2f}    ·    "
            f"peg responds {alpha * TILT_DEG:.1f}° of the 10° tilt"
        )

        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())

    write_mp4(out, frames, fps=FPS)
    print(f"wrote {out}  frames={len(frames)}")


if __name__ == "__main__":
    main()
