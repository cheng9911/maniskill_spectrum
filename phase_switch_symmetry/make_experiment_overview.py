from __future__ import annotations

"""Generate the paper's simulation-experiment overview figure.

A single schematic that summarizes every simulation experiment reported in
Table 1 of main.tex (tab:protocol): the task, the intervention, the frozen
39-condition manifest, the six experiment groups with their usable/fit counts,
and the totals. Uses the frozen numbers from main.tex Table 1.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ---- Palette ---------------------------------------------------------------
C_HEADER = "#1b4f72"      # dark blue
C_HEADER_FILL = "#eaf2f8"
C_PHYSICS = "#2471a3"     # blue  (rollout experiments)
C_PHYSICS_FILL = "#eaf2f8"
C_FIT = "#b9770e"         # amber (offline-fit experiments)
C_FIT_FILL = "#fdf2e3"
C_TOTAL = "#17202a"
C_TOTAL_FILL = "#e8e8e8"
C_TEXT = "#1c1c1c"
C_SUB = "#555555"


def box(ax, x, y, w, h, fill, edge, rounding=0.02):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={rounding}",
        linewidth=1.4, edgecolor=edge, facecolor=fill, zorder=2,
    )
    ax.add_patch(p)


def exp_box(ax, x, y, w, h, title, sub, count, unit, tag, kind):
    edge = C_PHYSICS if kind == "physics" else C_FIT
    fill = C_PHYSICS_FILL if kind == "physics" else C_FIT_FILL
    box(ax, x, y, w, h, fill, edge)
    cx = x + w / 2
    ax.text(cx, y + h - 0.155, title, ha="center", va="top",
            fontsize=12.5, fontweight="bold", color=C_TEXT, zorder=3)
    ax.text(cx, y + h - 0.34, sub, ha="center", va="top",
            fontsize=9.2, color=C_SUB, zorder=3)
    # big count + unit
    ax.text(cx, y + 0.135, count, ha="center", va="center",
            fontsize=20, fontweight="bold", color=edge, zorder=3)
    ax.text(cx, y + 0.052, unit, ha="center", va="top",
            fontsize=9.5, color=C_SUB, zorder=3)
    # status tag (top-left corner pill)
    if tag:
        tag_bbox = dict(boxstyle="round,pad=0.25", fc=edge, ec="none")
        ax.text(x + 0.08, y + h - 0.10, tag, ha="left", va="center",
                fontsize=7.8, color="white", fontweight="bold", bbox=tag_bbox, zorder=4)


def main():
    fig = plt.figure(figsize=(11.6, 7.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 7.4)
    ax.axis("off")

    # ---- Header block -----------------------------------------------------
    box(ax, 0.6, 6.30, 10.4, 0.98, C_HEADER_FILL, C_HEADER, rounding=0.03)
    ax.text(5.8, 7.08, "Keyed-to-circular peg insertion",
            ha="center", va="center", fontsize=15.5, fontweight="bold", color=C_HEADER, zorder=3)
    ax.text(5.8, 6.86, "ManiSkill / PhysX   ·   Panda manipulator, pd_joint_pos   ·   motion-planning actions",
            ha="center", va="center", fontsize=9.5, color=C_TEXT, zorder=3)
    ax.text(5.8, 6.64, "Intervention   c = [Δx, Δy, Δψ]   (task-local: lateral translation ×2, axial yaw)",
            ha="center", va="center", fontsize=9.5, color=C_TEXT, zorder=3)
    ax.text(5.8, 6.42, "Frozen manifest:  39 conditions = 30 mixed (train) + 8 isolated (held-out) + 1 zero baseline",
            ha="center", va="center", fontsize=9.5, color=C_TEXT, zorder=3)

    # ---- Experiment boxes (3 x 2 grid) ------------------------------------
    x0, y0, y1 = 0.6, 3.55, 4.95
    w, h = 3.32, 1.28
    gap = 0.14
    x1 = x0 + w + gap
    x2 = x1 + w + gap
    cx = 5.8

    experiments = [
        # (title, sub, count, unit, tag, kind)
        ("Single-seed\nphase switch", "1 seed  ·  39 conditions", "39", "usable episodes", "strict pass", "physics"),
        ("Five-seed\nreplication", "5 seeds × 39 conditions", "195", "usable episodes", "strict pass", "physics"),
        ("Few-shot sweep", "N ∈ {3, 5, 8, 10, 15, 20, 30}  ·  5 seeds", "1,815", "model fits", "no drops", "fit"),
        ("TP-GMM matched\nfew-shot", "N ∈ {5, 10, 20, 30}  ·  matched cells", "80", "model fits", "complete", "fit"),
        ("Rotated axis\n(Q₁ / Q₂)", "2 orientations × 3 seeds × 39", "232", "usable episodes", "frozen", "physics"),
        ("Circular symmetry\n(honest / placebo)", "2 variants × 3 seeds × 39", "234", "usable episodes", "collected", "physics"),
    ]
    positions = [
        (x0, y1), (x1, y1), (x2, y1),
        (x0, y0), (x1, y0), (x2, y0),
    ]
    for (title, sub, count, unit, tag, kind), (bx, by) in zip(experiments, positions):
        exp_box(ax, bx, by, w, h, title, sub, count, unit, tag, kind)

    # ---- Total bar --------------------------------------------------------
    box(ax, 0.6, 0.35, 10.4, 0.98, C_TOTAL_FILL, C_TOTAL, rounding=0.03)
    ax.text(5.8, 1.05, "6 experiments   ·   700 usable physics rollouts   ·   1,895 offline model fits",
            ha="center", va="center", fontsize=14.5, fontweight="bold", color=C_TOTAL, zorder=3)
    ax.text(5.8, 0.72, "All rollouts executed in simulation (ManiSkill/PhysX). Failed attempts retained with "
                       "condition id, attempt id, seed, stop reason, and source hashes.",
            ha="center", va="center", fontsize=8.8, color=C_SUB, zorder=3)

    # ---- Connectors (header -> grid, grid -> total) ------------------------
    for gx in (x0 + w / 2, x1 + w / 2, x2 + w / 2):
        ax.annotate("", xy=(gx, y1 + h + 0.005), xytext=(gx, 6.285),
                    arrowprops=dict(arrowstyle="-", color="#888888", lw=1.2, zorder=1))
    ax.annotate("", xy=(cx, 0.35 + 0.98 + 0.005), xytext=(cx, y0 - 0.005),
                arrowprops=dict(arrowstyle="-", color="#888888", lw=1.2, zorder=1))

    out_png = Path("experiment_overview.png")
    out_pdf = Path("experiment_overview.pdf")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_png}  and  {out_pdf}")


if __name__ == "__main__":
    main()
