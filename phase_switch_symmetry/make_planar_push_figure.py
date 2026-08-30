"""Planar-push generator-relevance figure (supplement).

Reads the per-generator relevance profiles of the headline Pdiag-finite model
(``phase_switch_symmetry_multiseed/planar_push/planar_push_profiles.npz``) and
renders a 6 x 2 grid: rows = SE(3) generators [du, dv, dw, roll, pitch, yaw],
columns = arm [heading_push, free_yaw_push]. Each cell plots the mean
``alpha_j(s)`` over the four push phases (reach -> push -> align -> retract)
with a seed/subset std band, against the model-independent solver oracle
(dashed).

The contrast is the point of the supplement: the table constraint RELEASES
du/dv/yaw (heading_push) or only du/dv (free_yaw_push), while dw/roll/pitch are
SUPPRESSED in BOTH arms (block stays on the table). The free_yaw column shows
yaw collapsing to ~0 — the same generator that is ~1 in the heading column —
the cross-task counterpoint to peg-in-hole's phase-dependent yaw drop.

Outputs SVG/PDF/PNG next to the benchmark artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

GENERATOR_NAMES = ("du", "dv", "dw", "roll", "pitch", "yaw")
GENERATOR_LABELS = {
    "du": "d$u$ (x)",
    "dv": "d$v$ (y)",
    "dw": "d$w$ (z)",
    "roll": "d$\\phi$ (roll)",
    "pitch": "d$\\theta$ (pitch)",
    "yaw": "d$\\psi$ (yaw)",
}
PHASE_LABELS = ("reach", "push", "align", "retract")
PHASE_BOUNDARIES = (0.25, 0.5, 0.75)

ARM_ORDER = ("heading_push", "free_yaw_push")
ARM_TITLES = {
    "heading_push": "heading_push  (du/dv/yaw tracked)",
    "free_yaw_push": "free_yaw_push  (du/dv tracked)",
}

# Generators relevant in at least one arm: du/dv (both), yaw (heading only).
SELECTIVE = ("du", "dv", "yaw")
NEUTRAL = "#4A6FA5"
ACCENT = "#C0392B"
ORACLE = "#7F7F7F"


def oracle_profile(arm, progress):
    """Model-independent oracle alpha_j(s): step-constant over the four phases.

    du/dv respond from push on (code >= 1); yaw (heading) responds from align on
    (code >= 2); dw/roll/pitch never respond; yaw never responds for free_yaw.
    """
    phase_codes = np.floor(progress * 4).astype(int).clip(0, 3)
    oracle = np.zeros((len(progress), 6), dtype=np.float64)
    oracle[:, 0] = (phase_codes >= 1).astype(np.float64)  # du
    oracle[:, 1] = (phase_codes >= 1).astype(np.float64)  # dv
    if arm == "heading_push":
        oracle[:, 5] = (phase_codes >= 2).astype(np.float64)  # yaw
    return oracle


def save_pub(fig, path: Path):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=600, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profiles",
        type=Path,
        default=Path(
            "phase_switch_symmetry_multiseed/planar_push/planar_push_profiles.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "phase_switch_symmetry_multiseed/se3_figures/planar_push_relevance"
        ),
    )
    parser.add_argument("--sample-size", type=int, default=None,
                        help="use the largest available sample size by default")
    args = parser.parse_args()

    data = np.load(args.profiles)
    profile = data["profile"]                      # (F, n_steps, 6)
    arm = np.asarray(data["arm"])
    sample_size = np.asarray(data["sample_size"])
    progress = np.asarray(data["progress"])

    n = args.sample_size or int(sample_size.max())

    fig, axes = plt.subplots(
        6, 2, figsize=(7.0, 6.8), sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.25, "wspace": 0.12},
    )

    for col, arm_name in enumerate(ARM_ORDER):
        mask = (arm == arm_name) & (sample_size == n)
        sel = profile[mask]                        # (F_n, n_steps, 6)
        means = sel.mean(axis=0)
        stds = sel.std(axis=0)
        for row, gen in enumerate(GENERATOR_NAMES):
            ax = axes[row, col]
            color = ACCENT if gen in SELECTIVE else NEUTRAL
            ax.fill_between(
                progress, means[:, row] - stds[:, row],
                means[:, row] + stds[:, row], color=color, alpha=0.18, lw=0,
            )
            ax.plot(progress, means[:, row], color=color, lw=1.4)
            ax.plot(progress, oracle_profile(arm_name, progress)[:, row],
                    color=ORACLE, lw=0.9, ls="--")
            ax.axhline(0.0, color="#CCCCCC", lw=0.6, zorder=0)
            for b in PHASE_BOUNDARIES:
                ax.axvline(b, color="#CCCCCC", lw=0.6, ls=":", zorder=0)
            ax.set_ylim(-0.05, 1.12)
            ax.set_xlim(0.0, 1.0)
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.tick_params(labelsize=6)
            if col == 0:
                ax.set_ylabel(GENERATOR_LABELS[gen], fontsize=7)
                ax.yaxis.set_label_coords(-0.24, 0.5)
            if row == 0:
                ax.set_title(ARM_TITLES[arm_name], fontsize=7)
            if row == 5:
                ax.set_xticks([0.125, 0.375, 0.625, 0.875])
                ax.set_xticklabels(PHASE_LABELS, fontsize=6)
            else:
                ax.set_xticks([])

    axes[0, 0].plot([], [], color=NEUTRAL, lw=1.4, label="fitted $\\alpha_j(s)$")
    axes[0, 0].plot([], [], color=ORACLE, lw=0.9, ls="--", label="oracle")
    axes[0, 0].legend(loc="upper left", fontsize=6, handlelength=2.2,
                      borderaxespad=0.2)

    fig.text(0.5, 0.005, "phase progress $s$ (reach -> push -> align -> retract)",
             ha="center", fontsize=7)
    fig.text(0.005, 0.5, "generator relevance $\\alpha_j(s)$",
             va="center", rotation=90, fontsize=7)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_pub(fig, args.output)
    n_fits = int(mask.sum())
    print(f"saved {args.output}.{{svg,pdf,png}}  (N={n}, "
          f"{n_fits} Pdiag-finite fits)")


if __name__ == "__main__":
    main()
