"""Multi-generator SE(3) relevance figure (supplement).

Reads the per-generator relevance profiles of the headline Pdiag-finite model
(``phase_switch_symmetry_multiseed/se3_multigen/multigen_profiles.npz``) and
renders a 6 x 1 grid: rows = SE(3) generators [du, dv, dw, roll, pitch, yaw],
single multigen arm (keyed gate -> rectangular slot). Each cell plots the mean
``alpha_j(s)`` over the four insertion phases (align -> enter -> unlock ->
insert) with a seed/subsets std band, against the model-independent solver
oracle (dashed). The du and yaw rows are the heroes: BOTH relevance profiles
drop across the unlock phase (the expert relaxes both generators), while the
four non-selective rows stay ~1 after the align rise. dv (tracked by the
slot's narrow y direction) stays ~1.

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
PHASE_LABELS = ("align", "enter", "unlock", "insert")
PHASE_BOUNDARIES = (0.25, 0.5, 0.75)

SELECTIVE = ("du", "yaw")           # both released at unlock
NEUTRAL = "#4A6FA5"
ACCENT = "#C0392B"
ORACLE = "#7F7F7F"


def oracle_profile(progress):
    """Model-independent oracle alpha_j(s): step-constant over the four phases.

    du and yaw are selective ([1,1,0,0]); dv/dw/roll/pitch are always-on
    during insertion ([1,1,1,1]).
    """
    phase_codes = np.floor(progress * 4).astype(int).clip(0, 3)
    oracle = np.ones((len(progress), 6), dtype=np.float64)
    selective = (phase_codes < 2).astype(np.float64)
    oracle[:, 0] = selective
    oracle[:, 5] = selective
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
            "phase_switch_symmetry_multiseed/se3_multigen/multigen_profiles.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/se3_figures/se3_multigen_relevance"),
    )
    parser.add_argument("--sample-size", type=int, default=None,
                        help="use the largest available sample size by default")
    args = parser.parse_args()

    data = np.load(args.profiles)
    profile = data["profile"]                      # (F, n_steps, 6)
    sample_size = np.asarray(data["sample_size"])
    progress = np.asarray(data["progress"])
    phase_codes = np.asarray(data["phase_codes"])

    n = args.sample_size or int(sample_size.max())
    mask = sample_size == n
    sel = profile[mask]                            # (F_n, n_steps, 6)
    means = sel.mean(axis=0)                       # (n_steps, 6)
    stds = sel.std(axis=0)

    fig, axes = plt.subplots(
        6, 1, figsize=(4.6, 6.8), sharex=True,
        gridspec_kw={"hspace": 0.25},
    )

    for row, gen in enumerate(GENERATOR_NAMES):
        ax = axes[row]
        color = ACCENT if gen in SELECTIVE else NEUTRAL
        alpha = means[:, row]
        alpha_std = stds[:, row]
        ax.fill_between(
            progress, alpha - alpha_std, alpha + alpha_std,
            color=color, alpha=0.18, lw=0,
        )
        ax.plot(progress, alpha, color=color, lw=1.4,
                label="fitted $\\alpha_j(s)$")
        ax.plot(progress, oracle_profile(progress)[:, row],
                color=ORACLE, lw=0.9, ls="--", label="oracle")
        ax.axhline(0.0, color="#CCCCCC", lw=0.6, zorder=0)
        # shade the unlock_yaw phase (where BOTH selective generators release)
        if gen in SELECTIVE:
            ax.axvspan(0.5, 0.75, color="#F5B7B1", alpha=0.25, zorder=0)
        for b in PHASE_BOUNDARIES:
            ax.axvline(b, color="#CCCCCC", lw=0.6, ls=":", zorder=0)
        ax.set_ylim(-0.05, 1.12)
        ax.set_xlim(0.0, 1.0)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.tick_params(labelsize=6)
        ax.set_ylabel(GENERATOR_LABELS[gen], fontsize=7,
                      color=color if gen in SELECTIVE else "black")
        ax.yaxis.set_label_coords(-0.22, 0.5)
        if row == 5:
            ax.set_xticks([0.125, 0.375, 0.625, 0.875])
            ax.set_xticklabels(PHASE_LABELS, fontsize=6)
        else:
            ax.set_xticks([])

    axes[0].legend(
        loc="upper left", fontsize=6, handlelength=2.2, borderaxespad=0.2,
    )
    fig.text(0.5, 0.005, "phase progress $s$ (four insertion phases)",
             ha="center", fontsize=7)
    fig.text(0.005, 0.5, "generator relevance $\\alpha_j(s)$",
             va="center", rotation=90, fontsize=7)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_pub(fig, args.output)
    print(f"saved {args.output}.{{svg,pdf,png}}  (N={n}, "
          f"{int(mask.sum())} Pdiag-finite fits)")

    # one-line sanity summary for the QA notes
    for j, gen in enumerate(GENERATOR_NAMES):
        pre = means[phase_codes < 2, j].mean()
        post = means[phase_codes >= 2, j].mean()
        print(f"  {gen:5s} {pre:.2f}->{post:.2f}  (Delta = {pre - post:.2f})")


if __name__ == "__main__":
    main()
