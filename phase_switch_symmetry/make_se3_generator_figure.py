"""6-generator SE(3) relevance figure (supplement).

Reads the per-generator relevance profiles of the headline Pdiag-finite model
(``phase_switch_symmetry_multiseed/se3_transfer/se3_transfer_profiles.npz``) and
renders a 6 x 2 quantitative grid: rows = SE(3) generators [du, dv, dw, roll,
pitch, yaw], columns = keyed / circular_honest arm. Each cell plots the mean
``alpha_j(s)`` over the four insertion phases (align -> enter -> unlock ->
insert) with a seed/subsets std band, against the model-independent solver oracle
(dashed). The yaw row is the hero: its relevance drops across the keyed
clearance (unlock) while it stays ~0 throughout the circular arm; the five
non-yaw rows are the non-selective background that stays ~1 after the align
rise.

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

# Restrained palette: one neutral family for the non-selective background, one
# accent for the hero yaw channel.
NEUTRAL = "#4A6FA5"
ACCENT = "#C0392B"
ORACLE = "#7F7F7F"


def oracle_profile(task, progress):
    """Model-independent oracle alpha_j(s): step-constant over the four phases."""
    phase_codes = np.floor(progress * 4).astype(int).clip(0, 3)
    oracle = np.ones((len(progress), 6), dtype=np.float64)
    if task == "keyed":
        oracle[:, 5] = (phase_codes < 2).astype(np.float64)
    else:
        oracle[:, 5] = 0.0
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
            "phase_switch_symmetry_multiseed/se3_transfer/se3_transfer_profiles.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/se3_figures/se3_generator_relevance"),
    )
    parser.add_argument("--sample-size", type=int, default=None,
                        help="use the largest available sample size by default")
    args = parser.parse_args()

    data = np.load(args.profiles)
    profile = data["profile"]                      # (F, n_steps, 6)
    task = np.asarray(data["task"])
    sample_size = np.asarray(data["sample_size"])
    progress = np.asarray(data["progress"])
    phase_codes = np.asarray(data["phase_codes"])

    n = args.sample_size or int(sample_size.max())
    mask = sample_size == n
    tasks = ("keyed", "circular_honest")
    task_titles = {"keyed": "keyed (square peg)", "circular_honest": "circular (round peg)"}

    # mean + std of alpha_j(s) across the retained (seed, subset) fits.
    means, stds = {}, {}
    for t in tasks:
        sel = profile[mask & (task == t)]           # (F_t, n_steps, 6)
        means[t] = sel.mean(axis=0)                 # (n_steps, 6)
        stds[t] = sel.std(axis=0)

    fig, axes = plt.subplots(
        6, 2, figsize=(7.0, 7.6), sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.25, "wspace": 0.12},
    )

    for row, gen in enumerate(GENERATOR_NAMES):
        color = ACCENT if gen == "yaw" else NEUTRAL
        for col, t in enumerate(tasks):
            ax = axes[row, col]
            alpha = means[t][:, row]
            alpha_std = stds[t][:, row]
            ax.fill_between(
                progress, alpha - alpha_std, alpha + alpha_std,
                color=color, alpha=0.18, lw=0,
            )
            ax.plot(progress, alpha, color=color, lw=1.4,
                    label="fitted $\\alpha_j(s)$")
            ax.plot(progress, oracle_profile(t, progress)[:, row],
                    color=ORACLE, lw=0.9, ls="--", label="oracle")
            ax.axhline(0.0, color="#CCCCCC", lw=0.6, zorder=0)
            # shade the unlock_yaw phase (where the keyed yaw releases)
            if gen == "yaw":
                ax.axvspan(0.5, 0.75, color="#F5B7B1", alpha=0.25, zorder=0)
            for b in PHASE_BOUNDARIES:
                ax.axvline(b, color="#CCCCCC", lw=0.6, ls=":", zorder=0)
            ax.set_ylim(-0.05, 1.12)
            ax.set_xlim(0.0, 1.0)
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.tick_params(labelsize=6)
            if row == 0:
                ax.set_title(task_titles[t], fontsize=7, pad=4)
            if col == 0:
                ax.set_ylabel(GENERATOR_LABELS[gen], fontsize=7,
                              color=color if gen == "yaw" else "black")
                ax.yaxis.set_label_coords(-0.34, 0.5)
            if row == 5:
                ax.set_xticks([0.125, 0.375, 0.625, 0.875])
                ax.set_xticklabels(PHASE_LABELS, fontsize=6)
            else:
                ax.set_xticks([])

    # single shared legend (drawn once, on the yaw keyed cell)
    axes[5, 0].legend(
        loc="lower left", fontsize=6, handlelength=2.2, borderaxespad=0.2,
    )
    fig.text(0.5, 0.005, "phase progress $s$ (four insertion phases)",
             ha="center", fontsize=7)
    fig.text(0.005, 0.5, "generator relevance $\\alpha_j(s)$",
             va="center", rotation=90, fontsize=7)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_pub(fig, args.output)
    print(f"saved {args.output}.{{svg,pdf,png}}  (N={n}, "
          f"{int(mask.sum())} Pdiag-finite fits per arm)")

    # one-line sanity summary for the QA notes
    for t in tasks:
        yaw_pre = means[t][phase_codes < 2, 5].mean()
        yaw_post = means[t][phase_codes >= 2, 5].mean()
        others_pre = means[t][phase_codes < 2, :5].mean()
        others_post = means[t][phase_codes >= 2, :5].mean()
        print(f"  {t:15s} yaw {yaw_pre:.2f}->{yaw_post:.2f} | "
              f"others {others_pre:.2f}->{others_post:.2f}")


if __name__ == "__main__":
    main()
