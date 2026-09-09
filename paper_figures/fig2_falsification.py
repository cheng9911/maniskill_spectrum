"""Figure 2 -- core falsification figure (two-column wide).

Four panels:
  (a) three matched intervention arms (vector schematic: peg cross-section)
  (b) nominal yaw motion  (deg vs phase progress s)
  (c) empirical yaw intervention response g_psi(s) = d(delta_psi)/d(Delta_psi)
  (d) learned alpha_yaw(s) relevance profile (Pdiag finite)

The honest vs placebo arms look different on nominal motion (b: placebo sweeps
~15 deg of yaw, honest stays ~0) yet both read g_psi ~ 0 (c) and alpha_yaw ~ 0
(d); only the keyed arm shows yaw relevance that collapses across the key
clearance (unlock) phase.

Data: frozen H5 rollouts (empirical response + nominal motion) and
``se3_transfer/se3_transfer_profiles.npz`` (learned alpha).  The placebo learned
alpha_yaw curve is NOT frozen -- only its scalar M1 = 0.0015 is -- so (d) draws
it as a flat reference line annotated with that scalar.
"""

from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle

from common import (
    CONDITION_COLORS,
    CONDITION_LABELS,
    MULTISEED,
    PHASE_BOUNDS,
    empirical_yaw_response,
    save_pub,
    style_axes,
)

HERE = __import__("pathlib").Path(__file__).resolve().parent
ARMS = ("keyed", "circular_honest", "circular_placebo")
PLACEBO_M1 = 0.0015  # frozen scalar (symmetry_transfer_summary.json, Pdiag finite)


def load_learned_alpha_yaw():
    data = np.load(MULTISEED / "se3_transfer" / "se3_transfer_profiles.npz")
    profile = data["profile"]                 # (F, 100, 6)
    task = np.asarray(data["task"])
    sample_size = np.asarray(data["sample_size"])
    progress = np.asarray(data["progress"])
    out = {}
    for arm in ("keyed", "circular_honest"):
        sel = (task == arm) & (sample_size == 30)
        out[arm] = profile[sel, :, 5].mean(axis=0)   # yaw column
    return progress, out


def _rot_arrow(ax, color, label=r"$\Delta\psi$"):
    """Curved intervention-rotation arrow above the peg."""
    ax.add_patch(FancyArrowPatch(
        (0.30, 0.82), (0.70, 0.82), connectionstyle="arc3,rad=0.42",
        arrowstyle="-|>", mutation_scale=9, color=color, lw=1.3,
    ))
    ax.text(0.5, 0.96, label, ha="center", va="center", fontsize=7.5, color=color)


def _draw_arm(ax, arm):
    """Top-down peg-in-socket cross-section (pure vector, no render)."""
    color = CONDITION_COLORS[arm]
    if arm == "keyed":
        # square socket + square peg + key tab (yaw locks after clearance)
        ax.add_patch(Rectangle((0.30, 0.34), 0.40, 0.40, fill=False,
                               ec="#888888", lw=1.1))
        ax.add_patch(Rectangle((0.355, 0.395), 0.29, 0.29, fill=True,
                               fc=color, ec=color, alpha=0.35))
        ax.add_patch(Rectangle((0.465, 0.685), 0.07, 0.055, fill=True,
                               fc=color, ec=color))
        _rot_arrow(ax, color)
        desc = "square key\n$\\Delta\\psi$ changes the gate\n$\\alpha_\\psi$ high pre-clearance"
    elif arm == "circular_honest":
        ax.add_patch(Circle((0.5, 0.54), 0.20, fill=False, ec="#888888", lw=1.1))
        ax.add_patch(Circle((0.5, 0.54), 0.145, fill=True, fc=color, ec=color, alpha=0.35))
        _rot_arrow(ax, color, label=r"$\Delta\psi$ (gauge)")
        desc = "round\n$\\Delta\\psi$ is gauge\n$\\alpha_\\psi$ stays 0"
    else:  # circular_placebo
        ax.add_patch(Circle((0.5, 0.54), 0.20, fill=False, ec="#888888", lw=1.1))
        ax.add_patch(Circle((0.5, 0.54), 0.145, fill=True, fc=color, ec=color, alpha=0.35))
        ax.add_patch(Arc((0.5, 0.54), 0.30, 0.30, theta1=0, theta2=15,
                         color=color, lw=1.5, ls="--"))
        ax.text(0.5 + 0.155 * np.cos(np.deg2rad(7.5)), 0.54 + 0.155 * np.sin(np.deg2rad(7.5)),
                "15°", ha="left", va="bottom", fontsize=6, color=color)
        _rot_arrow(ax, color)
        desc = "round + 15° sweep\nuncorrelated with $\\Delta\\psi$\n$\\alpha_\\psi$ stays 0"
    ax.text(0.5, 0.06, desc, ha="center", va="center", fontsize=6, linespacing=1.3)


def panel_design(ax):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    n = len(ARMS)
    for i, arm in enumerate(ARMS):
        ax_i = ax.inset_axes([i / n + 0.005, 0.0, 1 / n - 0.01, 1.0])
        ax_i.axis("off")
        ax_i.set_xlim(0, 1)
        ax_i.set_ylim(0, 1)
        ax_i.set_aspect("equal")
        _draw_arm(ax_i, arm)
        ax_i.set_title(CONDITION_LABELS[arm], fontsize=7.5,
                       color=CONDITION_COLORS[arm], pad=4)


def _phase_lines(ax):
    for b in PHASE_BOUNDS:
        ax.axvline(b, color="#dddddd", lw=0.6, ls=":", zorder=0)


def panel_nominal(ax, curves):
    for arm in ARMS:
        progress, _, _, nominal = curves[arm]
        ax.plot(progress, nominal, color=CONDITION_COLORS[arm], lw=1.3,
                label=CONDITION_LABELS[arm])
    _phase_lines(ax)
    ax.axhline(0, color="#dddddd", lw=0.6, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_xticks([0.125, 0.375, 0.625, 0.875])
    ax.set_xticklabels(("Align", "Enter", "Unlock", "Insert"), fontsize=6)
    ax.set_ylabel("nominal yaw (deg)", fontsize=7)
    ax.set_title("b  Nominal motion", loc="left", fontsize=8.5, fontweight="bold")
    style_axes(ax)


def panel_empirical(ax, curves):
    for arm in ARMS:
        progress, _, g_yaw, _ = curves[arm]
        ax.plot(progress, g_yaw, color=CONDITION_COLORS[arm], lw=1.6 if arm == "keyed" else 1.3,
                label=CONDITION_LABELS[arm])
    _phase_lines(ax)
    ax.axhline(0, color="#dddddd", lw=0.6, zorder=0)
    ax.axvspan(0.5, 0.75, color="#f5b7b1", alpha=0.22, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_xticks([0.125, 0.375, 0.625, 0.875])
    ax.set_xticklabels(("Align", "Enter", "Unlock", "Insert"), fontsize=6)
    ax.set_ylabel(r"$g_\psi(s)=\partial\delta\psi/\partial\Delta\psi$", fontsize=7)
    ax.set_title("c  Empirical intervention response", loc="left", fontsize=8.5, fontweight="bold")
    style_axes(ax)


def panel_learned(ax, progress, learned):
    for arm in ("keyed", "circular_honest"):
        ax.plot(progress, learned[arm], color=CONDITION_COLORS[arm], lw=1.6 if arm == "keyed" else 1.3)
    ax.axhline(PLACEBO_M1, color=CONDITION_COLORS["circular_placebo"], lw=1.1, ls="--")
    ax.annotate("placebo $\\alpha_\\psi$ (scalar M1=0.0015)",
                xy=(1.0, PLACEBO_M1), xytext=(0.56, 0.14),
                fontsize=6, color=CONDITION_COLORS["circular_placebo"],
                arrowprops=dict(arrowstyle="-", color=CONDITION_COLORS["circular_placebo"], lw=0.7))
    _phase_lines(ax)
    ax.axhline(0, color="#dddddd", lw=0.6, zorder=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xticks([0.125, 0.375, 0.625, 0.875])
    ax.set_xticklabels(("Align", "Enter", "Unlock", "Insert"), fontsize=6)
    ax.set_ylabel(r"learned $\alpha_\psi(s)$", fontsize=7)
    ax.set_title("d  Learned yaw relevance", loc="left", fontsize=8.5, fontweight="bold")
    style_axes(ax)


def main():
    # Serif (Times/STIX) typography to match Fig. 4 and Fig. 5.
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif"],
            "mathtext.fontset": "stix",
            "font.size": 7.5,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.5,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    curves = {arm: empirical_yaw_response(arm) for arm in ARMS}
    progress, learned = load_learned_alpha_yaw()

    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=False)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[0.95, 1.0],
                  hspace=0.55, wspace=0.45, left=0.07, right=0.98, top=0.94, bottom=0.09)

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    ax_a.set_title("a  Three matched intervention arms", loc="left", fontsize=8.5, fontweight="bold")
    panel_design(ax_a)
    panel_nominal(ax_b, curves)
    panel_empirical(ax_c, curves)
    panel_learned(ax_d, progress, learned)

    # one shared legend, drawn once (3 series -> legend present)
    handles = [plt.Line2D([], [], color=CONDITION_COLORS[a], lw=1.6,
                          label=CONDITION_LABELS[a]) for a in ARMS]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.5,
               handlelength=1.6, bbox_to_anchor=(0.5, 0.005), frameon=False)

    for ax in (ax_b, ax_c, ax_d):
        ax.set_xlabel("phase progress $s$", fontsize=7)

    save_pub(fig, HERE / "fig2_falsification")
    # sanity summary
    for arm in ARMS:
        _, pc, g, nom = curves[arm]
        pre = g[pc < 2].mean()
        post = g[pc >= 2].mean()
        print(f"  {arm:16s} g_psi pre={pre:.3f} post={post:.3f}  "
              f"nominal yaw {nom.min():.2f}..{nom.max():.2f} deg")


if __name__ == "__main__":
    main()
