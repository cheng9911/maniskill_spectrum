"""Fig. 1 -- Intervention-response audit in keyed insertion.

The paper's central claim is that generator *relevance* is phase-dependent and
must be recovered from *intervention response*, not read off the nominal
trajectory.  This figure turns that claim into a mechanistic picture, all real
data (no simulated trajectories):

    (a) the nominal keyed-insertion trajectory (3D): the peg approaches the
        socket from above and inserts, its key orientation staying fixed;
    (b) the same task under held-out isolated yaw interventions
        (Delta psi in {-30, 0, +30} deg): the peg *tracks* the intervention
        during align + enter (key axis rotates to +/-30 deg), then *relaxes*
        once the key clears (the 1 -> 0 response switch);
    (c) a placebo (circular) control: the nominal trajectory is geometrically
        identical, yet the same yaw interventions leave the key orientation
        unchanged -- no propagation, so yaw is irrelevant here;
    (d) the learned yaw relevance alpha_psi(s) (mean +/- 1 s.d. over 18 frozen
        fits) against the empirical isolated-response profile g_psi(s), with the
        clearance boundary and the response half-decay crossing.

The key visual argument: the nominal trajectories in (a) and (c) look the same,
so "what the peg nominally does" cannot reveal yaw relevance -- only the
intervention response in (b) vs (c) can, and the learned profile in (d)
recovers exactly that phase-dependent response.

Data sources
------------
Trajectories       phase_switch_symmetry_rollouts_rotated/{rotated_Q1,circular_placebo}_seed_20260818.h5
Empirical g_psi    common.empirical_yaw_response()  (benchmark_phase_switch_baselines)
Learned alpha_psi  phase_switch_symmetry_multiseed/se3_transfer/se3_transfer_profiles.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from transforms3d.quaternions import quat2mat

from common import COMP, CONTROL, GRAY_DARK, GRID, TEXT, OURS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ROLLOUTS = ROOT / "phase_switch_symmetry_rollouts_rotated"
LEARNED_NPZ = (
    ROOT / "phase_switch_symmetry_multiseed" / "se3_transfer" / "se3_transfer_profiles.npz"
)
OUTPUT = HERE / "fig1_intervention_audit"

# Reuse the project's rollout readers (light module: h5py/numpy only).
sys.path.insert(0, str(ROOT / "phase_switch_symmetry"))
from analyze_phase_switch_rollouts import pose_yaw  # noqa: E402

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
YAW = 5  # generator index of yaw in the six-channel learned profile

PHASE_LABELS = ("Align", "Enter", "Unlock", "Insert")
UNLOCK_S = 0.50  # unlock phase opens at progress s = 0.5 (2nd phase boundary)

# Arms -> H5 file (seed 20260818) and the episodes used for the trajectory panels.
ARM_FILE = {
    "keyed": "rotated_Q1_seed_20260818.h5",
    "placebo": "circular_placebo_seed_20260818.h5",
}
# (nominal, minus30, plus30) episode indices for each arm.
EPISODES = {
    "keyed": (3, 0, 7),
    "placebo": (3, 0, 6),
}

# Intervention line colors: neutral nominal, blue -30 deg, vermilion +30 deg.
INT_STYLE = {
    "0": dict(color=GRAY_DARK, lw=1.3, label=r"$\Delta\psi=0^\circ$"),
    "-30": dict(color=COMP, lw=1.3, label=r"$\Delta\psi=-30^\circ$"),
    "+30": dict(color=CONTROL, lw=1.3, label=r"$\Delta\psi=+30^\circ$"),
}

PRE_CLEAR_COLOR = COMP       # yaw-relevant regime (align + enter)

# Body-frame key axis (radial direction on the peg); yaw rotates it in-plane.
KEY_AXIS = np.array([1.0, 0.0, 0.0])

VIEW = dict(elev=18, azim=-58)  # shared 3D viewpoint for (a)/(b)/(c)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_trajectory(arm: str, episode: int):
    """Return (pos, key_dir, yaw, phase) for one rollout episode.

    ``pos`` is the 3D peg position (x, y, z); ``key_dir`` is the peg's key axis
    in world frame (body-frame KEY_AXIS rotated by the peg quaternion), so its
    swing in the x-y plane is exactly the yaw response and any out-of-plane tilt
    would be roll/pitch.
    """
    path = ROLLOUTS / ARM_FILE[arm]
    with h5py.File(path, "r") as f:
        group = f[f"episode_{episode}"]
        peg = np.asarray(group["peg_pose"])
        phase = np.asarray(group["solver_phase"])
    pos = peg[:, :3]
    rot = np.array([quat2mat(q) for q in peg[:, 3:]])  # (T,3,3), wxyz quaternions
    key_dir = np.einsum("tij,j->ti", rot, KEY_AXIS)
    return pos, key_dir, pose_yaw(peg), phase


def socket_xyz(arm: str) -> np.ndarray:
    """Socket (goal) 3D position from the nominal episode's goal_pose."""
    path = ROLLOUTS / ARM_FILE[arm]
    with h5py.File(path, "r") as f:
        goal = np.asarray(f[f"episode_{EPISODES[arm][0]}"]["goal_pose"])
    return goal[0, :3]


def learned_yaw_profile() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Learned continuous yaw relevance alpha_psi(s): mean +/- 1 s.d. over the
    18 frozen (seed, subset) fits at N=30 in the keyed condition."""
    data = np.load(LEARNED_NPZ)
    profile = np.asarray(data["profile"])          # (fits, steps, 6)
    task = np.asarray(data["task"])
    sample_size = np.asarray(data["sample_size"])
    progress = np.asarray(data["progress"])
    mask = (task == "keyed") & (sample_size == 30)
    yaw = profile[mask][:, :, YAW]
    return progress, yaw.mean(axis=0), yaw.std(axis=0, ddof=1)


def empirical_response() -> dict:
    """Empirical isolated yaw-response g_psi(s) for keyed and placebo arms."""
    import common

    out = {}
    for arm in ("keyed", "circular_placebo"):
        progress, phase_codes, g_yaw, _nominal_yaw = common.empirical_yaw_response(arm)
        out[arm] = (progress, phase_codes, g_yaw)
    return out


def half_decay_crossing(progress: np.ndarray, g_yaw: np.ndarray) -> float:
    """First downward crossing of g_psi(s) = 0.5 at/after the unlock phase."""
    start = int(np.searchsorted(progress, UNLOCK_S))
    for i in range(max(start, 1), len(progress)):
        y0, y1 = g_yaw[i - 1], g_yaw[i]
        if y0 >= 0.5 > y1:
            frac = (0.5 - y0) / (y1 - y0)
            return float(progress[i - 1] + frac * (progress[i] - progress[i - 1]))
    return float("nan")


# --------------------------------------------------------------------------
# Rendering helpers (3D)
# --------------------------------------------------------------------------
def _draw_orient3d(ax, pos, direction, color, length, lw, zorder=5):
    """One 3D segment showing the peg key axis at ``pos`` along ``direction``."""
    p0 = pos - length * direction
    p1 = pos + length * direction
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
            color=color, lw=lw, solid_capstyle="round", zorder=zorder)


def _draw_align_marks3d(ax, pos, key_dir, phase, color, lw):
    """Three key-axis segments along the align approach: at the start, midpoint
    and arrival of the align phase (where keyed yaw rotates)."""
    idx = np.flatnonzero(phase == 3)
    if len(idx) < 2:
        idx = np.arange(len(pos))
    lengths = (0.03, 0.03, 0.05)
    for f, length in zip((0.0, 0.5, 1.0), lengths):
        i = idx[int(round(f * (len(idx) - 1)))]
        _draw_orient3d(ax, pos[i], key_dir[i], color, length, lw)


def _style_axes3d(ax, arm):
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    ax.set_zlabel("$z$ (m)")
    ax.tick_params(labelsize=6.0, width=0.5, length=2.0, pad=-2)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.0)
    ax.view_init(**VIEW)
    ax.set_box_aspect((0.17, 0.28, 0.27))
    gx, gy, gz = socket_xyz(arm)
    ax.scatter([gx], [gy], [gz], s=22, c="white", edgecolors=TEXT, linewidths=0.9, zorder=6)


def panel_nominal(ax, arm):
    ax.set_title("(a)  Nominal keyed insertion", loc="left", fontweight="semibold", pad=5)
    pos, key_dir, yaw, phase = load_trajectory(arm, EPISODES[arm][0])
    m = phase >= 3
    pos_f, key_f, phase_f = pos[m], key_dir[m], phase[m]
    ax.plot(pos_f[:, 0], pos_f[:, 1], pos_f[:, 2], color=PRE_CLEAR_COLOR, lw=1.6, zorder=3)
    _draw_align_marks3d(ax, pos_f, key_f, phase_f, TEXT, 1.1)
    ax.scatter([pos_f[0, 0]], [pos_f[0, 1]], [pos_f[0, 2]], s=10, color=TEXT, zorder=6)
    _style_axes3d(ax, arm)
    gx, gy, gz = socket_xyz(arm)
    ax.text(gx + 0.06, gy - 0.03, gz + 0.06, "clearance", fontsize=6.2, color=GRAY_DARK)


def panel_interventions(ax, arm, title):
    ax.set_title(title, loc="left", fontweight="semibold", pad=5)
    for tag, ep in (("-30", EPISODES[arm][1]), ("0", EPISODES[arm][0]), ("+30", EPISODES[arm][2])):
        pos, key_dir, yaw, phase = load_trajectory(arm, ep)
        m = phase >= 3
        pos_f, key_f, phase_f = pos[m], key_dir[m], phase[m]
        color, lw, label = INT_STYLE[tag]["color"], INT_STYLE[tag]["lw"], INT_STYLE[tag]["label"]
        ax.plot(pos_f[:, 0], pos_f[:, 1], pos_f[:, 2], color=color, lw=lw, zorder=3, label=label)
        _draw_align_marks3d(ax, pos_f, key_f, phase_f, color, 1.1)
    _style_axes3d(ax, arm)
    ax.legend(loc="upper left", fontsize=6.0, frameon=False, borderaxespad=0.1,
              handlelength=1.3, labelspacing=0.25)


def panel_response(ax, progress, g_yaw, learned_mean, learned_std, s_half):
    ax.set_title("(d)  Learned relevance vs. empirical response",
                 loc="left", fontweight="semibold", pad=5)
    ax.fill_between(progress, learned_mean - learned_std, learned_mean + learned_std,
                    color=OURS, alpha=0.15, linewidth=0, zorder=2)
    ax.plot(progress, learned_mean, color=OURS, lw=1.6, zorder=3)
    ax.plot(progress, g_yaw, color=COMP, lw=1.6, zorder=3)
    ax.text(0.74, 0.42, r"learned $\hat{\alpha}_\psi(s)$", fontsize=6.7, color=OURS,
            ha="left", va="bottom", zorder=4)
    ax.text(0.27, 1.02, r"empirical $g_\psi(s)$", fontsize=6.7, color=COMP,
            ha="left", va="top", zorder=4)
    # clearance / unlock boundary
    ax.axvline(UNLOCK_S, color=GRAY_DARK, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.text(UNLOCK_S + 0.006, 0.05, "clearance", fontsize=6.2, color=GRAY_DARK,
            va="bottom", ha="left", rotation=90)
    # half-decay crossing
    ax.axhline(0.5, color=GRID, lw=0.6, ls=":", zorder=1)
    ax.scatter([s_half], [0.5], s=16, facecolor="white", edgecolor=GRAY_DARK,
               linewidth=0.8, zorder=5)
    ax.text(s_half - 0.01, 0.56, rf"$s_{{0.5}}={s_half:.3f}$", fontsize=6.2,
            color=GRAY_DARK, ha="right", va="bottom")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.05, 1.14)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel("yaw relevance / response")
    ax.set_xlabel("phase progress $s$")
    centers = [0.125, 0.375, 0.625, 0.875]
    trans = ax.get_xaxis_transform()
    for ctr, name in zip(centers, PHASE_LABELS):
        ax.text(ctr, 0.975, name, ha="center", va="top", transform=trans,
                fontsize=6.5, color=GRAY_DARK, zorder=5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRAY_DARK)
        ax.spines[spine].set_linewidth(0.5)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
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

    progress, learned_mean, learned_std = learned_yaw_profile()
    resp = empirical_response()
    keyed_progress, _keyed_phase, keyed_g = resp["keyed"]
    _p_prog, _p_phase, placebo_g = resp["circular_placebo"]
    s_half = half_decay_crossing(keyed_progress, keyed_g)

    fig = plt.figure(figsize=(7.1, 5.6))
    gs = GridSpec(2, 2, figure=fig, left=0.055, right=0.99, top=0.90, bottom=0.06,
                  hspace=0.30, wspace=0.22)

    ax_a = fig.add_subplot(gs[0, 0], projection="3d")
    ax_b = fig.add_subplot(gs[0, 1], projection="3d")
    ax_c = fig.add_subplot(gs[1, 0], projection="3d")
    ax_d = fig.add_subplot(gs[1, 1])

    panel_nominal(ax_a, "keyed")
    panel_interventions(ax_b, "keyed", "(b)  Keyed yaw interventions")
    panel_interventions(ax_c, "placebo", "(c)  Placebo (circular) control")
    panel_response(ax_d, keyed_progress, keyed_g, learned_mean, learned_std, s_half)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=400, facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".pdf"), facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    print(f"saved {OUTPUT}.{{png,pdf,svg}}")
    print(f"half-decay crossing s_0.5 = {s_half:.6f}")
    print(f"placebo g_psi range = [{float(placebo_g.min()):.3f}, {float(placebo_g.max()):.3f}]")


if __name__ == "__main__":
    main()
