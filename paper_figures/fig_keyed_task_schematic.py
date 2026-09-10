"""Keyed-insertion task schematic -- one figure, side + top orthographic views.

A single figure showing the whole KeyedCircularPhaseSwitch-v1 insertion process
as *multiple overlaid snapshot ghosts* ("multiple trajectories on one image"):

    TOP view    side cross-section (x-z): the peg descends through the keyed
                gate into the circular bore -- the descent/insertion trajectory.
    BOTTOM view top cross-section (x-y): the key (a 0.042 x 0.028 rectangle)
                must be yaw-aligned to thread the rectangular keyway (0.046 x
                0.032); once it clears the gate it rotates freely (full SO(2))
                inside the circular bore (radius 0.030).

This is the geometric origin of the paper's phase-dependent yaw relevance: the
keyway locks yaw (to ~+/-6 deg) while the key threads the gate, and releases it
the moment the key clears -- the "locked -> free" switch.

Geometry is taken verbatim from ``phase_switch_symmetry_env.py`` (metres):

    shaft radius 0.013 (z in [-0.029, +0.045] rel. peg centre)
    key    0.042 x 0.028 (z in [-0.045, -0.029] rel. centre), half-x 0.021, half-y 0.014
    keyway 0.046 x 0.032  (= 2 x (KEY_HALF + 0.002 clearance))
    gate   z in [0.053, 0.065];  bore radius 0.030 (z in [0, 0.053]);  socket outer 0.070
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle

from common import GRAY_DARK, GRID, TEXT

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "fig_keyed_task_schematic"

# --- task geometry (metres), z is the vertical insertion axis -----------------
SHAFT_RADIUS = 0.013
KEY_HALF_X = 0.021
KEY_HALF_Y = 0.014
KEY_Z = (-0.045, -0.029)     # key z-extent relative to peg centre
SHAFT_Z = (-0.029, 0.045)    # shaft z-extent relative to peg centre

GATE_Z = (0.053, 0.065)      # keyed gate plate
BORE_Z = (0.0, 0.053)        # circular bore (floor at z=0)
KEYWAY_HALF_X = 0.023        # KEY_HALF_X + 0.002 clearance
KEYWAY_HALF_Y = 0.016        # KEY_HALF_Y + 0.002 clearance
BORE_RADIUS = 0.030
SOCKET_OUTER = 0.070

# --- palette ------------------------------------------------------------------
SHAFT_FILL = "#ECECEC"
SOCKET_FILL = "#B9B9B9"
KEY_FILL = "#3278B8"          # accent for the key (the load-bearing feature)

# --- descent snapshot ghosts (peg centre height, world z) ---------------------
GHOSTS = (0.170, 0.150, 0.130, 0.112, 0.095, 0.078, 0.060)
PHASES = {                    # zc -> phase label shown on the side view
    0.170: "Align",
    0.112: "Align",
    0.095: "Enter",
    0.078: "Unlock",
    0.060: "Insert",
}
ALPHA_TOP, ALPHA_BOT = 0.16, 1.0


def _ghost_alpha(zc: float) -> float:
    """Fade ghost snapshots top -> bottom so the descent reads as a trajectory."""
    lo, hi = GHOSTS[0], GHOSTS[-1]
    t = (zc - lo) / (hi - lo)
    return ALPHA_TOP + t * (ALPHA_BOT - ALPHA_TOP)


# --------------------------------------------------------------------------
# Side view (x-z): the descent
# --------------------------------------------------------------------------
def draw_socket_side(ax):
    """Cross-section of the socket: two gate blocks (the keyway) above a bore."""
    gh = GATE_Z[1] - GATE_Z[0]
    bh = BORE_Z[1] - BORE_Z[0]
    # keyed gate -- the "two rectangles" flanking the keyway slot (narrower gap)
    ax.add_patch(Rectangle((KEYWAY_HALF_X, GATE_Z[0]), SOCKET_OUTER - KEYWAY_HALF_X, gh,
                           facecolor=SOCKET_FILL, edgecolor=TEXT, linewidth=0.8, zorder=2))
    ax.add_patch(Rectangle((-SOCKET_OUTER, GATE_Z[0]), SOCKET_OUTER - KEYWAY_HALF_X, gh,
                           facecolor=SOCKET_FILL, edgecolor=TEXT, linewidth=0.8, zorder=2))
    # bore walls (wider gap, radius 0.030) -> the gate overhangs the bore
    ax.add_patch(Rectangle((BORE_RADIUS, BORE_Z[0]), SOCKET_OUTER - BORE_RADIUS, bh,
                           facecolor=SOCKET_FILL, edgecolor=TEXT, linewidth=0.8, zorder=2))
    ax.add_patch(Rectangle((-SOCKET_OUTER, BORE_Z[0]), SOCKET_OUTER - BORE_RADIUS, bh,
                           facecolor=SOCKET_FILL, edgecolor=TEXT, linewidth=0.8, zorder=2))
    # floor / table top
    ax.add_patch(Rectangle((-SOCKET_OUTER, -0.004), 2 * SOCKET_OUTER, 0.004,
                           facecolor=SOCKET_FILL, edgecolor=TEXT, linewidth=0.8, zorder=2))
    # keyway guide lines (the load-bearing constraint)
    for x0 in (-KEYWAY_HALF_X, KEYWAY_HALF_X):
        ax.plot([x0, x0], [GATE_Z[0], GATE_Z[1]], color=GRID, lw=0.5, ls=":", zorder=1)


def draw_peg_side(ax, zc, alpha):
    """Cross-section of the keyed peg: rectangular key below the cylindrical shaft."""
    ax.add_patch(Rectangle((-SHAFT_RADIUS, zc + SHAFT_Z[0]), 2 * SHAFT_RADIUS,
                           SHAFT_Z[1] - SHAFT_Z[0], facecolor=SHAFT_FILL, edgecolor=TEXT,
                           linewidth=0.8, alpha=alpha, zorder=3))
    ax.add_patch(Rectangle((-KEY_HALF_X, zc + KEY_Z[0]), 2 * KEY_HALF_X,
                           KEY_Z[1] - KEY_Z[0], facecolor=KEY_FILL, edgecolor=TEXT,
                           linewidth=0.8, alpha=alpha, zorder=4))


def panel_side(ax):
    draw_socket_side(ax)
    for zc in GHOSTS:
        draw_peg_side(ax, zc, _ghost_alpha(zc))
    ax.set_xlim(-0.088, 0.088)
    ax.set_ylim(-0.010, 0.185)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # part labels
    ax.annotate("shaft", xy=(-0.013, 0.150), xytext=(-0.080, 0.152),
                fontsize=6.5, color=TEXT, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=GRAY_DARK, lw=0.6))
    ax.annotate("key", xy=(-0.021, 0.106), xytext=(-0.080, 0.108),
                fontsize=6.5, color=TEXT, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=GRAY_DARK, lw=0.6))
    ax.annotate("keyed gate", xy=(0.045, 0.059), xytext=(0.040, 0.118),
                fontsize=6.5, color=TEXT, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=GRAY_DARK, lw=0.6))
    ax.annotate("bore", xy=(0.060, 0.025), xytext=(0.040, 0.048),
                fontsize=6.5, color=TEXT, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=GRAY_DARK, lw=0.6))
    # insertion direction
    ax.annotate("", xy=(-0.055, 0.150), xytext=(-0.055, 0.108),
                arrowprops=dict(arrowstyle="-|>", color=GRAY_DARK, lw=1.0))
    ax.text(-0.070, 0.158, "insert", fontsize=6.3, color=GRAY_DARK, ha="left", va="bottom")
    # phase labels along the right margin
    for zc, name in PHASES.items():
        ax.text(0.082, zc + 0.018, name, fontsize=6.0, color=GRAY_DARK, ha="left", va="center")


# --------------------------------------------------------------------------
# Top view (x-y): the yaw constraint / rotation
# --------------------------------------------------------------------------
def key_corners(yaw: float):
    """Four corners of the key (0.042 x 0.028) rotated by ``yaw`` about z."""
    c, s = np.cos(yaw), np.sin(yaw)
    pts = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        x, y = sx * KEY_HALF_X, sy * KEY_HALF_Y
        pts.append((c * x - s * y, s * x + c * y))
    return pts


def draw_socket_top(ax):
    """Top view of the socket: outer disc, keyway rectangle, circular bore below."""
    # socket body disc
    ax.add_patch(Circle((0, 0), SOCKET_OUTER, facecolor=SOCKET_FILL, edgecolor=TEXT,
                        linewidth=0.8, zorder=1))
    # keyway rectangular hole (white = empty passage through the gate)
    ax.add_patch(Rectangle((-KEYWAY_HALF_X, -KEYWAY_HALF_Y), 2 * KEYWAY_HALF_X, 2 * KEYWAY_HALF_Y,
                           facecolor="white", edgecolor=TEXT, linewidth=0.9, zorder=2))
    # circular bore below the gate (dashed = hidden edge)
    ax.add_patch(Circle((0, 0), BORE_RADIUS, facecolor="none", edgecolor=GRAY_DARK,
                        linewidth=0.6, ls=(0, (3, 3)), zorder=3))


def panel_top(ax):
    draw_socket_top(ax)

    # (i) threading the keyway: yaw locked -- aligned key sits in the rectangle
    ax.add_patch(Polygon(key_corners(0.0), closed=True, facecolor=KEY_FILL,
                         edgecolor=TEXT, linewidth=0.8, zorder=5))

    # (ii) after clearing the gate: the key rotates freely -- a fan of ghosts
    for yaw in (np.deg2rad(20), np.deg2rad(40), np.deg2rad(60), np.deg2rad(80)):
        ax.add_patch(Polygon(key_corners(yaw), closed=True, facecolor=KEY_FILL,
                             edgecolor=TEXT, linewidth=0.5, alpha=0.30, zorder=4))
    # rotation arrow (curved, inside the bore)
    r = 0.027
    ax.annotate("", xy=(r * np.cos(1.35), r * np.sin(1.35)),
                xytext=(r * np.cos(0.40), r * np.sin(0.40)),
                arrowprops=dict(arrowstyle="->", color=GRAY_DARK, lw=1.0,
                                connectionstyle="arc3,rad=0.30"), zorder=6)

    ax.set_xlim(-0.088, 0.088)
    ax.set_ylim(-0.088, 0.088)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # labels
    ax.annotate("keyway\n(0.046 x 0.032)", xy=(0.023, 0.0), xytext=(0.058, -0.030),
                fontsize=6.2, color=TEXT, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=GRAY_DARK, lw=0.6))
    ax.annotate("bore (r=0.030)", xy=(0.030, 0.0), xytext=(0.058, 0.030),
                fontsize=6.2, color=GRAY_DARK, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=GRAY_DARK, lw=0.6))
    ax.text(-0.078, 0.072, "in keyway:\nyaw locked", fontsize=6.2, color=TEXT,
            ha="left", va="center")
    ax.text(-0.078, -0.060, "clears gate:\nyaw free", fontsize=6.2, color=GRAY_DARK,
            ha="left", va="center")


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Liberation Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.0,
            "axes.linewidth": 0.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(3.05, 5.6))
    gs = fig.add_gridspec(2, 1, left=0.04, right=0.97, top=0.96, bottom=0.04,
                          height_ratios=(1.05, 1.0), hspace=0.34)
    ax_side = fig.add_subplot(gs[0])
    ax_top = fig.add_subplot(gs[1])

    panel_side(ax_side)
    panel_top(ax_top)
    ax_side.set_title("(a)  descent  (side cross-section)", loc="left", fontsize=8.0, pad=4)
    ax_top.set_title("(b)  yaw  (top cross-section)", loc="left", fontsize=8.0, pad=4)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=400, facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".pdf"), facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
    print(f"saved {OUTPUT}.{{png,pdf,svg}}")


if __name__ == "__main__":
    main()
