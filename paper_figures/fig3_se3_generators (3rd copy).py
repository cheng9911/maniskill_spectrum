"""Figure 3 -- full SE(3) generator structure (two-column wide).

Four panels carrying the generator-relevance story:

  (a) cross-task generator structure bubble matrix: rows = six SE(3) generators,
      columns = 13 representative tasks grouped by relation family.  Circle area
      and color both encode progress-mean learned alpha_j (Pdiag finite, N=30);
      cells below the display threshold are dropped so the sparse active-set
      structure reads as rows of dots.
  (b) paired controls, stacked: symmetry control (keyed vs circular-symmetry
      gauge) on top, relation-constraint control (heading-constrained vs free-yaw
      pushing) below.  Both show that yaw relevance tracks the task's geometric
      constraint rather than a fixed-coordinate law.
  (c) task-local basis structure: two 6x6 response-operator schematics -- near
      diagonal in the task-local basis, dense in a rotated basis.  The 6x6
      matrices are schematic; only the scalar off-diagonal norms are frozen
      (see Fig. 4, which carries the same scalars).
  (d) multi-generator selectivity: phase-mean relevance of all six generators for
      the multi-generator probe (6 x 4 heatmap).  All six are simultaneously
      active, then only du and yaw selectively release -- not winner-take-all.

Data: libero_relation_suite_profiles.npz, planar_push_profiles.npz,
se3_transfer_profiles.npz, se3_multigen/multigen_profiles.npz.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from common import (
    COMP,
    CONTROL,
    GEN_LABELS,
    GENS,
    GRAY_DARK,
    GRAY_LIGHT,
    GRID,
    MULTISEED,
    PHASE_BOUNDS,
    RELEVANCE_CMAP,
    TEXT,
    save_pub,
    style_axes,
)

HERE = Path(__file__).resolve().parent

# matrix task columns, grouped by relation family (display order):
#   translation-dominant | revolute/joint | placement | keyed insert
TASKS = [
    ("Drawer", "libero", "drawer_middle_open"),
    ("Plate", "libero", "plate_front_push"),
    ("Free-yaw", "planar", "free_yaw_push"),
    ("Heading", "planar", "heading_push"),
    ("Knob", "libero", "stove_knob_turn"),
    ("Microwave", "libero", "microwave_door_revolute"),
    ("Bowl/Stove", "libero", "bowl_on_stove"),
    ("Bowl/Plate", "libero", "bowl_on_plate"),
    ("Cream/Bowl", "libero", "cream_cheese_in_bowl"),
    ("Moka/Stove", "libero", "moka_pot_on_stove"),
    ("Wine/Cab.", "libero", "wine_bottle_on_cabinet"),
    ("Wine/Rack", "libero", "wine_bottle_on_rack"),
    ("Keyed", "se3", "keyed"),
]
# thin dividers between the four relation families (the matrix x-axis spans
# -0.5..12.5 for 13 columns, so family gaps fall at 3.5, 5.5, 11.5).
FAMILY_DIVIDERS = [3.5, 5.5, 11.5]
FAMILY_LABELS = [
    ("Translation / planar", 0, 4),
    ("Revolute", 4, 6),
    ("Placement", 6, 12),
    ("Insertion", 12, 13),
]

# Progress-mean relevance is a fraction in [0,1]; the observed maximum (keyed
# insertion) is ~0.94, so a 0..1 scale needs no clipping.
ALPHA_MAX = 1.0

# Bubble-matrix display settings.  DOT_THRESHOLD is a visual-only cut: cells
# below it are not drawn, which surfaces the sparse active set; it never feeds a
# statistic or selector.  Keep the original indigo-version area encoding: marker
# area is directly proportional to relevance.  Labels are added independently,
# so annotating every visible bubble does not change the old panel geometry.
DOT_THRESHOLD = 0.08
BUBBLE_AREA_SCALE = 180.0

# Falsification-control colors (paper-wide semantics): the yaw-relevant condition
# (keyed / heading-constrained) is blue (COMP), its gauge control (circular
# symmetry / free-yaw) is vermilion (CONTROL).
ACTIVE_COLOR = COMP
CONTROL_COLOR = CONTROL

# Phase names for the two control families (drawn as interval labels above the
# x-axis, matching the phase_codes transitions at 0.25 / 0.5 / 0.75).
INSERTION_PHASES = ("align", "enter", "unlock", "insert")
PUSH_PHASES = ("reach", "push", "align", "retract")

# Shared relevance ramp: white -> indigo (see common.RELEVANCE_CMAP).  The bubble
# matrix (a) and multi-generator heatmap (d) encode the same quantity, so they
# deliberately share this one ramp and one colorbar.
STRUCT_CMAP = RELEVANCE_CMAP


def _bubble_text_color(value):
    """Choose black/white text from the rendered bubble luminance."""
    rgba = STRUCT_CMAP(mpl.colors.Normalize(0.0, ALPHA_MAX)(value))
    rgb_linear = [
        channel / 12.92 if channel <= 0.04045
        else ((channel + 0.055) / 1.055) ** 2.4
        for channel in rgba[:3]
    ]
    luminance = 0.2126 * rgb_linear[0] + 0.7152 * rgb_linear[1] + 0.0722 * rgb_linear[2]
    black_contrast = (luminance + 0.05) / 0.05
    white_contrast = 1.05 / (luminance + 0.05)
    return "white" if white_contrast > black_contrast else TEXT


def _task_phase_means():
    """{(src, key): (6, 4)} phase-mean relevance (gen x phase) for every task
    column, from the frozen N=30 Pdiag-finite profiles."""
    out = {}

    # libero relation profiles (Pdiag finite, N=30).
    lib = np.load(MULTISEED / "libero_relation_suite" / "libero_relation_suite_profiles.npz")
    lib_prof, lib_task, lib_ss, lib_ph = (
        lib["profile"], lib["task_key"], lib["sample_size"], lib["phase_codes"],
    )
    for t in np.unique(lib_task):
        prof = lib_prof[(lib_task == t) & (lib_ss == 30)]
        out[("libero", str(t))] = np.stack(
            [prof[:, lib_ph == p, :].mean(axis=(0, 1)) for p in range(4)], axis=1
        )

    # planar push profiles (Pdiag finite, N=30).
    pp = np.load(MULTISEED / "planar_push" / "planar_push_profiles.npz")
    for arm in ("free_yaw_push", "heading_push"):
        prof = pp["profile"][(pp["arm"] == arm) & (pp["sample_size"] == 30)]
        out[("planar", arm)] = np.stack(
            [prof[:, pp["phase_codes"] == p, :].mean(axis=(0, 1)) for p in range(4)],
            axis=1,
        )

    # se3 keyed (Pdiag finite, N=30).
    se3 = np.load(MULTISEED / "se3_transfer" / "se3_transfer_profiles.npz")
    prof = se3["profile"][(se3["task"] == "keyed") & (se3["sample_size"] == 30)]
    out[("se3", "keyed")] = np.stack(
        [prof[:, se3["phase_codes"] == p, :].mean(axis=(0, 1)) for p in range(4)],
        axis=1,
    )
    return out


def mean_alpha_matrix():
    """(n_tasks, 6) progress-mean alpha_j matrix from the frozen per-task data."""
    phase_means = _task_phase_means()
    mat = np.zeros((len(TASKS), 6))
    for i, (_, src, key) in enumerate(TASKS):
        mat[i] = phase_means[(src, key)].mean(axis=1)
    return mat


def panel_dot_matrix(ax, mat):
    # Sparse bubble matrix: rows = generators (6), columns = tasks (13).  Circle
    # area and color both encode progress-mean relevance; near-zero cells are
    # dropped so the active set reads as rows of dots, not a solid color block.
    # Every drawn circle is annotated with its value (white on dark bubbles,
    # dark on light ones) so the panel reads as a data table, not a schematic.
    shown = mat.T        # 6 generators x 13 tasks
    xs, ys, sizes, colors = [], [], [], []
    for gi in range(6):
        for ti in range(len(TASKS)):
            value = float(shown[gi, ti])
            if value < DOT_THRESHOLD:
                continue
            xs.append(ti)
            ys.append(gi)
            sizes.append(BUBBLE_AREA_SCALE * value)
            colors.append(value)

    sc = ax.scatter(xs, ys, s=sizes, c=colors, cmap=STRUCT_CMAP, vmin=0.0,
                    vmax=ALPHA_MAX, edgecolors="none", linewidths=0, zorder=3)
    for x, y, value in zip(xs, ys, colors):
        text_color = _bubble_text_color(value)
        ax.text(x, y, f"{value:.2f}", ha="center", va="center",
                fontsize=5.0, color=text_color, zorder=4)

    ax.set_xlim(-0.5, len(TASKS) - 0.5)
    ax.set_ylim(-0.5, 5.5)
    ax.invert_yaxis()  # du (row 0) on top, matching imshow origin="upper"
    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels([t[0] for t in TASKS], fontsize=6.1, rotation=30,
                       ha="right", rotation_mode="anchor")
    # ax.set_xlabel("task relation / instance", fontsize=7.0, labelpad=4)
    ax.set_yticks(range(6))
    ax.set_yticklabels([GEN_LABELS[g] for g in GENS], fontsize=7)
    for x in FAMILY_DIVIDERS:
        ax.axvline(x, color=GRID, lw=0.4, zorder=1)
    # family labels above the matrix, one per relation family.
    trans = ax.get_xaxis_transform()
    for name, start, end in FAMILY_LABELS:
        center = (start + end - 1) / 2
        ax.text(center, 1.01, name, transform=trans, ha="center", va="bottom",
                fontsize=5.5, color=GRAY_DARK)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("(a)  Task-dependent generator structure", loc="left",
                 fontsize=8.2, fontweight="semibold", pad=8)
    return sc


def _continuous_yaw(rel_path, key_field, key_value):
    """(progress, mean, std) of the yaw generator's continuous law, N=30."""
    data = np.load(MULTISEED / rel_path)
    sel = (data[key_field] == key_value) & (data["sample_size"] == 30)
    profiles = data["profile"][sel][:, :, GENS.index("yaw")]
    progress = data["progress"]
    return progress, profiles.mean(axis=0), profiles.std(axis=0, ddof=1)


def _multigen_phase_matrix():
    """(6, 4) phase-mean relevance matrix for the multi-generator probe, N=30."""
    data = np.load(MULTISEED / "se3_multigen" / "multigen_profiles.npz")
    sel = data["sample_size"] == 30
    profiles = data["profile"][sel]                 # fits x progress x generator
    phases = data["phase_codes"]
    per_fit_phase = np.stack(
        [profiles[:, phases == p, :].mean(axis=1) for p in range(4)], axis=1
    )                                               # fits x phase x generator
    return per_fit_phase.mean(axis=0).T             # generator x phase


def panel_control(ax, series, phase_labels, progress, anchors, show_y):
    """Two yaw laws (active vs control): mean +- 1 s.d., direct labels, phases."""
    for (label, mean, std), color, (s0, a0, ha) in zip(
        series, (ACTIVE_COLOR, CONTROL_COLOR), anchors
    ):
        ax.fill_between(progress, mean - std, mean + std, color=color,
                        alpha=0.15, linewidth=0, zorder=2)
        ax.plot(progress, mean, color=color, lw=1.6, zorder=3)
        ax.text(s0, a0, label, ha=ha, va="bottom", fontsize=6.7,
                color=color, zorder=4)
    for b in PHASE_BOUNDS:
        ax.axvline(b, color=GRID, lw=0.45, ls=(0, (2, 2)), zorder=1)
    ax.axhline(0, color=GRID, lw=0.45, zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.08)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.5", "1"])
    ax.set_yticks([0.0, 0.5, 1.0])
    if show_y:
        # Keep the mathtext subscript above the 5 pt rendered-glyph floor.
        ax.set_ylabel(r"$\alpha_\psi(s)$", fontsize=6.8)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel("phase progress $s$", fontsize=6.8, labelpad=2)
    # phase names are interval labels (not sample points), placed inside the top.
    centers = [0.125, 0.375, 0.625, 0.875]
    trans = ax.get_xaxis_transform()
    for center, name in zip(centers, phase_labels):
        ax.text(center, 1.1, name, ha="center", va="top", transform=trans,
                fontsize=6.1, color=GRAY_DARK, zorder=5)
    style_axes(ax)


def panel_multigen_heatmap(ax, matrix):
    """6 x 4 phase-mean heatmap; every cell annotated so color carries the pattern
    while the numbers carry the exact audit (Nature correlation-matrix style)."""
    ax.pcolormesh(np.arange(5) - 0.5, np.arange(7) - 0.5, matrix,
                  cmap=STRUCT_CMAP, vmin=0.0, vmax=ALPHA_MAX,
                  shading="flat", edgecolors="white", linewidth=0.3)
    ax.set_ylim(5.5, -0.5)  # generator 0 (du) on top, matching panel (a)
    ax.set_xticks(range(4))
    ax.set_xticklabels(INSERTION_PHASES, fontsize=6.1)
    ax.set_yticks(range(6))
    ax.set_yticklabels([GEN_LABELS[g] for g in GENS], fontsize=6.1)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("phase", fontsize=6.8, labelpad=2)
    for gi in range(6):
        for phase in range(4):
            value = matrix[gi, phase]
            color = "white" if value >= 0.68 else TEXT
            ax.text(phase, gi, f"{value:.2f}", ha="center", va="center",
                    fontsize=5.4, color=color)


# --------------------------------------------------------------------------
# Panel (c): task-local basis structure
# --------------------------------------------------------------------------
# Frozen scalar off-diagonal Frobenius norms (EXPERIMENTS_RECORD.md; the 6x6
# operator matrices themselves are not frozen to disk, so the heatmaps are
# deterministic schematics -- Fig. 4 carries the same scalars).
OFFDIAG_LOCAL = 0.15
OFFDIAG_ROTATED = 2.77


def _build_operators(seed: int = 0):
    """Deterministic schematic 6x6 response operators with the frozen norms."""
    rng = np.random.default_rng(seed)

    def _off_norm(m):
        m = np.asarray(m, dtype=float)
        return float(np.linalg.norm(m - np.diag(np.diag(m))))

    def _make(norm):
        off = rng.standard_normal((6, 6))
        off = (off + off.T) / 2
        np.fill_diagonal(off, 0.0)
        off = off / np.linalg.norm(off) * norm
        return np.eye(6) + off

    A_local = _make(OFFDIAG_LOCAL)
    A_rot = _make(OFFDIAG_ROTATED)
    assert abs(_off_norm(A_local) - OFFDIAG_LOCAL) < 1e-9
    assert abs(_off_norm(A_rot) - OFFDIAG_ROTATED) < 1e-9
    return A_local, A_rot


def _off_diag_ratio(A):
    """Normalized off-diagonal mass r_off = ||A - Diag(diag A)||_F / ||A||_F."""
    off = A - np.diag(np.diag(A))
    return float(np.linalg.norm(off) / np.linalg.norm(A))


def panel_basis_test(fig, subspec):
    """Panel (c): task-local basis structure -- two 6x6 operator heatmaps (near
    diagonal in the task-local basis, dense in a rotated basis), annotated with
    the normalized off-diagonal mass r_off."""
    A_local, A_rot = _build_operators()
    r_local = _off_diag_ratio(A_local)
    r_rot = _off_diag_ratio(A_rot)

    gs_c = GridSpecFromSubplotSpec(1, 2, subplot_spec=subspec,
                                   width_ratios=[1.0, 1.0], wspace=0.30)
    ax_l = fig.add_subplot(gs_c[0, 0])
    ax_r = fig.add_subplot(gs_c[0, 1])

    for ax, M in ((ax_l, A_local), (ax_r, A_rot)):
        ax.imshow(np.abs(M), aspect="equal", cmap=STRUCT_CMAP, vmin=0.0, vmax=1.2)
        ax.set_xticks(range(6))
        ax.set_xticklabels([GEN_LABELS[g] for g in GENS], fontsize=5.4,
                           rotation=35, ha="right")
        ax.set_yticks(range(6))
        ax.set_yticklabels([GEN_LABELS[g] for g in GENS], fontsize=5.4)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax_l.text(0.5, -0.16, "task-local basis", transform=ax_l.transAxes,
              ha="center", va="top", fontsize=6.0, color=TEXT)
    ax_r.text(0.5, -0.16, "rotated basis", transform=ax_r.transAxes,
              ha="center", va="top", fontsize=6.0, color=TEXT)
    ax_l.text(0.5, -0.34, rf"$r_\mathrm{{off}}={r_local:.2f}$",
              transform=ax_l.transAxes, ha="center", va="top",
              fontsize=6.0, color=GRAY_DARK)
    ax_r.text(0.5, -0.34, rf"$r_\mathrm{{off}}={r_rot:.2f}$",
              transform=ax_r.transAxes, ha="center", va="top",
              fontsize=6.0, color=GRAY_DARK)
    return ax_l, ax_r


def add_panel_title(fig, spec, text, dy=0.014, fontsize=7.9):
    """Panel title anchored to a grid cell's top-left (the spec rect, not the
    axes box), so stacked or aspect-shrunk subpanels keep their title on the
    panel's true top and (b)/(c)/(d) sit on one horizontal line."""
    pos = spec.get_position(fig)
    fig.text(pos.x0, pos.y1 + dy, text, ha="left", va="bottom",
             fontsize=fontsize, fontweight="semibold")


def main():
    # Serif (Times) typography, matching Fig. 2 / Fig. 6 and the supplement.
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
    mat = mean_alpha_matrix()
    print("progress-mean relevance (task x [du dv dw roll pitch yaw]):")
    for (name, _, _), row in zip(TASKS, mat):
        print(f"  {name:12s} " + " ".join(f"{v:5.2f}" for v in row))
    mat_max = float(mat.max())
    if mat_max > ALPHA_MAX:
        raise ValueError(f"mean relevance {mat_max:.3f} exceeds color scale {ALPHA_MAX}")

    # continuous controls (yaw law for the active vs gauge condition).
    keyed_prog, keyed_mean, keyed_std = _continuous_yaw(
        "se3_transfer/se3_transfer_profiles.npz", "task", "keyed")
    _, circ_mean, circ_std = _continuous_yaw(
        "se3_transfer/se3_transfer_profiles.npz", "task", "circular_honest")
    head_prog, head_mean, head_std = _continuous_yaw(
        "planar_push/planar_push_profiles.npz", "arm", "heading_push")
    _, free_mean, free_std = _continuous_yaw(
        "planar_push/planar_push_profiles.npz", "arm", "free_yaw_push")

    multi = _multigen_phase_matrix()

    fig = plt.figure(figsize=(7.15, 3.90))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[0.46, 0.54],
                  hspace=0.42, left=0.07, right=0.97, top=0.95, bottom=0.13)

    # Keep panel (a) and its colorbar inside the same top-row grid.  This prevents
    # the colorbar from extending beyond the lower-row right edge and keeps the
    # assembled figure within the intended two-column width.
    gs_top = GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[0, 0],
        width_ratios=[1.0, 0.008, 0.055], wspace=0.035,
    )
    ax_a = fig.add_subplot(gs_top[0, 0])
    sc = panel_dot_matrix(ax_a, mat)
    cax_slot = fig.add_subplot(gs_top[0, 1])
    cax_slot.set_axis_off()
    # Old-version colorbar feel: a tall, very narrow independent strip.
    cax = cax_slot.inset_axes([0.0, 0.0, 1.0, 1.0])

    # Bottom row: (b) stacked paired controls | (c) basis structure | (d) multigen.
    gs_bottom = GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[1, 0],
        width_ratios=[0.95, 1.25, 0.85], wspace=0.30,
    )

    # panel (b): two stacked control curves (symmetry top, constraint bottom).
    gs_b = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_bottom[0, 0],
                                   height_ratios=[1.0, 1.0], hspace=0.34)
    ax_b1 = fig.add_subplot(gs_b[0, 0])
    ax_b2 = fig.add_subplot(gs_b[1, 0])

    panel_control(ax_b1,
                  [("Keyed", keyed_mean, keyed_std),
                   ("Circular symmetry", circ_mean, circ_std)],
                  INSERTION_PHASES, keyed_prog,
                  [(0.38, 0.75, "left"), (0.18, 0.27, "left")],
                  show_y=True)
    # ax_b1.text(0.98, 0.98, "symmetry", transform=ax_b1.transAxes,
    #            ha="right", va="top", fontsize=6.1, color=GRAY_DARK)

    panel_control(ax_b2,
                  [("Heading-constrained", head_mean, head_std),
                   ("Free-yaw", free_mean, free_std)],
                  PUSH_PHASES, head_prog,
                  [(0.84, 0.73, "right"), (0.30, 0.09, "center")],
                  show_y=False)
    # ax_b2.text(0.98, 0.98, "constraint", transform=ax_b2.transAxes,
    #            ha="right", va="top", fontsize=6.1, color=GRAY_DARK)

    # panel (c): task-local basis structure (two 6x6 operator heatmaps).
    panel_basis_test(fig, gs_bottom[0, 1])

    # panel (d): multi-generator selectivity.
    ax_d = fig.add_subplot(gs_bottom[0, 2])
    panel_multigen_heatmap(ax_d, multi)

    # Panel titles on the bottom row's common top edge, so (b)/(c)/(d) align
    # even though (c)'s imshow aspect shrinks its axes box.
    add_panel_title(fig, gs_bottom[0, 0], "(b)  Paired controls")
    add_panel_title(fig, gs_bottom[0, 1], "(c)  Task-local basis structure")
    add_panel_title(fig, gs_bottom[0, 2], "(d)  Multi-generator selectivity")

    # One shared relevance colorbar (a and d encode the same quantity), slim and
    # vertical, inset on the upper right of panel (a).
    cbar = fig.colorbar(sc, cax=cax, orientation="vertical")
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["0", "0.5", "1.0"])
    cbar.ax.tick_params(labelsize=6.5, width=0, length=0, colors=GRAY_DARK)
    cbar.outline.set_linewidth(0.5)   # keep only the border, no tick marks
    cbar.set_label(r"mean relevance $\bar{\alpha}$", fontsize=7.0, labelpad=2)

    save_pub(fig, HERE / "fig3_se3_generators")
    print(f"mean relevance max: {mat_max:.3f}")


if __name__ == "__main__":
    main()