"""Supplement Figure Sx -- full structural audit of learned generator relevance.

This is the complete structural audit figure: the two falsification controls
(symmetry, relation-constraint) plus the full-SE(3) and multi-generator heatmaps.
Main-text Fig. 3 carries the condensed version -- (a) the cross-task bubble
matrix, (b) the symmetry control, (c) the relation-constraint control, and
(d) the multi-generator heatmap; this supplement keeps all four audit panels
together for completeness.

Every heatmap cell is the mean learned Pdiag-finite relevance across 18 frozen
(seed, subset) fits at N=30, then averaged within one task phase.  The panels are
deliberately chosen from experiments that store the full six-generator profiles;
no matrix is reconstructed, simulated, or filled by hand.

Panel map
---------
a  Full SE(3) selectivity: all six learned generator profiles (heatmap).
b  Multi-generator selectivity: all six learned generator profiles (heatmap).
c  Symmetry control: continuous yaw relevance alpha_psi(s) for keyed versus
   circular tasks (mean curve + 1 s.d. band over 18 fits).
d  Relation-constraint control: continuous yaw relevance alpha_psi(s) for
   heading-constrained versus free-yaw pushing (mean + 1 s.d. band).

Color denotes phase-mean relevance (see colorbar, panel-a scale).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parent / "phase_switch_symmetry_multiseed"
OUTPUT = HERE / "figS_structural_relevance"

GENS = ("du", "dv", "dw", "roll", "pitch", "yaw")
GEN_LABELS = (r"$\delta u$", r"$\delta v$", r"$\delta w$", r"$\delta\phi$",
              r"$\delta\theta$", r"$\delta\psi$")
INSERTION_PHASES = ("align", "enter", "unlock", "insert")
PUSH_PHASES = ("reach", "push", "align", "retract")
SAMPLE_SIZE = 30
EXPECTED_FITS = 18
# All displayed phase means lie below 1.05.  This tight range avoids an unused
# high-end colorbar while leaving the learned relevance scale interpretable.
ALPHA_MAX = 1.05

# Two CVD-separated line colors for the counterfactual controls: the yaw-relevant
# condition (keyed / heading-constrained) is blue, its gauge control (circular /
# free-yaw) is orange.
ACTIVE_COLOR = "#2a78d6"
CONTROL_COLOR = "#eb6834"

# A light white->blue sequential ramp (not the deep standard "Blues"): low
# relevance fades to near-white so the selective structure reads at a glance,
# while 0.8..1.0 keeps gradation instead of collapsing to navy.
_HEAT_BLUE = [
    "#f7fafd", "#e7f0fa", "#d2e3f6", "#bcd4f1", "#a3c4ec",
    "#88b3e6", "#6da2e0", "#5291d9", "#3d82d2", "#2a78d6",
]
HEAT_CMAP = mpl.colors.LinearSegmentedColormap.from_list("heat_blue", _HEAT_BLUE, N=256)


def phase_relevance(
    source: Path,
    condition_field: str | None = None,
    condition_value: str | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return a 6 x 4 matrix of phase-mean relevance from stored profiles."""
    data = np.load(source)
    mask = np.asarray(data["sample_size"]) == SAMPLE_SIZE
    if condition_field is not None and condition_value is not None:
        mask &= np.asarray(data[condition_field]) == condition_value
    profiles = np.asarray(data["profile"])[mask]
    phases = np.asarray(data["phase_codes"])
    if profiles.ndim != 3 or profiles.shape[2] != len(GENS):
        raise ValueError(f"Unexpected profile tensor in {source}")
    if profiles.shape[0] != EXPECTED_FITS:
        raise ValueError(
            f"Expected {EXPECTED_FITS} frozen fits, found {profiles.shape[0]} in {source}"
        )
    if set(np.unique(phases)) != {0, 1, 2, 3}:
        raise ValueError(f"Expected four phase codes in {source}")
    per_fit_phase = np.stack(
        [profiles[:, phases == phase, :].mean(axis=1) for phase in range(4)], axis=1
    )  # fit x phase x generator
    return (
        per_fit_phase.mean(axis=0).T,
        per_fit_phase.std(axis=0, ddof=1).T,
        int(profiles.shape[0]),
    )


def continuous_relevance(
    source: Path,
    condition_field: str,
    condition_value: str,
    gen_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (progress, mean, std, boundaries) of one generator's continuous
    profile, averaged over the EXPECTED_FITS frozen fits at N=SAMPLE_SIZE."""
    data = np.load(source)
    mask = np.asarray(data["sample_size"]) == SAMPLE_SIZE
    mask &= np.asarray(data[condition_field]) == condition_value
    profiles = np.asarray(data["profile"])[mask]
    if profiles.shape[0] != EXPECTED_FITS:
        raise ValueError(
            f"Expected {EXPECTED_FITS} frozen fits, found {profiles.shape[0]}"
        )
    progress = np.asarray(data["progress"])
    phase_codes = np.asarray(data["phase_codes"])
    curve = profiles[:, :, gen_idx]  # (fits, steps)
    mean = curve.mean(axis=0)
    std = curve.std(axis=0, ddof=1)
    chg = np.where(np.diff(phase_codes) != 0)[0]
    boundaries = progress[chg + 1]
    return progress, mean, std, boundaries


def style_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    phase_labels: tuple[str, str, str, str],
    show_y: bool = True,
    show_x: bool = True,
) -> mpl.collections.QuadMesh:
    """pcolormesh retains each heatmap cell as an editable vector rectangle."""
    image = ax.pcolormesh(
        np.arange(5) - 0.5,
        np.arange(7) - 0.5,
        matrix,
        cmap=HEAT_CMAP,
        vmin=0.0,
        vmax=ALPHA_MAX,
        shading="flat",
        edgecolors="white",
        linewidth=0.25,
    )
    ax.set_ylim(5.5, -0.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(phase_labels if show_x else [])
    ax.set_yticks(range(6))
    ax.set_yticklabels(GEN_LABELS if show_y else [])
    ax.tick_params(length=0, pad=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.45)
        spine.set_color("0.45")
    return image


def annotate_cells(
    ax: plt.Axes,
    matrix: np.ndarray,
    gen_rows: list[int],
) -> None:
    """Annotate only the decisive generator rows, turning the heatmap from a
    color block into structural evidence (selective release, not absolute)."""
    for gi in gen_rows:
        for phase in range(4):
            value = matrix[gi, phase]
            color = "0.97" if value > 0.78 else "0.15"
            ax.text(phase, gi, f"{value:.2f}", ha="center", va="center",
                    fontsize=5.7, color=color)


def style_yaw_profile(
    ax: plt.Axes,
    series: list[tuple[str, np.ndarray, np.ndarray]],
    phase_labels: tuple[str, str, str, str],
    colors: list[str],
    progress: np.ndarray,
    boundaries: np.ndarray,
    anchors: list[tuple[float, float, str]],
) -> None:
    """Draw one generator's continuous relevance law: mean curve + 1 s.d. band,
    numeric s-axis, phase-region labels on top, phase boundaries as dashed lines.
    Each series is direct-labelled at its own (s, alpha, ha) anchor."""
    for (label, mean, std), color, (s0, a0, ha) in zip(series, colors, anchors):
        ax.plot(progress, mean, color=color, linewidth=1.6, zorder=3)
        ax.fill_between(progress, mean - std, mean + std, color=color,
                        alpha=0.15, linewidth=0, zorder=2)
        ax.text(s0, a0, label, ha=ha, va="bottom", fontsize=6.7,
                color=color, zorder=4)
    for b in boundaries:
        ax.axvline(b, color="0.86", lw=0.45, ls=(0, (2, 2)), zorder=1)
    ax.axhline(0.0, color="0.86", lw=0.45, zorder=1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.03, 1.13)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1.0"])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel(r"$\alpha_{\psi}(s)$")
    ax.set_xlabel(r"phase progress $s$")
    # phase names are interval labels, not sample points: place them inside the
    # top of the axes (not above the spine) so they never collide with the title.
    centers = [0.125, 0.375, 0.625, 0.875]
    trans = ax.get_xaxis_transform()
    for center, name in zip(centers, phase_labels):
        ax.text(center, 0.975, name, ha="center", va="top",
                transform=trans, fontsize=6.5, color="0.38", zorder=5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.5)
        ax.spines[spine].set_color("0.45")


def write_source_data(
    entries: list[
        tuple[str, str, tuple[str, ...], np.ndarray, np.ndarray, int, tuple[str, ...]]
    ]
) -> None:
    fields = [
        "panel",
        "condition",
        "sample_size",
        "fit_count",
        "generator",
        "phase",
        "mean_relevance",
        "std_relevance",
        "displayed_in_figure",
    ]
    with OUTPUT.with_name(OUTPUT.name + "_source_data.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (
            panel,
            condition,
            phase_labels,
            matrix,
            std_matrix,
            fit_count,
            displayed_generators,
        ) in entries:
            for gen, row, std_row in zip(GENS, matrix, std_matrix):
                for phase, value, std_value in zip(phase_labels, row, std_row):
                    writer.writerow(
                        {
                            "panel": panel,
                            "condition": condition,
                            "sample_size": SAMPLE_SIZE,
                            "fit_count": fit_count,
                            "generator": gen,
                            "phase": phase,
                            "mean_relevance": float(value),
                            "std_relevance": float(std_value),
                            "displayed_in_figure": gen in displayed_generators,
                        }
                    )


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

    se3 = DATA_ROOT / "se3_transfer" / "se3_transfer_profiles.npz"
    multigen = DATA_ROOT / "se3_multigen" / "multigen_profiles.npz"
    planar = DATA_ROOT / "planar_push" / "planar_push_profiles.npz"

    keyed, keyed_std, n_keyed = phase_relevance(se3, "task", "keyed")
    multi, multi_std, n_multi = phase_relevance(multigen, "task", "multigen")
    circular, circular_std, n_circular = phase_relevance(se3, "task", "circular_honest")
    heading, heading_std, n_heading = phase_relevance(planar, "arm", "heading_push")
    free_yaw, free_yaw_std, n_free_yaw = phase_relevance(planar, "arm", "free_yaw_push")
    phase_mean_max = max(
        matrix.max() for matrix in (keyed, multi, circular, heading, free_yaw)
    )
    if phase_mean_max > ALPHA_MAX:
        raise ValueError(
            f"Color scale clips a phase mean ({phase_mean_max:.6f} > {ALPHA_MAX:.6f})"
        )

    yaw_idx = GENS.index("yaw")

    fig = plt.figure(figsize=(7.1, 5.0))
    outer = fig.add_gridspec(
        2,
        3,
        left=0.085,
        right=0.965,
        top=0.94,
        bottom=0.095,
        hspace=0.43,
        wspace=0.26,
        height_ratios=(0.82, 1.18),
        width_ratios=(1.0, 1.0, 0.035),
    )
    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    ax_c = fig.add_subplot(outer[1, 0])
    ax_d = fig.add_subplot(outer[1, 1])
    cbar_ax = fig.add_subplot(outer[0, 2])

    dummy_ax = fig.add_subplot(outer[1, 2])
    dummy_ax.axis("off")

    image = style_heatmap(ax_a, keyed, INSERTION_PHASES, show_y=True, show_x=True)
    annotate_cells(ax_a, keyed, [yaw_idx])
    style_heatmap(ax_b, multi, INSERTION_PHASES, show_y=False, show_x=True)
    annotate_cells(ax_b, multi, [0, yaw_idx])  # du + yaw both collapse

    c_prog, c_keyed_mean, c_keyed_std, c_bounds = continuous_relevance(
        se3, "task", "keyed", yaw_idx
    )
    _, c_circ_mean, c_circ_std, _ = continuous_relevance(
        se3, "task", "circular_honest", yaw_idx
    )
    d_prog, d_heading_mean, d_heading_std, d_bounds = continuous_relevance(
        planar, "arm", "heading_push", yaw_idx
    )
    _, d_free_mean, d_free_std, _ = continuous_relevance(
        planar, "arm", "free_yaw_push", yaw_idx
    )

    style_yaw_profile(
        ax_c,
        [
            ("Keyed", c_keyed_mean, c_keyed_std),
            ("Circular symmetry", c_circ_mean, c_circ_std),
        ],
        INSERTION_PHASES,
        [ACTIVE_COLOR, CONTROL_COLOR],
        c_prog,
        c_bounds,
        [(0.43, 1.035, "left"), (0.18, 0.27, "left")],
    )
    style_yaw_profile(
        ax_d,
        [
            ("Heading-constrained", d_heading_mean, d_heading_std),
            ("Free-yaw", d_free_mean, d_free_std),
        ],
        PUSH_PHASES,
        [ACTIVE_COLOR, CONTROL_COLOR],
        d_prog,
        d_bounds,
        [(0.84, 0.73, "right"), (0.30, 0.09, "center")],
    )

    ax_a.set_title(r"(a)  Full-$SE(3)$ selectivity", loc="left", fontweight="semibold", pad=6)
    ax_b.set_title("(b)  Multi-generator selectivity", loc="left", fontweight="semibold", pad=6)
    ax_c.set_title("(c)  Symmetry control", loc="left", fontweight="semibold", pad=6)
    ax_d.set_title("(d)  Relation-constraint control", loc="left", fontweight="semibold", pad=6)

    # Vertical colorbar in its own narrow column, serving only the a/b heatmaps.
    cbar = fig.colorbar(image, cax=cbar_ax, orientation="vertical")
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["0", "0.5", "1.0"])
    cbar.ax.tick_params(labelsize=6.5, length=2.0, width=0.45)
    cbar.outline.set_linewidth(0.45)
    cbar.ax.set_title(r"$\bar{\alpha}$", fontsize=7.0, pad=4)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=400, facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".pdf"), facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".svg"), facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)

    write_source_data(
        [
            ("a", "keyed_insertion", INSERTION_PHASES, keyed, keyed_std, n_keyed, GENS),
            ("b", "multigen_insertion", INSERTION_PHASES, multi, multi_std, n_multi, GENS),
            ("c", "keyed_insertion", INSERTION_PHASES, keyed, keyed_std, n_keyed, ("yaw",)),
            ("c", "circular_symmetry", INSERTION_PHASES, circular, circular_std, n_circular, ("yaw",)),
            ("d", "heading_push", PUSH_PHASES, heading, heading_std, n_heading, ("yaw",)),
            ("d", "free_yaw_push", PUSH_PHASES, free_yaw, free_yaw_std, n_free_yaw, ("yaw",)),
        ]
    )
    print(f"saved {OUTPUT}.{{png,pdf,svg,tiff}}")
    print(
        "fit counts: "
        f"keyed={n_keyed}, multigen={n_multi}, circular={n_circular}, "
        f"heading={n_heading}, free_yaw={n_free_yaw}"
    )


if __name__ == "__main__":
    main()
