"""Publication revision of Fig. 3 with empirical basis matrices.

Contract: SE(3) relevance depends on task constraints and can release multiple
generators selectively. Quantitative overview (a), controls (b), empirical basis
comparison (c), and phase decomposition (d). Python, 183 x 108 mm, editable
PDF/SVG and 600-dpi PNG. No new or simulated observations are introduced.

Reuses the original N=30 relevance aggregation exactly. Panel c loads signed
insert-phase response matrices estimated from frozen simulation trajectories.
Run from any directory; output is placed beside this script.
"""

from pathlib import Path
import csv

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

import fig3_se3_generators as source
from common import GENS, GEN_LABELS, RELEVANCE_CMAP, COMP, CONTROL, TEXT, GRID

HERE = Path(__file__).resolve().parent
PREFIX = HERE / "fig3_se3_generators_publication"
# Learned coefficients are not probabilities: the phase means reach 1.0125.
NORM = mpl.colors.Normalize(0, 1.05)
AREA = 230


def text_color(value):
    rgb = np.asarray(RELEVANCE_CMAP(NORM(value))[:3])
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055)**2.4)
    lum = linear @ np.array([.2126, .7152, .0722])
    return "white" if 1.05 / (lum + .05) > (lum + .05) / .05 else TEXT


def title(fig, x, y, letter, label):
    fig.text(x, y, letter, fontsize=10, weight="bold", va="bottom")
    fig.text(x + .026, y, label, fontsize=8.3, weight="semibold", va="bottom")


def clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.5, width=.6, pad=2)


def overview(fig, matrix):
    ax = fig.add_axes([.075, .615, .845, .275])
    shown = matrix.T
    for row in (0, 2, 4):
        ax.axhspan(row - .5, row + .5, color="#F5F5F8", zorder=0)
    yy, xx = np.indices(shown.shape)
    # Every cell has a neutral locator. All values have proportional-area marks;
    # small values remain in the export even when their marks are sub-pixel.
    ax.scatter(xx.ravel(), yy.ravel(), s=3, color="#DADAE2", linewidths=0, zorder=1)
    artist = ax.scatter(xx.ravel(), yy.ravel(), s=AREA * shown.ravel(),
                        c=shown.ravel(), cmap=RELEVANCE_CMAP, norm=NORM,
                        linewidths=0, zorder=2)
    for (row, col), value in np.ndenumerate(shown):
        if value >= source.DOT_THRESHOLD:
            ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                    fontsize=5.8, color=text_color(value), zorder=3)
    for x in source.FAMILY_DIVIDERS:
        ax.axvline(x, color="#C8C8D2", lw=.6, zorder=1)
    ax.axhline(2.5, color="#BDBDC9", lw=.65)
    for name, start, end in source.FAMILY_LABELS:
        center = (start + end - 1) / 2
        ax.text(center, 1.045, name, transform=ax.get_xaxis_transform(),
                ha="center", fontsize=6.6, color="#555560")
        ax.plot([start-.4, end-.6], [1.025, 1.025],
                transform=ax.get_xaxis_transform(), color="#BABAC5",
                lw=.6, clip_on=False)
    ax.set(xlim=(-.5, 12.5), ylim=(5.5, -.5),
           xticks=range(13), yticks=range(6))
    labels = [t[0].replace("/", "/\n") for t in source.TASKS]
    ax.set_xticklabels(labels, fontsize=6.1, rotation=30, ha="right",
                       rotation_mode="anchor")
    ax.set_yticklabels([GEN_LABELS[g] for g in GENS], fontsize=7.7)
    ax.tick_params(length=0, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    title(fig, .04, .949, "a", "Task-dependent generator structure")
    fig.text(.948, .952, "Pdiag finite · N = 30", ha="right", fontsize=7,
             color="#555560")
    cax = fig.add_axes([.94, .615, .009, .275])
    cbar = fig.colorbar(artist, cax=cax, ticks=[0, .5, 1.05])
    cbar.ax.set_yticklabels(["0", "0.5", "1.05"])
    cbar.ax.tick_params(length=0, pad=3, labelsize=6.2)
    cbar.outline.set_visible(False)
    fig.text(.991, .7525, "Mean relevance (a, d)", rotation=90,
             ha="center", va="center", fontsize=7)
    return ax


def controls(fig):
    title(fig, .04, .443, "b", "Constraint controls")
    configs = [
        (.279, "se3_transfer/se3_transfer_profiles.npz", "task",
         ("keyed", "circular_honest"), ("Keyed", "Circular symmetry"),
         source.INSERTION_PHASES, ((.34, .68), (.15, .31))),
        (.105, "planar_push/planar_push_profiles.npz", "arm",
         ("heading_push", "free_yaw_push"), ("Heading-constrained", "Free-yaw"),
         source.PUSH_PHASES, ((.29, .78), (.14, .13))),
    ]
    rows = []
    for index, (bottom, path, key, arms, labels, phases, anchors) in enumerate(configs):
        ax = fig.add_axes([.075, bottom, .265, .125])
        for boundary in source.PHASE_BOUNDS:
            ax.axvline(boundary, lw=.55, color=GRID, ls=(0, (3, 3)))
        for arm, label, color, style, (x, y) in zip(
                arms, labels, (COMP, CONTROL), ("-", "--"), anchors):
            progress, mean, sd = source._continuous_yaw(path, key, arm)
            # Adjacent phases retain both endpoint samples at each boundary.
            assert np.isfinite([mean, sd]).all() and np.all(np.diff(progress) >= 0)
            ax.fill_between(progress, mean-sd, mean+sd, color=color, alpha=.18,
                            linewidth=0)
            ax.plot(progress, mean, color=color, lw=1.5, ls=style)
            ax.text(x, y, label, color=color, fontsize=6.7, va="bottom")
            rows.extend((arm, float(p), float(m), float(s))
                        for p, m, s in zip(progress, mean, sd))
            # Do not silently clip uncertainty bands.
            assert (mean-sd).min() > -.04 and (mean+sd).max() < 1.12
        ax.set(xlim=(0, 1), ylim=(-.04, 1.12), xticks=[0, .5, 1], yticks=[0, .5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_yticklabels(["0", "0.5", "1"])
        ax.set_ylabel(r"$\alpha_\psi(s)$", fontsize=8)
        for center, phase in zip((.125, .375, .625, .875), phases):
            ax.text(center, 1.065, phase.capitalize(), transform=ax.transAxes,
                    ha="center", fontsize=6.2, color="#555560")
        if index == 1:
            ax.set_xlabel("Phase progress, $s$", labelpad=2)
        clean(ax)
    return rows


def basis(fig):
    title(fig, .385, .443, "c", "Basis dependence")
    matrices = source._build_operators()
    fig.text(.403, .417, "Empirical response · insert phase", fontsize=6.3, color="#555560")
    for index, (x, name, matrix) in enumerate(zip((.403, .56), ("Task-local", "Rotated"), matrices)):
        ax = fig.add_axes([x, .193, .133, .2045])
        if np.abs(matrix).max() > 1.05:
            raise ValueError("Response exceeds signed coefficient color scale")
        im = ax.imshow(matrix, cmap=source.BASIS_CMAP, vmin=-1.05, vmax=1.05)
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        ax.set_xticklabels([GEN_LABELS[g] for g in GENS], fontsize=5.6)
        ax.set_yticklabels([GEN_LABELS[g] for g in GENS] if index == 0 else [], fontsize=5.6)
        ax.tick_params(length=0, pad=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.text(x+.0665, .152, name, ha="center", fontsize=7)
        fig.text(x+.0665, .125, rf"$r_\mathrm{{off}}={source._off_diag_ratio(matrix):.3f}$",
                 ha="center", fontsize=8)
    cax = fig.add_axes([.435, .087, .23, .012])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=6, length=0, pad=2)
    fig.text(.55, .040, "Signed response coefficient", ha="center", fontsize=6.3)
    return matrices


def multigen(fig, matrix):
    title(fig, .735, .443, "d", "Selective release")
    ax = fig.add_axes([.77, .13, .19, .274])
    ax.pcolormesh(np.arange(5)-.5, np.arange(7)-.5, matrix, cmap=RELEVANCE_CMAP,
                  norm=NORM, shading="flat", edgecolors="white", linewidth=.6)
    ax.set(xlim=(-.5, 3.5), ylim=(5.5, -.5), xticks=range(4), yticks=range(6))
    ax.set_xticklabels([p.capitalize() for p in source.INSERTION_PHASES],
                       rotation=30, ha="right", rotation_mode="anchor", fontsize=6.3)
    ax.set_yticklabels([GEN_LABELS[g] for g in GENS], fontsize=7.7)
    for (row, col), value in np.ndenumerate(matrix):
        ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=6.3,
                color=text_color(value))
    for row in (0, 5):
        ax.add_patch(Rectangle((2.5, row-.5), 1, 1, fill=False,
                               edgecolor="#252532", linewidth=1.05))
    ax.tick_params(length=0, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)


def compact_layout(fig):
    """Remove 8 mm between rows and 3 mm below without shrinking any glyphs."""
    original_height, final_height = 119.0, 108.0
    for ax in fig.axes:
        x, y, width, height = ax.get_position(original=True).bounds
        removed = 11.0 if y >= .5 else 3.0
        ax.set_position([x, (y*original_height-removed)/final_height,
                         width, height*original_height/final_height])
    for text in fig.texts:
        x, y = text.get_position()
        removed = 11.0 if y >= .5 else 3.0
        text.set_position((x, (y*original_height-removed)/final_height))
    fig.set_size_inches(183/25.4, final_height/25.4)


def write_csv(suffix, header, rows):
    with PREFIX.with_name(PREFIX.name + suffix).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans", "font.size": 7,
        "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "text.color": TEXT, "axes.labelcolor": TEXT,
        "axes.linewidth": .6, "svg.fonttype": "none", "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })
    matrix = source.mean_alpha_matrix()
    multi = source._multigen_phase_matrix()
    for values in (matrix, multi):
        if not np.isfinite(values).all() or values.min() < NORM.vmin or values.max() > NORM.vmax:
            raise ValueError("Relevance values exceed the declared shared color scale")
    fig = plt.figure(figsize=(7.20472440945, 4.68503937008), facecolor="white")
    overview(fig, matrix)
    control_rows = controls(fig)
    basis_matrices = basis(fig)
    multigen(fig, multi)
    compact_layout(fig)
    # Preserve exact physical dimensions instead of shrinking via tight cropping.
    fig.savefig(PREFIX.with_suffix(".pdf"))
    fig.savefig(PREFIX.with_suffix(".svg"))
    fig.savefig(PREFIX.with_suffix(".png"), dpi=600)
    fig.savefig(PREFIX.with_name(PREFIX.name + "_preview.png"), dpi=300)
    write_csv("_overview.csv", ("task", *GENS),
              ((task[0], *row) for task, row in zip(source.TASKS, matrix)))
    write_csv("_controls.csv", ("condition", "progress", "mean", "sd"), control_rows)
    write_csv("_multigen.csv", ("generator", *source.INSERTION_PHASES), zip(GENS, *multi.T))
    write_csv("_basis.csv", ("basis", "output", "input", "coefficient"),
              ((name, GENS[i], GENS[j], value)
               for name, matrix in zip(("local", "rotated-1"), basis_matrices)
               for (i, j), value in np.ndenumerate(matrix)))
    print(f"Saved {PREFIX.name}: PDF, SVG, PNG, preview and four source-data CSVs")
    print(f"All 78 overview and 24 phase cells retained; shared scale [0, {NORM.vmax}]")
    print(f"Observed maxima: overview={matrix.max():.8f}, multigen={multi.max():.8f}")
    plt.close(fig)


if __name__ == "__main__":
    main()
