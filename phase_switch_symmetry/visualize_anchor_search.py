from __future__ import annotations

"""Visualize the rotated-axis anchor feasibility map.

Reads one or more coarse-search CSVs (x, y, z, reachable, ik_distance) and draws
the workspace feasibility map: a 3D scatter of anchor cells colored by the IK
distance (0 = feasible, large = infeasible), with the Panda base and the derived
peg grasp position marked. A companion 2D slice view shows the x-y heatmap at
each z level.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat

from phase_switch_symmetry_env import PEG_START, SOCKET_CENTER

ROBOT_BASE = np.array([-0.615, 0.0, 0.0])


def peg_position(anchor, orientation):
    """World grasp position of the peg for a given anchor and orientation Q."""
    Q = quat2mat(orientation)
    return np.asarray(anchor, dtype=float) + Q @ (PEG_START - SOCKET_CENTER)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument(
        "--orientation-quat", nargs=4, type=float, required=True,
        help="Orientation Q used to derive the peg grasp position.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("phase_switch_symmetry_rollouts_rotated/anchor_map")
    )
    args = parser.parse_args()

    labels = args.labels if args.labels else [f"Q{i}" for i in range(len(args.files))]
    orientation = np.asarray(args.orientation_quat, dtype=np.float64)
    orientation = orientation / np.linalg.norm(orientation)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
        }
    )

    for label, path in zip(labels, args.files):
        df = pd.read_csv(path)
        fig = plt.figure(figsize=(9.0, 3.6), constrained_layout=True)
        ax = fig.add_subplot(1, 2, 1, projection="3d")
        c = df["ik_distance"].to_numpy(dtype=float)
        sc = ax.scatter(
            df.x, df.y, df.z, c=c, cmap="viridis_r", s=40, depthshade=False,
            edgecolors="0.3", linewidths=0.3,
        )
        ax.scatter(*ROBOT_BASE, c="r", marker="^", s=90, label="Panda base")
        peg = peg_position(df[["x", "y", "z"]].to_numpy(dtype=float), orientation)
        ax.scatter(peg[:, 0], peg[:, 1], peg[:, 2], c="k", marker="x", s=12,
                   alpha=0.5, label="peg grasp pos")
        ax.set_xlabel("anchor x (m)")
        ax.set_ylabel("anchor y (m)")
        ax.set_zlabel("anchor z (m)")
        ax.set_title(f"{label}: 3D feasibility (color = IK dist)", loc="left",
                     fontweight="bold")
        ax.legend(frameon=False, fontsize=6.5)
        fig.colorbar(sc, ax=ax, shrink=0.5, label="min IK distance (m)")

        # 2D slices: one x-y heatmap per z level.
        zs = sorted(df.z.unique())
        nz = len(zs)
        ax2 = fig.add_subplot(1, 2, 2)
        # build a combined heatmap: x rows, (y x z) columns, or use a grid.
        import matplotlib.colors as mcolors

        norm = mcolors.Normalize(vmin=np.nanmin(c), vmax=np.nanmax(c))
        xvals = sorted(df.x.unique())
        yvals = sorted(df.y.unique())
        grid = np.full((len(xvals), len(yvals) * nz), np.nan)
        for zi, z in enumerate(zs):
            sub = df[df.z == z].set_index(["x", "y"])
            for xi, x in enumerate(xvals):
                for yi, y in enumerate(yvals):
                    if (x, y) in sub.index:
                        grid[xi, yi + zi * len(yvals)] = sub.loc[(x, y), "ik_distance"]
        im = ax2.imshow(grid, aspect="auto", cmap="viridis_r", norm=norm,
                        origin="lower")
        ax2.set_yticks(range(len(xvals)), [f"{v:.2f}" for v in xvals])
        ax2.set_ylabel("anchor x (m)")
        ax2.set_xticks(range(0, len(yvals) * nz, len(yvals)),
                       [f"{v:.2f}" for v in zs])
        ax2.set_xlabel("anchor z (m)  [blocks: y from low to high]")
        ax2.set_title(f"{label}: x-y slices per z", loc="left", fontweight="bold")
        fig.colorbar(im, ax=ax2, shrink=0.8, label="min IK distance (m)")

        out = args.out.with_name(f"{args.out.stem}_{label}")
        fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print("saved:", out.with_suffix(".png"))

    # Text summary of the closest-to-feasible anchors.
    for label, path in zip(labels, args.files):
        df = pd.read_csv(path)
        if df.reachable.any():
            print(f"{label}: {int(df.reachable.sum())} feasible")
        else:
            best = df.nsmallest(3, "ik_distance")[["x", "y", "z", "ik_distance"]]
            print(f"{label}: no feasible cell; closest 3 by IK distance:")
            print(best.to_string(index=False))


if __name__ == "__main__":
    main()
