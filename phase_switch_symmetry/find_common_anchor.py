from __future__ import annotations

"""Find a common workspace anchor across orientations.

Takes the three coarse-feasibility CSVs produced by `search_rotated_anchor.py`
(one per orientation, same grid), joins them on (x, y, z), and finds the
grid cells that are feasible for ALL orientations. Among those, it picks the
cell with the largest distance to the nearest infeasible cell (the deepest
interior point), which is the most robust anchor under planner/contact
stochasticity.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files",
        nargs="+",
        type=Path,
        required=True,
        help="One coarse-feasibility CSV per orientation, in order.",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None, help="Orientation labels."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("phase_switch_symmetry_rollouts_rotated/common_anchor.csv"),
    )
    args = parser.parse_args()

    labels = args.labels if args.labels else [f"Q{i}" for i in range(len(args.files))]
    frames = []
    for label, path in zip(labels, args.files):
        frame = pd.read_csv(path)
        frame = frame.rename(columns={"reachable": f"reachable_{label}"})
        frames.append(frame[["x", "y", "z", f"reachable_{label}"]])
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["x", "y", "z"], how="inner")

    reach_cols = [f"reachable_{label}" for label in labels]
    merged["common_feasible"] = merged[reach_cols].all(axis=1)

    # Margin: distance from each common-feasible cell to the nearest infeasible
    # cell (infeasible in any orientation) AND to the search-box boundary, so a
    # cell touching the box edge is not mistaken for a deep interior point.
    grid_xyz = merged[["x", "y", "z"]].to_numpy(dtype=float)
    feasible = merged["common_feasible"].to_numpy(dtype=bool)
    lo = grid_xyz.min(axis=0)
    hi = grid_xyz.max(axis=0)
    d_boundary = np.min(
        np.stack(
            [grid_xyz - lo, hi - grid_xyz], axis=0
        ).reshape(2 * len(lo), -1),
        axis=0,
    )
    margins = np.full(len(merged), -1.0)
    if feasible.any():
        infeasible_xyz = grid_xyz[~feasible]
        if len(infeasible_xyz) == 0:
            d_infeasible = np.full(feasible.sum(), np.inf)
        else:
            d_infeasible = cdist(grid_xyz[feasible], infeasible_xyz).min(axis=1)
        margins[feasible] = np.minimum(d_infeasible, d_boundary[feasible])
    merged["margin_m"] = margins

    n_common = int(feasible.sum())
    print(f"common feasible cells: {n_common}/{len(merged)}")
    if n_common > 0:
        best = merged[merged.common_feasible].sort_values(
            "margin_m", ascending=False
        ).iloc[0]
        print(
            "best anchor: "
            f"({best.x:.3f}, {best.y:.3f}, {best.z:.3f}) margin={best.margin_m:.3f} m"
        )
    else:
        print("no common feasible cell; expand or shift the workspace box")

    merged.to_csv(args.out, index=False)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
