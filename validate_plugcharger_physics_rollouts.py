#!/usr/bin/env python3
"""
validate_plugcharger_physics_rollouts.py

Fast integrity check for outputs created by collect_plugcharger_causal_physics.py.
This is NOT the final P(s) analysis. It verifies that the data really contains
variable-length PhysX execution traces and physical contact, rather than the old
80-frame geometric interpolation.
"""

import argparse
from pathlib import Path
import h5py
import numpy as np
import pandas as pd


def analyze(path: Path):
    rows = []
    with h5py.File(path, "r") as f:
        source = f.attrs.get("source_type", "")
        note = f.attrs.get("note", "")
        keys = sorted(
            [k for k in f.keys() if k.startswith("episode_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        for k in keys:
            g = f[k]
            tcp = np.asarray(g["tcp_pose"])
            rec = np.asarray(g["receptacle_pose"])
            contact = np.asarray(g["contact_force"])
            succ = np.asarray(g["success"]).astype(bool)
            c = np.asarray(g["causal_delta"])
            force = np.linalg.norm(contact, axis=1)

            rec_drift = np.max(
                np.linalg.norm(rec[:, :3] - rec[0, :3], axis=1)
            )
            rows.append(
                dict(
                    episode=k,
                    steps=len(g["actions"]),
                    states=len(tcp),
                    dx_m=c[0],
                    dy_m=c[1],
                    dyaw_rad=c[2],
                    success=bool(succ[-1]),
                    ever_success=bool(np.any(succ)),
                    max_contact_force_N=float(force.max()),
                    contact_frames=int(np.sum(force > 1e-3)),
                    receptacle_max_position_drift_m=float(rec_drift),
                )
            )

    df = pd.DataFrame(rows)
    print("source_type:", source)
    print("note:", note)
    print("\nEpisode summary:")
    print(df.to_string(index=False))

    print("\nChecks")
    print("------")
    print("episodes:", len(df))
    print("step range:", int(df.steps.min()), "to", int(df.steps.max()))
    print("success:", int(df.success.sum()), "/", len(df))
    print(
        "episodes with charger-receptacle contact:",
        int((df.contact_frames > 0).sum()),
        "/",
        len(df),
    )
    print(
        "median peak contact force [N]:",
        float(df.max_contact_force_N.median()),
    )
    print(
        "max receptacle drift [m]:",
        float(df.receptacle_max_position_drift_m.max()),
    )

    old_waypoint_signature = (
        df.steps.nunique() == 1 and int(df.steps.iloc[0]) in (79, 80)
    )
    if old_waypoint_signature:
        print(
            "\nWARNING: every episode is exactly 79/80 steps. "
            "Inspect whether this is accidentally the old waypoint dataset."
        )
    else:
        print(
            "\nPASS: trajectories do not have the fixed old 80-frame "
            "waypoint signature."
        )

    if (df.contact_frames > 0).sum() == 0:
        print(
            "WARNING: no charger-receptacle contact detected. "
            "The solver may not be reaching insertion."
        )

    out = path.with_name(path.stem + "_integrity.csv")
    df.to_csv(out, index=False)
    print("saved:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trajectory", type=Path)
    args = ap.parse_args()
    analyze(args.trajectory)


if __name__ == "__main__":
    main()
