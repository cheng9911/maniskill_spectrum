from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def analyze(path: Path):
    rows = []
    with h5py.File(path, "r") as f:
        task = f.attrs.get("task", "")
        keys = sorted([k for k in f if k.startswith("episode_")], key=lambda x: int(x.split("_")[-1]))
        for k in keys:
            g = f[k]
            force = np.linalg.norm(np.asarray(g["contact_force"]), axis=1)
            socket = np.asarray(g["socket_pose"])
            rows.append(
                dict(
                    episode=k,
                    generator=g.attrs.get("generator", ""),
                    steps=len(g["actions"]),
                    states=len(g["peg_pose"]),
                    success=bool(np.asarray(g["success"])[-1]),
                    max_contact_force_N=float(force.max()),
                    contact_frames=int(np.sum(force > 1e-3)),
                    final_dist_m=float(np.asarray(g["obj_to_goal_dist"])[-1]),
                    final_yaw_err_rad=float(np.asarray(g["yaw_err"])[-1]),
                    socket_drift_m=float(np.max(np.linalg.norm(socket[:, :3] - socket[0, :3], axis=1))),
                    solver_error=g.attrs.get("solver_error", ""),
                    dx=float(np.asarray(g["causal_delta"])[0]),
                    dy=float(np.asarray(g["causal_delta"])[1]),
                    dyaw=float(np.asarray(g["causal_delta"])[2]),
                )
            )
    df = pd.DataFrame(rows)
    print("task:", task)
    print(df.to_string(index=False))
    print("\nChecks")
    print("------")
    print("episodes:", len(df))
    print("step range:", int(df.steps.min()), "to", int(df.steps.max()))
    print("success:", int(df.success.sum()), "/", len(df))
    print("contact episodes:", int((df.contact_frames > 0).sum()), "/", len(df))
    print("max socket drift [m]:", float(df.socket_drift_m.max()))
    print("solver errors:", int((df.solver_error.astype(str) != "").sum()))
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
