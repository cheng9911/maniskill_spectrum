from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from transforms3d.euler import quat2euler


def analyze(path: Path, strict: bool):
    rows = []
    with h5py.File(path, "r") as data_file:
        keys = sorted(
            [key for key in data_file if key.startswith("episode_")],
            key=lambda key: int(key.split("_")[-1]),
        )
        for key in keys:
            group = data_file[key]
            force = np.linalg.norm(np.asarray(group["contact_force"])[1:], axis=1)
            terminated = np.asarray(group["terminated"], dtype=bool)
            truncated = np.asarray(group["truncated"], dtype=bool)
            done = terminated | truncated
            final_pose = np.asarray(group["peg_pose"])[-1]
            rows.append(
                dict(
                    episode=key,
                    condition_id=int(group.attrs["condition_id"]),
                    attempt_id=int(group.attrs["attempt_id"]),
                    probe_type=str(group.attrs["probe_type"]),
                    socket_yaw_deg=float(group.attrs["socket_yaw_deg"]),
                    keyed_yaw_deg=float(group.attrs["keyed_yaw_deg"]),
                    post_clear_yaw_deg=float(group.attrs["post_clear_yaw_deg"]),
                    success=bool(np.asarray(group["success"])[-1]),
                    final_key_clearance_m=float(
                        np.asarray(group["key_clearance_margin"])[-1]
                    ),
                    final_peg_yaw_deg=float(np.rad2deg(quat2euler(final_pose[3:])[2])),
                    max_contact_force_N=float(force.max()),
                    contact_frames=int(np.sum(force > 1e-3)),
                    stop_reason=str(group.attrs.get("stop_reason", "")),
                    solver_error=str(group.attrs.get("solver_error", "")),
                    post_done_actions=int(
                        0
                        if not np.any(done)
                        else len(done) - 1 - int(np.flatnonzero(done)[0])
                    ),
                )
            )
    frame = pd.DataFrame(rows)
    selected = []
    for condition_id, condition_rows in frame.groupby("condition_id"):
        probe_type = condition_rows.probe_type.iloc[0]
        if probe_type == "matched_preclear":
            candidates = condition_rows[condition_rows.final_key_clearance_m > 0.002]
        elif probe_type == "mismatched_preclear":
            candidates = condition_rows[
                (condition_rows.final_key_clearance_m <= 0.0)
                & (condition_rows.contact_frames > 0)
            ]
        else:
            candidates = condition_rows[condition_rows.success]
        if not candidates.empty:
            selected.append(candidates.iloc[0])
    selected_frame = pd.DataFrame(selected)

    condition_count = int(frame.condition_id.nunique())
    matched = selected_frame[selected_frame.probe_type == "matched_preclear"]
    mismatched = selected_frame[selected_frame.probe_type == "mismatched_preclear"]
    postclear = selected_frame[selected_frame.probe_type == "arbitrary_postclear_yaw"]
    yaw_tracking_error = np.abs(postclear.final_peg_yaw_deg - postclear.post_clear_yaw_deg)
    checks = {
        "all_probe_conditions_validated": len(selected_frame) == condition_count == 6,
        "matched_key_clears": len(matched) == 2
        and bool((matched.final_key_clearance_m > 0.002).all()),
        "mismatched_key_blocked_by_contact": len(mismatched) == 2
        and bool((mismatched.final_key_clearance_m <= 0.0).all())
        and bool((mismatched.contact_frames > 0).all()),
        "postclear_arbitrary_yaw_succeeds": len(postclear) == 2
        and bool(postclear.success.all()),
        "postclear_yaw_is_physically_realized": len(postclear) == 2
        and bool((yaw_tracking_error < 5.0).all()),
        "no_post_terminal_actions": bool((frame.post_done_actions == 0).all()),
    }
    summary = {
        "trajectory": str(path),
        "episodes": len(frame),
        "conditions": condition_count,
        "selected_valid_probes": len(selected_frame),
        "matched_clearance_m": matched.final_key_clearance_m.tolist(),
        "mismatched_clearance_m": mismatched.final_key_clearance_m.tolist(),
        "mismatched_contact_force_N": mismatched.max_contact_force_N.tolist(),
        "postclear_final_yaw_deg": postclear.final_peg_yaw_deg.tolist(),
        "checks": checks,
    }
    stem = path.stem
    frame.to_csv(path.with_name(f"{stem}_integrity.csv"), index=False)
    selected_frame.to_csv(path.with_name(f"{stem}_selected_probes.csv"), index=False)
    summary_path = path.with_name(f"{stem}_validation_summary.json")
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    print(json.dumps(summary, indent=2))
    print("saved:", summary_path)
    if strict and not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit("failed geometry probe checks: " + ", ".join(failed))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    analyze(args.trajectory, args.strict)


if __name__ == "__main__":
    main()
