from __future__ import annotations

"""Generate the frozen planar-push 6-vector context manifest (own module).

Reuses the FROZEN SE(3) intervention rows (``intervention_rows_se3`` from
collect_phase_switch_rotated.py) so the push task's 75 conditions — 1 baseline
+ 14 isolated + 60 mixed — are the SAME 6-vectors as the peg-in-hole task's
manifest. Identical interventions across two different tasks is exactly what the
cross-task out-of-plane contrast needs.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import phase_switch_symmetry_planar_push_env as pp
from collect_phase_switch_rotated import intervention_rows_se3

PUSH_GEOMETRY = {
    "BLOCK_HALF_X": pp.BLOCK_HALF_X,
    "BLOCK_HALF_Y": pp.BLOCK_HALF_Y,
    "BLOCK_HALF_Z": pp.BLOCK_HALF_Z,
    "TARGET_POS": pp.TARGET_POS.tolist(),
    "BLOCK_START": pp.BLOCK_START.tolist(),
    "PUSH_SUCCESS_POS_TOL": pp.PUSH_SUCCESS_POS_TOL,
    "PUSH_SUCCESS_YAW_TOL": pp.PUSH_SUCCESS_YAW_TOL,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixed-samples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "phase_switch_symmetry_multiseed/planar_push/planar_push_fixed_contexts.json"
        ),
    )
    args = parser.parse_args()

    rows = intervention_rows_se3(args.mixed_samples, args.seed)
    conditions = []
    for condition_id, row in enumerate(rows):
        cd = np.asarray(row["causal_delta"], dtype=np.float64)
        if cd.shape != (6,) or not np.isfinite(cd).all():
            raise ValueError(f"row {condition_id} is not a finite 6-vector")
        conditions.append(
            {
                "condition_id": condition_id,
                "generator": row["generator"],
                "causal_delta": cd.tolist(),
            }
        )

    manifest = {
        "schema_version": 2,
        "env_id": "PlanarPush-v1",
        "generator_basis": ["du", "dv", "dw", "d_roll", "d_pitch", "d_yaw"],
        "generator_scale": {
            "translation_m": 0.012,
            "roll_pitch_deg": 15.0,
            "yaw_deg": 30.0,
        },
        "mixed_samples": args.mixed_samples,
        "seed": args.seed,
        "geometry": PUSH_GEOMETRY,
        "conditions": conditions,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("saved:", args.output, "conditions:", len(conditions))


if __name__ == "__main__":
    main()
