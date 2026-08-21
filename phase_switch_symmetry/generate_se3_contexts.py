from __future__ import annotations

"""Freeze the SE(3) 6-generator intervention manifest (se3_fixed_contexts.json).

Mirror of the SE(2) fixed_contexts.json discipline: contiguous condition_ids,
one generator label per row, a 6-vector causal_delta, and a geometry block. The
manifest is the single source of truth for both collection and the rank-6 subset
generator, so it is written once and then only read (sha256-enforced).
"""

import argparse
import json
from pathlib import Path

import numpy as np

import phase_switch_symmetry_env  # noqa: F401 - geometry constants + env registration
from collect_phase_switch_rotated import intervention_rows_se3


SE3_GEOMETRY = {
    "BORE_INNER_RADIUS": phase_switch_symmetry_env.BORE_INNER_RADIUS,
    "FINAL_PEG_Z": phase_switch_symmetry_env.FINAL_PEG_Z,
    "GATE_CLEARANCE": phase_switch_symmetry_env.GATE_CLEARANCE,
    "GATE_Z_MAX": phase_switch_symmetry_env.GATE_Z_MAX,
    "GATE_Z_MIN": phase_switch_symmetry_env.GATE_Z_MIN,
    "KEY_CLEAR_PEG_Z": phase_switch_symmetry_env.KEY_CLEAR_PEG_Z,
    "KEY_HALF_X": phase_switch_symmetry_env.KEY_HALF_X,
    "KEY_HALF_Y": phase_switch_symmetry_env.KEY_HALF_Y,
    "KEY_Z_MAX": phase_switch_symmetry_env.KEY_Z_MAX,
    "KEY_Z_MIN": phase_switch_symmetry_env.KEY_Z_MIN,
    "PEG_HALF_LENGTH": phase_switch_symmetry_env.PEG_HALF_LENGTH,
    "PRE_ENTRY_PEG_Z": phase_switch_symmetry_env.PRE_ENTRY_PEG_Z,
    "SHAFT_RADIUS": phase_switch_symmetry_env.SHAFT_RADIUS,
    "SOCKET_OUTER_RADIUS": phase_switch_symmetry_env.SOCKET_OUTER_RADIUS,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixed-samples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/se3_fixed_contexts.json"),
    )
    args = parser.parse_args()

    rows = intervention_rows_se3(args.mixed_samples, args.seed)
    conditions = []
    for condition_id, row in enumerate(rows):
        causal_delta = np.asarray(row["causal_delta"], dtype=np.float64)
        if causal_delta.shape != (6,) or not np.isfinite(causal_delta).all():
            raise ValueError(f"row {condition_id} is not a finite 6-vector")
        conditions.append(
            {
                "condition_id": condition_id,
                "generator": row["generator"],
                "causal_delta": causal_delta.tolist(),
            }
        )

    manifest = {
        "schema_version": 2,
        "env_id": "KeyedCircularPhaseSwitchSE3-v1",
        "generator_basis": ["du", "dv", "dw", "d_roll", "d_pitch", "d_yaw"],
        "generator_scale": {
            "translation_m": 0.012,
            "roll_pitch_deg": 15.0,
            "yaw_deg": 30.0,
        },
        "mixed_samples": args.mixed_samples,
        "seed": args.seed,
        "geometry": SE3_GEOMETRY,
        "conditions": conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print("saved:", args.output)
    print("conditions:", len(conditions))
    from collections import Counter

    print("by generator:", dict(Counter(c["generator"] for c in conditions)))


if __name__ == "__main__":
    main()
