from __future__ import annotations

"""Freeze the pitch-amplitude sweep manifest for P_sim(s) identification.

This is the identification TRAINING set for the cross-amplitude response law
``alpha_pitch(s)``. The +-10 deg amplitude is deliberately EXCLUDED (hold-out),
so a later frozen-law prediction at 10 deg is a genuine out-of-sample
interpolation (bounded by +-7.5 and +-12.5 deg), not a memorized point.

The +-10 deg tilt is an amplitude correspondence only; it does not claim the
real hole's unknown tilt axis/direction. ``usage`` marks these trajectories as
the identification set, distinct from the dev pickup-context set and from the
matched-amplitude sim_10degree set.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase_switch_symmetry_env  # noqa: F401 - geometry constants + env registration


# Pitch amplitudes in degrees. Excludes +-10 deg (the hold-out / real amplitude).
PITCH_AMPLITUDES_DEG = [0.0, 2.5, -2.5, 5.0, -5.0, 7.5, -7.5, 12.5, -12.5, 15.0, -15.0]

# Single reference pickup context for identification (the original pedestal).
# Execution context (pickup) is orthogonal to the relation intervention (tilt);
# E-invariance of the law is left as a follow-up, sanity-checked by the dev set.
REFERENCE_PICKUP_POSITION = [-0.25, -0.18, 0.28]


def _conditions():
    conditions = []
    condition_id = 0
    for deg in PITCH_AMPLITUDES_DEG:
        if abs(deg) < 1e-9:
            generator = "baseline"
            causal_delta = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            generator = "pitch"
            causal_delta = [0.0, 0.0, 0.0, 0.0, float(np.deg2rad(deg)), 0.0]
        conditions.append(
            {
                "condition_id": condition_id,
                "pickup_id": 0,
                "pickup_position": list(REFERENCE_PICKUP_POSITION),
                "generator": generator,
                "amplitude_deg": float(deg),
                "causal_delta": causal_delta,
            }
        )
        condition_id += 1
    return conditions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("real_relation_consistency/tilt_sweep/contexts.json"),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("real_relation_consistency/tilt_sweep/protocol.json"),
    )
    parser.add_argument(
        "--usage",
        default="train",
        help="Split label recorded in the manifest (train/eval/dev).",
    )
    args = parser.parse_args()

    conditions = _conditions()
    manifest = {
        "schema_version": 3,
        "env_id": "CircularPhaseSwitchSE3-v1",
        "generator_basis": ["du", "dv", "dw", "d_roll", "d_pitch", "d_yaw"],
        "usage": args.usage,
        "scope": "pitch-amplitude sweep for alpha_pitch(s) identification; +-10 deg held out",
        "holdout_amplitudes_deg": [10.0, -10.0],
        "pickup_position": list(REFERENCE_PICKUP_POSITION),
        "amplitudes_deg": list(PITCH_AMPLITUDES_DEG),
        "conditions": conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    manifest_sha256 = hashlib.sha256(args.output.read_bytes()).hexdigest()

    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "context_manifest_sha256": manifest_sha256,
        "seeds": [20260910, 20270910, 20280910],
        "retries_per_condition": 1,
        "robot_init_qpos_noise": 0.01,
        "task_anchor": [-0.15, 0.0, 0.08],
        "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "yaw_mode": "honest",
        "usage": args.usage,
        "scope": (
            "Identify alpha_pitch(s) (cross-amplitude tilt response law) from a "
            "pitch sweep that excludes +-10 deg; freeze the law; then predict the "
            "+10 deg terminal axis response against the real machine's 10.30 deg. "
            "Not exact-axis matching; real tilt axis remains unknown."
        ),
        "model": "Pdiag finite (SE(3)) = SE3SmoothFinitePDiagModel; P(s)=alpha(s) "
        "realized by C0 Exp(diag(alpha(s)) Log(C0^-1 C)) C0^-1 X0(s)",
        "failures": "Keep all attempts; missing phase6 explicitly recorded, no successful-retry selection",
        "equivalence_margin_deg": None,
    }
    with args.protocol.open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, indent=2)

    print("saved:", args.output)
    print("saved:", args.protocol)
    print("amplitudes (deg):", PITCH_AMPLITUDES_DEG)
    print("total conditions:", len(conditions))
    print("context_manifest_sha256:", manifest_sha256)


if __name__ == "__main__":
    main()
