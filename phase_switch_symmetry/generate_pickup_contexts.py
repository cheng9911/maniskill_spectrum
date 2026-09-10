from __future__ import annotations

"""Freeze the pickup-context x relation-intervention (E x c) manifest.

Matched sim-to-real protocol: pickup position is an EXECUTION CONTEXT
(initial-condition variation), orthogonal to the hole RELATION intervention
``causal_delta``. Each condition is the full cross product of pickup_positions
x tilt_conditions, with contiguous condition_ids, so the collector runs the
grid as a single manifest (sha256-enforced).

The default tilt set mirrors the real machine's two conditions (vertical + one
10 deg tilted hole). The +10 deg pitch tilt is an amplitude correspondence only;
it does not claim the real hole's unknown tilt axis/direction. ``usage`` marks
these trajectories as development/evaluation, NOT the identification training
set, so a later hold-out of +10 deg is not contaminated by this collection.
"""

import argparse
import json
from pathlib import Path

import numpy as np

import phase_switch_symmetry_env  # noqa: F401 - geometry constants + env registration


# Default pickup pedestal positions [x, y, z_top]. z_top is pinned to
# PEDESTAL_POS[2]; only x/y are open. The first is the original pedestal.
# Socket horizontal center for the default task_anchor is [-0.15, 0].
DEFAULT_PICKUP_POSITIONS = [
    [-0.25, -0.18, 0.28],
    [-0.30, -0.02, 0.28],
    [-0.20, -0.30, 0.28],
]

# Minimum horizontal clearance from the socket center to a pickup pedestal:
# socket outer radius 0.07 + pedestal half-width 0.03.
MIN_CLEARANCE = 0.10


# World horizontal position of the socket for the default collection setup. This
# MUST match the collector's ``--task-anchor`` ([-0.15, 0, 0.08]). The env's
# SOCKET_CENTER ([0.10, 0, 0]) is only the __init__ fallback and is overridden
# by every collector in this repo, so it is NOT the socket's world position here.
DEFAULT_SOCKET_CENTER = np.array([-0.15, 0.0], dtype=np.float64)


def _socket_horizontal_center():
    return DEFAULT_SOCKET_CENTER.copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("real_relation_consistency/pickup_contexts/contexts.json"),
    )
    parser.add_argument(
        "--tilt-deg",
        type=float,
        default=10.0,
        help="Magnitude of the single tilted-hole intervention (deg).",
    )
    parser.add_argument(
        "--usage",
        default="dev",
        help="Split label recorded in the manifest (dev/eval/train).",
    )
    parser.add_argument(
        "--socket-center",
        nargs=2,
        type=float,
        default=None,
        help="Socket horizontal center [x, y] for the clearance check "
        "(default: collector task_anchor [-0.15, 0]).",
    )
    args = parser.parse_args()

    z_top = phase_switch_symmetry_env.PEDESTAL_POS[2]
    pickup_positions = [
        dict(pickup_id=i, position=[float(p[0]), float(p[1]), float(z_top)])
        for i, p in enumerate(DEFAULT_PICKUP_POSITIONS)
    ]

    tilt_conditions = [
        dict(generator="baseline", causal_delta=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        dict(
            generator="pitch",
            causal_delta=[0.0, 0.0, 0.0, 0.0, float(np.deg2rad(args.tilt_deg)), 0.0],
        ),
    ]

    conditions = []
    condition_id = 0
    for pickup in pickup_positions:
        for tilt in tilt_conditions:
            conditions.append(
                {
                    "condition_id": condition_id,
                    "pickup_id": pickup["pickup_id"],
                    "pickup_position": pickup["position"],
                    "generator": tilt["generator"],
                    "causal_delta": tilt["causal_delta"],
                }
            )
            condition_id += 1

    socket_center = (
        _socket_horizontal_center()
        if args.socket_center is None
        else np.asarray(args.socket_center, dtype=np.float64)
    )
    clearances = {}
    for pickup in pickup_positions:
        xy = np.asarray(pickup["position"][:2], dtype=np.float64)
        clearance = float(np.linalg.norm(xy - socket_center))
        clearances[pickup["pickup_id"]] = clearance
        flag = "" if clearance >= MIN_CLEARANCE else "  <-- BELOW MIN CLEARANCE"
        print(
            f"pickup {pickup['pickup_id']}: {pickup['position']} "
            f"clearance {clearance:.3f} m{flag}"
        )

    manifest = {
        "schema_version": 3,
        "env_id": "CircularPhaseSwitchSE3-v1",
        "generator_basis": ["du", "dv", "dw", "d_roll", "d_pitch", "d_yaw"],
        "usage": args.usage,
        "socket_center": socket_center.tolist(),
        "min_clearance_m": MIN_CLEARANCE,
        "pickup_positions": pickup_positions,
        "tilt_conditions": tilt_conditions,
        "conditions": conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print("saved:", args.output)
    print("pickup positions:", len(pickup_positions))
    print("tilt conditions:", len(tilt_conditions))
    print("total conditions:", len(conditions))


if __name__ == "__main__":
    main()
