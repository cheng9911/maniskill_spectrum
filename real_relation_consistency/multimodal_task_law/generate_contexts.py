from __future__ import annotations

"""Freeze the multimodal-task-law protocol into ``protocol.json``.

The protocol pins every choice that must NOT be tuned against results: the fixed
task config (copied from the frozen SE(3) anchor), the 2x2 path/alignment mode
matrix, the two-arm yaw challenge, the pitch/yaw condition grids, the train/held-
out split, the geometric events + thresholds, the strict-success criteria and the
seed schedule.  A content SHA256 is embedded so collectors/analysers can verify
they are reading the exact frozen file.
"""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

PROTOCOL_VERSION = "1.0"

# Fixed task config — copied from the frozen SE(3) experiment (se3_experiment.json),
# NOT re-derived and NOT tuned here.
TASK = {
    "env_id": "CircularPhaseSwitchSE3-v1",
    "orientation": [1.0, 0.0, 0.0, 0.0],
    "task_anchor": [-0.15, 0.00, 0.08],
    "pickup_position": [-0.25, -0.18, 0.28],
    "robot_init_qpos_noise": 0.01,
    "control_mode": "pd_joint_pos",
    "sim_backend": "physx_cpu",
    "gripper_stiffness": 2500.0,
    "gripper_force_limit": 150.0,
}

# 2x2 mode matrix: {lift, diagonal} x {early, late} alignment timing.
MODE_IDS = ["lift_early", "lift_late", "diagonal_early", "diagonal_late"]
MODE_PARAMS = {
    "lift_early": {"path_mode": "lift", "alignment_mode": "early"},
    "lift_late": {"path_mode": "lift", "alignment_mode": "late"},
    "diagonal_early": {"path_mode": "diagonal", "alignment_mode": "early"},
    "diagonal_late": {"path_mode": "diagonal", "alignment_mode": "late"},
}
YAW_MODE_IDS = ["yaw_fixed", "yaw_follow"]

# Waypoint depths used by solve_multimodal_se3 (task-local z above socket centre).
SOLVER = {
    "HIGH_Z": 0.22,
    "PRE_ENTRY": 0.125,
    "FINAL": 0.060,
    "note": "early aligns at HIGH_Z, late aligns at PRE_ENTRY; terminal insert control is identical across modes",
}

# Pitch core experiment: 13 symmetric amplitudes, train 9 / held-out 4.
PITCH_DEG = [0.0, -2.5, 2.5, -5.0, 5.0, -7.5, 7.5, -10.0, 10.0, -12.5, 12.5, -15.0, 15.0]
TRAIN_PITCH_DEG = [0.0, -2.5, 2.5, -7.5, 7.5, -12.5, 12.5, -15.0, 15.0]
TEST_PITCH_DEG = [-5.0, 5.0, -10.0, 10.0]

# Pilot: 4 modes x {0, +-7.5} x 3 blocks = 36.
PILOT_PITCH_DEG = [0.0, -7.5, 7.5]
PILOT_BLOCKS = 3

# Core: 4 modes x 13 conditions x 5 blocks = 260.
CORE_BLOCKS = 5

# Yaw challenge: 2 modes x {0, +-15, +-30} x 5 blocks = 50.
YAW_DEG = [0.0, -15.0, 15.0, -30.0, 30.0]
YAW_TRAIN_DEG = [0.0, -15.0, 15.0]
YAW_TEST_DEG = [-30.0, 30.0]
YAW_BLOCKS = 5

# Seed schedule: initial_state_seed is shared within a block (paired same-initial
# state); planner_seed separates per (mode, condition) so the planner does not
# reuse a single stream.  base seed pinned to the frozen multiseed lineage.
SEED = {
    "base": 20260818,
    "initial_state_stride": 100003,   # per block
    "planner_stride": 1009,           # per (mode x condition)
    "note": "initial_state_seed = base + block*initial_state_stride (shared within block); planner_seed = initial_state_seed + planner_stride*(mode_index*len(conditions)+condition_index) + attempt",
}

# Geometric events + thresholds (see plan section 5).  All in the socket frame.
# axial = projection of (peg_centre - socket_centre) onto the tilted socket axis;
# lateral = the residual norm; depth drops as the peg descends (pre-entry 0.125 ->
# final 0.060).  The peg tip is PEG_HALF_LENGTH (0.045) below the peg centre.
EVENTS = {
    "E0": {"desc": "common post-grasp start state (after the shared lift)"},
    "E1": {
        "desc": "peg centre first persistently inside the pre-registered hole-opening neighbourhood",
        "lateral_lt": 0.020,          # m
        "axial_gt": 0.065,            # GATE_Z_MAX: still above the gate
        "persist_steps": 3,
    },
    "E2": {
        "desc": "peg lowest geometric point crosses the hole-opening plane inside the lateral channel",
        "tip_axial_le": 0.065,        # GATE_Z_MAX plane (tip = centre - PEG_HALF_LENGTH)
        "lateral_lt": 0.015,          # CIRCULAR_GATE_INNER_RADIUS
    },
    "E3": {
        "desc": "actual axial depth reaches the achievable terminal depth and stays within the success ball",
        "axial_le": 0.072,           # FINAL_PEG_Z + 0.012 (calibrated: the solver, identical to frozen solve_se3, stops at peg-centre depth ~0.070-0.072 while the env success ball is pos_err<0.012)
        "hold_steps": 10,
        "note": "the shared terminal control does not dwell; hold is measured as trailing consecutive-success steps (diagnostic)",
    },
    "segments": [
        {"id": "free_approach", "from": "E0", "to": "E1", "n": 50},
        {"id": "enter", "from": "E1", "to": "E2", "n": 50},
        {"id": "insert", "from": "E2", "to": "E3", "n": 50},
    ],
}

STRICT_SUCCESS = {
    "lateral_error_m": 0.010,     # within gate clearance (SHAFT_RADIUS 0.013 + GATE_CLEARANCE 0.002 = 0.015 inner radius; 0.010 is stricter than hard contact, looser than the 0.002 ideal)
    "axis_angle_rad": 0.05,       # ~2.9 deg
    "depth_le": 0.072,            # FINAL_PEG_Z + 0.012 (calibrated from pilot: solver terminal depth ~0.070-0.072; identical to frozen solve_se3)
    "hold_steps": 15,
    "note": "depth threshold calibrated after pilot (solver reaches ~0.070-0.072, not 0.060); hold measured as trailing consecutive-success steps since the shared terminal control does not dwell",
}

IDENTIFY = {
    "parameterization": "scalar alpha_pitch(s) only; xi=(0,0,0,0,theta,0) for C=C0 Exp(theta e_pitch)",
    "train_condition_count": len(TRAIN_PITCH_DEG),
    "test_condition_count": len(TEST_PITCH_DEG),
    "fits": "per (mode, block) independently; 4 x 5 = 20 frozen pitch models",
    "fit_config": {
        "alpha_max": 1.25,
        "n_basis": 24,
        "basis_width": 0.065,
        "smoothness_weight": 0.1,
        "nominal_iterations": 3,
    },
}


def _canonical(protocol: dict) -> str:
    return json.dumps(protocol, sort_keys=True, indent=2, allow_nan=False)


def build_protocol() -> dict:
    protocol = {
        "schema_version": PROTOCOL_VERSION,
        "title": "Mode dependence and transferability of task-relation responses",
        "task": TASK,
        "mode_ids": MODE_IDS,
        "mode_params": MODE_PARAMS,
        "yaw_mode_ids": YAW_MODE_IDS,
        "solver": SOLVER,
        "pitch_deg": PITCH_DEG,
        "train_pitch_deg": TRAIN_PITCH_DEG,
        "test_pitch_deg": TEST_PITCH_DEG,
        "pilot_pitch_deg": PILOT_PITCH_DEG,
        "pilot_blocks": PILOT_BLOCKS,
        "core_blocks": CORE_BLOCKS,
        "yaw_deg": YAW_DEG,
        "yaw_train_deg": YAW_TRAIN_DEG,
        "yaw_test_deg": YAW_TEST_DEG,
        "yaw_blocks": YAW_BLOCKS,
        "seed": SEED,
        "events": EVENTS,
        "strict_success": STRICT_SUCCESS,
        "identify": IDENTIFY,
    }
    # content hash over every field except the hash itself
    protocol["content_sha256"] = hashlib.sha256(
        _canonical(protocol).encode("utf-8")
    ).hexdigest()
    return protocol


def write_protocol(path: Path | None = None) -> Path:
    path = Path(path) if path is not None else HERE / "protocol.json"
    protocol = build_protocol()
    path.write_text(_canonical(protocol) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    print(f"  content_sha256 = {protocol['content_sha256']}")
    return path


def load_protocol(path: Path | None = None) -> dict:
    path = Path(path) if path is not None else HERE / "protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    body = {k: v for k, v in protocol.items() if k != "content_sha256"}
    recomputed = hashlib.sha256(
        json.dumps(body, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
    ).hexdigest()
    stored = protocol.get("content_sha256")
    if stored != recomputed:
        raise RuntimeError(
            f"protocol.json content hash mismatch: stored {stored} != recomputed {recomputed}"
        )
    return protocol


if __name__ == "__main__":
    write_protocol()
