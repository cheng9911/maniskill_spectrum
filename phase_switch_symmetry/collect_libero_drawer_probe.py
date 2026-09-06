from __future__ import annotations

"""Collect a cheap controlled LIBERO/robosuite drawer relation probe.

This is intentionally not a policy-learning collector.  It loads the real LIBERO
``open_the_middle_drawer_of_the_cabinet`` MuJoCo model, then generates a small
counterfactual grid by directly setting the cabinet's prismatic drawer joint.
The recorded task curve is the drawer pose expressed in a rail-aligned task
frame, while the raw LIBERO world body pose is stored alongside it for audit.

Generator basis: [du, dv, dw, roll, pitch, yaw].
Oracle relation: only du (translation along the drawer rail) is tracked; the
remaining five generators are suppressed by the prismatic mechanism.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np


SEEDS = [20260818, 20270818, 20280818]
GENERATOR_BASIS = ["du", "dv", "dw", "d_roll", "d_pitch", "d_yaw"]
PHASE_CODES = (3, 4, 5, 6)
PHASE_NAMES = ("reach", "slide_open", "hold_open", "retract")
CONTEXT_SCALE = np.array(
    [0.012, 0.012, 0.012, np.deg2rad(15.0), np.deg2rad(15.0), np.deg2rad(30.0)],
    dtype=np.float64,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def matrix_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rx @ ry @ rz


def quat_wxyz_from_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        return np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ],
            dtype=np.float64,
        )
    diag = np.diag(matrix)
    axis = int(np.argmax(diag))
    if axis == 0:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        quat = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
            ]
        )
    elif axis == 1:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        quat = np.array(
            [
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
            ]
        )
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        quat = np.array(
            [
                (matrix[1, 0] - matrix[0, 1]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
            ]
        )
    return quat.astype(np.float64) / np.linalg.norm(quat)


def se3_from_pose6(pose6: np.ndarray) -> np.ndarray:
    pose6 = np.asarray(pose6, dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = matrix_from_euler(pose6[3], pose6[4], pose6[5])
    transform[:3, 3] = pose6[:3]
    return transform


def pose_from_se3(transform: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [transform[:3, 3], quat_wxyz_from_matrix(transform[:3, :3])]
    )


def intervention_rows_se3(n_mixed: int, seed: int) -> list[dict]:
    """Same 6D intervention design as the frozen SE(3)/planar manifests."""
    deg = np.deg2rad
    rows = [dict(generator="baseline", causal_delta=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
    rows.extend(dict(generator="du", causal_delta=[d, 0.0, 0.0, 0.0, 0.0, 0.0]) for d in (-0.015, 0.015))
    rows.extend(dict(generator="dv", causal_delta=[0.0, d, 0.0, 0.0, 0.0, 0.0]) for d in (-0.015, 0.015))
    rows.extend(dict(generator="dw", causal_delta=[0.0, 0.0, d, 0.0, 0.0, 0.0]) for d in (-0.015, 0.015))
    rows.extend(dict(generator="roll", causal_delta=[0.0, 0.0, 0.0, deg(a), 0.0, 0.0]) for a in (-15.0, 15.0))
    rows.extend(dict(generator="pitch", causal_delta=[0.0, 0.0, 0.0, 0.0, deg(a), 0.0]) for a in (-15.0, 15.0))
    rows.extend(dict(generator="yaw", causal_delta=[0.0, 0.0, 0.0, 0.0, 0.0, deg(a)]) for a in (-30.0, -15.0, 15.0, 30.0))
    rng = np.random.default_rng(seed)
    for _ in range(n_mixed):
        rows.append(
            dict(
                generator="mixed",
                causal_delta=[
                    float(rng.uniform(-0.012, 0.012)),
                    float(rng.uniform(-0.012, 0.012)),
                    float(rng.uniform(-0.012, 0.012)),
                    float(rng.uniform(deg(-15.0), deg(15.0))),
                    float(rng.uniform(deg(-15.0), deg(15.0))),
                    float(rng.uniform(deg(-30.0), deg(30.0))),
                ],
            )
        )
    return rows


def write_preregistration(output_root: Path, rows: list[dict], args) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    conditions = []
    for condition_id, row in enumerate(rows):
        causal_delta = np.asarray(row["causal_delta"], dtype=np.float64)
        conditions.append(
            {
                "condition_id": condition_id,
                "generator": row["generator"],
                "causal_delta": causal_delta.tolist(),
            }
        )

    manifest_path = output_root / "libero_drawer_fixed_contexts.json"
    manifest = {
        "schema_version": 1,
        "env_id": "LIBERO/robosuite open_the_middle_drawer_of_the_cabinet",
        "generator_basis": GENERATOR_BASIS,
        "generator_scale": {
            "translation_m": 0.012,
            "roll_pitch_deg": 15.0,
            "yaw_deg": 30.0,
        },
        "mixed_samples": args.mixed_samples,
        "seed": args.context_seed,
        "conditions": conditions,
    }
    with manifest_path.open("w", encoding="utf-8") as output_file:
        json.dump(manifest, output_file, indent=2)

    experiment_path = output_root / "libero_drawer_experiment.json"
    experiment = {
        "schema_version": 1,
        "status": "preregistered_before_fitting",
        "title": "LIBERO drawer prismatic-sliding relation probe",
        "context_manifest": str(manifest_path.resolve()),
        "context_manifest_sha256": sha256(manifest_path),
        "env_id": "LIBERO/robosuite open_the_middle_drawer_of_the_cabinet",
        "task": "drawer/prismatic sliding",
        "generator_basis": GENERATOR_BASIS,
        "oracle_selector": [1, 0, 0, 0, 0, 0],
        "oracle_note": (
            "Only du, the rail-aligned drawer translation, changes the drawer "
            "state. dv/dw/roll/pitch/yaw are present in the ghost target context "
            "but are suppressed by the prismatic joint."
        ),
        "pose_frame": (
            "object_pose is the LIBERO drawer body expressed in a rail-aligned "
            "task frame; libero_body_pose_world stores the raw MuJoCo body pose."
        ),
        "phases": list(PHASE_NAMES),
        "phase_codes": list(PHASE_CODES),
        "active_phases": ["slide_open", "hold_open"],
        "nominal_open_displacement_m": args.nominal_open,
        "closed_joint_qpos": args.closed_qpos,
        "seeds": SEEDS,
    }
    with experiment_path.open("w", encoding="utf-8") as output_file:
        json.dump(experiment, output_file, indent=2)
    return manifest_path, experiment_path


def make_libero_env(camera_size: int):
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils import get_libero_path

    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    task = suite.get_task(0)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=camera_size,
        camera_widths=camera_size,
    )
    return env, str(bddl)


def collect_seed(path: Path, rows: list[dict], seed: int, args) -> None:
    env, bddl = make_libero_env(args.camera_size)
    env.seed(seed)
    env.reset()
    sim = env.sim
    model = sim.model
    data = sim.data
    joint_id = model.joint_name2id("wooden_cabinet_1_middle_level")
    qpos_addr = int(model.jnt_qposadr[joint_id])
    body_id = model.body_name2id("wooden_cabinet_1_cabinet_middle")
    joint_range = np.asarray(model.jnt_range[joint_id], dtype=np.float64)

    nominal_target_pose = pose_from_se3(
        se3_from_pose6(np.array([args.nominal_open, 0.0, 0.0, 0.0, 0.0, 0.0]))
    )
    started = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as data_file:
        data_file.attrs["schema_version"] = 1
        data_file.attrs["env_id"] = "LIBERO/robosuite open_the_middle_drawer_of_the_cabinet"
        data_file.attrs["bddl_file"] = bddl
        data_file.attrs["seed"] = seed
        data_file.attrs["pose_frame"] = "rail_aligned_task_frame"
        data_file.attrs["nominal_target_pose"] = nominal_target_pose
        data_file.attrs["closed_joint_qpos"] = args.closed_qpos
        data_file.attrs["joint_range"] = joint_range
        data_file.attrs["phase_codes"] = PHASE_CODES
        data_file.attrs["phase_names"] = PHASE_NAMES

        for condition_id, row in enumerate(rows):
            causal_delta = np.asarray(row["causal_delta"], dtype=np.float64)
            target_displacement = args.nominal_open + float(causal_delta[0])
            target_qpos = args.closed_qpos - target_displacement
            if target_qpos < joint_range[0] or target_qpos > joint_range[1]:
                raise RuntimeError(
                    f"condition {condition_id} target qpos {target_qpos:.4f} "
                    f"outside joint range {joint_range}"
                )
            target_pose = pose_from_se3(
                se3_from_pose6(
                    np.array([args.nominal_open, 0.0, 0.0, 0.0, 0.0, 0.0])
                )
                @ se3_from_pose6(causal_delta)
            )

            phases = []
            object_pose = []
            libero_body_pose_world = []
            joint_qpos = []
            rail_displacement = []
            for phase_code in PHASE_CODES:
                phase_u = np.linspace(0.0, 1.0, args.steps_per_phase)
                if phase_code == 3:
                    displacement_values = np.zeros(args.steps_per_phase)
                elif phase_code == 4:
                    displacement_values = target_displacement * phase_u
                else:
                    displacement_values = np.full(args.steps_per_phase, target_displacement)
                for displacement in displacement_values:
                    qpos = args.closed_qpos - float(displacement)
                    data.qpos[qpos_addr] = qpos
                    sim.forward()
                    phases.append(phase_code)
                    joint_qpos.append(qpos)
                    rail_displacement.append(displacement)
                    object_pose.append([displacement, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
                    libero_body_pose_world.append(
                        np.concatenate([data.xpos[body_id], data.xquat[body_id]])
                    )

            group = data_file.create_group(f"episode_{condition_id:04d}")
            group.attrs["condition_id"] = condition_id
            group.attrs["generator"] = row["generator"]
            group.attrs["seed"] = seed
            group.attrs["target_joint_qpos"] = target_qpos
            group.create_dataset("causal_delta", data=causal_delta)
            group.create_dataset("target_pose", data=np.tile(target_pose, (len(phases), 1)))
            group.create_dataset("object_pose", data=np.asarray(object_pose, dtype=np.float64))
            group.create_dataset(
                "libero_body_pose_world",
                data=np.asarray(libero_body_pose_world, dtype=np.float64),
            )
            group.create_dataset("joint_qpos", data=np.asarray(joint_qpos, dtype=np.float64))
            group.create_dataset(
                "rail_displacement", data=np.asarray(rail_displacement, dtype=np.float64)
            )
            group.create_dataset("solver_phase", data=np.asarray(phases, dtype=np.int32))
            success = np.zeros(len(phases), dtype=bool)
            success[-1] = True
            group.create_dataset("success", data=success)
            group.create_dataset("truncated", data=np.zeros(len(phases), dtype=bool))

    env.close()
    print(f"saved {path} episodes={len(rows)} seconds={time.time() - started:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/libero_drawer"),
    )
    parser.add_argument(
        "--rollout-root",
        type=Path,
        default=Path("phase_switch_symmetry_rollouts_libero_drawer"),
    )
    parser.add_argument("--context-seed", type=int, default=20260818)
    parser.add_argument("--mixed-samples", type=int, default=60)
    parser.add_argument("--nominal-open", type=float, default=0.10)
    parser.add_argument("--closed-qpos", type=float, default=0.0)
    parser.add_argument("--steps-per-phase", type=int, default=12)
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    args = parser.parse_args()

    rows = intervention_rows_se3(args.mixed_samples, args.context_seed)
    manifest_path, experiment_path = write_preregistration(args.output_root, rows, args)
    print("context_manifest:", manifest_path)
    print("experiment:", experiment_path)
    for seed in args.seeds:
        collect_seed(args.rollout_root / f"drawer_seed_{seed}.h5", rows, seed, args)


if __name__ == "__main__":
    main()
