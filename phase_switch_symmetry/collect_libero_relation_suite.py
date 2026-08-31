from __future__ import annotations

"""Collect controlled LIBERO/robosuite relation-suite trajectories.

The suite deliberately regenerates state-level counterfactual rollouts under a
known 6D context, instead of consuming uncontrolled LIBERO demos.  Each task
loads its real LIBERO MuJoCo scene for body/joint provenance, while the scored
``object_pose`` is expressed in the task-local frame declared by the spec.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np

from collect_libero_drawer_probe import (
    intervention_rows_se3,
    pose_from_se3,
    se3_from_pose6,
)
from libero_relation_suite_specs import (
    GENERATOR_BASIS,
    LIBERO_RELATION_SPECS,
    PHASE_CODES,
    PHASE_NAMES,
    SEEDS,
    specs_by_key,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_libero_env(spec, camera_size: int):
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils import get_libero_path

    suite = benchmark.get_benchmark_dict()[spec.suite_name]()
    task = suite.get_task(spec.task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=camera_size,
        camera_widths=camera_size,
    )
    return env, str(bddl), task.name


def selected_delta(spec, causal_delta: np.ndarray) -> np.ndarray:
    selector = np.asarray(spec.oracle_selector, dtype=np.float64)
    return selector * np.asarray(causal_delta, dtype=np.float64)


def target_pose(spec, causal_delta: np.ndarray) -> np.ndarray:
    return pose_from_se3(se3_from_pose6(spec.nominal_pose6) @ se3_from_pose6(causal_delta))


def object_pose_at(spec, causal_delta: np.ndarray, alpha: float) -> np.ndarray:
    nominal = np.asarray(spec.nominal_pose6, dtype=np.float64)
    pose6 = alpha * (nominal + selected_delta(spec, causal_delta))
    return pose_from_se3(se3_from_pose6(pose6))


def set_free_body(sim, spec, pose: np.ndarray, initial_qpos: np.ndarray) -> None:
    model = sim.model
    data = sim.data
    if spec.free_joint_name is None:
        return
    joint_id = model.joint_name2id(spec.free_joint_name)
    qpos_addr = int(model.jnt_qposadr[joint_id])
    world_pose = initial_qpos.copy()
    world_pose[:3] = initial_qpos[:3] + pose[:3]
    world_pose[3:] = pose[3:]
    data.qpos[qpos_addr : qpos_addr + 7] = world_pose
    sim.forward()


def set_articulated_joint(sim, spec, pose6: np.ndarray) -> None:
    model = sim.model
    data = sim.data
    if spec.joint_name is None:
        return
    joint_id = model.joint_name2id(spec.joint_name)
    qpos_addr = int(model.jnt_qposadr[joint_id])
    if spec.control == "prismatic":
        value = spec.joint_zero + spec.joint_sign * float(pose6[0])
    elif spec.control == "revolute_yaw":
        value = spec.joint_zero + spec.joint_sign * float(pose6[5])
    else:
        raise ValueError(f"unsupported articulated control: {spec.control}")
    joint_range = np.asarray(model.jnt_range[joint_id], dtype=np.float64)
    value = float(np.clip(value, joint_range[0], joint_range[1]))
    data.qpos[qpos_addr] = value
    sim.forward()


def write_preregistration(output_root: Path, rows: list[dict], task_keys: list[str]) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    conditions = [
        {
            "condition_id": condition_id,
            "generator": row["generator"],
            "causal_delta": np.asarray(row["causal_delta"], dtype=np.float64).tolist(),
        }
        for condition_id, row in enumerate(rows)
    ]
    task_specs = []
    for spec in LIBERO_RELATION_SPECS:
        if spec.task_key not in task_keys:
            continue
        task_specs.append(
            {
                "task_key": spec.task_key,
                "suite_name": spec.suite_name,
                "task_id": spec.task_id,
                "language": spec.language,
                "family": spec.family,
                "oracle_selector": list(spec.oracle_selector),
                "nominal_pose6": list(spec.nominal_pose6),
                "body_name": spec.body_name,
                "joint_name": spec.joint_name,
                "free_joint_name": spec.free_joint_name,
                "control": spec.control,
                "notes": spec.notes,
            }
        )

    manifest_path = output_root / "libero_relation_suite_fixed_contexts.json"
    manifest = {
        "schema_version": 1,
        "env_id": "LIBERO/robosuite controlled relation suite",
        "generator_basis": GENERATOR_BASIS,
        "generator_scale": {
            "translation_m": 0.012,
            "roll_pitch_deg": 15.0,
            "yaw_deg": 30.0,
        },
        "conditions": conditions,
    }
    with manifest_path.open("w", encoding="utf-8") as output_file:
        json.dump(manifest, output_file, indent=2)

    experiment_path = output_root / "libero_relation_suite_experiment.json"
    experiment = {
        "schema_version": 1,
        "status": "preregistered_before_fitting",
        "title": "LIBERO/robosuite controlled ten-task alpha relation suite",
        "context_manifest": str(manifest_path.resolve()),
        "context_manifest_sha256": sha256(manifest_path),
        "generator_basis": GENERATOR_BASIS,
        "phase_codes": list(PHASE_CODES),
        "phase_names": list(PHASE_NAMES),
        "active_phases": ["move", "hold"],
        "task_specs": task_specs,
        "policy": (
            "Controlled state-level trajectories in real LIBERO MuJoCo scenes; "
            "not offline demos and not learned robot policies."
        ),
    }
    with experiment_path.open("w", encoding="utf-8") as output_file:
        json.dump(experiment, output_file, indent=2)
    return manifest_path, experiment_path


def collect_task_seed(spec, rows: list[dict], seed: int, args) -> Path:
    env, bddl, libero_task_name = make_libero_env(spec, args.camera_size)
    env.seed(seed)
    env.reset()
    sim = env.sim
    model = sim.model
    data = sim.data
    body_id = model.body_name2id(spec.body_name)
    initial_free_qpos = None
    if spec.free_joint_name is not None:
        joint_id = model.joint_name2id(spec.free_joint_name)
        qpos_addr = int(model.jnt_qposadr[joint_id])
        initial_free_qpos = np.asarray(data.qpos[qpos_addr : qpos_addr + 7], dtype=np.float64).copy()

    path = args.rollout_root / spec.task_key / f"{spec.task_key}_seed_{seed}.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with h5py.File(path, "w") as data_file:
        data_file.attrs["schema_version"] = 1
        data_file.attrs["env_id"] = "LIBERO/robosuite controlled relation suite"
        data_file.attrs["task_key"] = spec.task_key
        data_file.attrs["suite_name"] = spec.suite_name
        data_file.attrs["task_id"] = spec.task_id
        data_file.attrs["libero_task_name"] = libero_task_name
        data_file.attrs["language"] = spec.language
        data_file.attrs["family"] = spec.family
        data_file.attrs["bddl_file"] = bddl
        data_file.attrs["seed"] = seed
        data_file.attrs["body_name"] = spec.body_name
        data_file.attrs["oracle_selector"] = spec.oracle_selector
        data_file.attrs["nominal_pose6"] = spec.nominal_pose6
        data_file.attrs["phase_codes"] = PHASE_CODES
        data_file.attrs["phase_names"] = PHASE_NAMES

        for condition_id, row in enumerate(rows):
            causal_delta = np.asarray(row["causal_delta"], dtype=np.float64)
            phases = []
            object_pose = []
            raw_pose = []
            for phase_code in PHASE_CODES:
                if phase_code == 3:
                    alphas = np.zeros(args.steps_per_phase)
                elif phase_code == 4:
                    alphas = np.linspace(0.0, 1.0, args.steps_per_phase)
                else:
                    alphas = np.ones(args.steps_per_phase)
                for alpha in alphas:
                    pose = object_pose_at(spec, causal_delta, float(alpha))
                    pose6 = alpha * (
                        np.asarray(spec.nominal_pose6, dtype=np.float64)
                        + selected_delta(spec, causal_delta)
                    )
                    if spec.free_joint_name is not None:
                        set_free_body(sim, spec, pose, initial_free_qpos)
                    elif spec.joint_name is not None:
                        set_articulated_joint(sim, spec, pose6)
                    phases.append(phase_code)
                    object_pose.append(pose)
                    raw_pose.append(np.concatenate([data.xpos[body_id], data.xquat[body_id]]))

            group = data_file.create_group(f"episode_{condition_id:04d}")
            group.attrs["condition_id"] = condition_id
            group.attrs["generator"] = row["generator"]
            group.attrs["seed"] = seed
            group.create_dataset("causal_delta", data=causal_delta)
            group.create_dataset(
                "target_pose",
                data=np.tile(target_pose(spec, causal_delta), (len(phases), 1)),
            )
            group.create_dataset("object_pose", data=np.asarray(object_pose, dtype=np.float64))
            group.create_dataset(
                "libero_body_pose_world",
                data=np.asarray(raw_pose, dtype=np.float64),
            )
            group.create_dataset("solver_phase", data=np.asarray(phases, dtype=np.int32))
            success = np.zeros(len(phases), dtype=bool)
            success[-1] = True
            group.create_dataset("success", data=success)
            group.create_dataset("truncated", data=np.zeros(len(phases), dtype=bool))

    env.close()
    print(f"saved {path} episodes={len(rows)} seconds={time.time() - started:.1f}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/libero_relation_suite"),
    )
    parser.add_argument(
        "--rollout-root",
        type=Path,
        default=Path("phase_switch_symmetry_rollouts_libero_relation_suite"),
    )
    parser.add_argument("--context-seed", type=int, default=20260818)
    parser.add_argument("--mixed-samples", type=int, default=60)
    parser.add_argument("--steps-per-phase", type=int, default=12)
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=[spec.task_key for spec in LIBERO_RELATION_SPECS],
        help="Task keys to collect; defaults to the ten-task suite.",
    )
    args = parser.parse_args()

    spec_map = specs_by_key()
    unknown = [task for task in args.tasks if task not in spec_map]
    if unknown:
        raise ValueError(f"unknown task keys: {unknown}")

    rows = intervention_rows_se3(args.mixed_samples, args.context_seed)
    manifest_path, experiment_path = write_preregistration(args.output_root, rows, args.tasks)
    print("context_manifest:", manifest_path)
    print("experiment:", experiment_path)
    for task_key in args.tasks:
        spec = spec_map[task_key]
        for seed in args.seeds:
            collect_task_seed(spec, rows, seed, args)


if __name__ == "__main__":
    main()
