from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


DEFAULT_SEEDS = [20260818, 20270818, 20280818, 20290818, 20300818]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_contexts(source: Path):
    by_condition = {}
    with h5py.File(source, "r") as data_file:
        for key in data_file:
            if not key.startswith("episode_"):
                continue
            group = data_file[key]
            condition_id = int(group.attrs["condition_id"])
            row = {
                "condition_id": condition_id,
                "generator": str(group.attrs["generator"]),
                "causal_delta": np.asarray(
                    group["causal_delta"], dtype=np.float64
                ).tolist(),
            }
            if condition_id in by_condition and by_condition[condition_id] != row:
                raise RuntimeError(
                    f"condition {condition_id} differs between retained attempts"
                )
            by_condition[condition_id] = row
        source_seed = int(data_file.attrs["seed"])
        env_id = str(data_file.attrs["env_id"])
        geometry_json = str(data_file.attrs["geometry_json"])
    condition_ids = sorted(by_condition)
    if condition_ids != list(range(39)):
        raise RuntimeError(f"expected condition ids 0..38, found {condition_ids}")
    rows = [by_condition[index] for index in condition_ids]
    generators = [row["generator"] for row in rows]
    if generators.count("yaw") != 5 or generators.count("translation") != 4:
        raise RuntimeError("expected five yaw and four translation conditions")
    if generators.count("mixed") != 30:
        raise RuntimeError("expected thirty mixed conditions")
    mixed = np.asarray(
        [row["causal_delta"] for row in rows if row["generator"] == "mixed"]
    )
    if np.linalg.matrix_rank(mixed - mixed.mean(axis=0)) != 3:
        raise RuntimeError("frozen mixed contexts are not full rank")
    return rows, source_seed, env_id, json.loads(geometry_json)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(
            "phase_switch_symmetry_rollouts/"
            "keyed_circular_phase_switch_physics_v2.h5"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--robot-init-qpos-noise", type=float, default=0.01)
    parser.add_argument("--retries-per-condition", type=int, default=3)
    args = parser.parse_args()

    if len(args.seeds) != 5 or len(set(args.seeds)) != 5:
        raise ValueError("the preregistered replication requires five unique seeds")
    if args.robot_init_qpos_noise < 0:
        raise ValueError("robot initialization noise must be nonnegative")
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows, source_seed, env_id, geometry = freeze_contexts(args.source)

    context_manifest = {
        "schema_version": 1,
        "source_dataset": str(args.source.resolve()),
        "source_dataset_sha256": sha256(args.source),
        "source_seed": source_seed,
        "env_id": env_id,
        "geometry": geometry,
        "conditions": rows,
    }
    context_path = args.output_root / "fixed_contexts.json"
    with context_path.open("w", encoding="utf-8") as output_file:
        json.dump(context_manifest, output_file, indent=2)

    experiment = {
        "schema_version": 1,
        "status": "preregistered_before_collection",
        "context_manifest": str(context_path.resolve()),
        "context_manifest_sha256": sha256(context_path),
        "seeds": args.seeds,
        "robot_init_qpos_noise_rad": args.robot_init_qpos_noise,
        "retries_per_condition": args.retries_per_condition,
        "frozen_model": {
            "alpha_max": 1.25,
            "n_basis": 24,
            "basis_width": 0.065,
            "smoothness_weight": 0.1,
            "nominal_iterations": 3,
        },
        "success_criteria": {
            "g_translation_final_min": 0.90,
            "g_yaw_preclear_min": 0.90,
            "abs_g_yaw_final_max": 0.10,
            "switch_absolute_error_max": 0.03,
            "pdiag_task_error_below_scalar_every_seed": True,
        },
        "collection_policy": {
            "fixed_contexts": True,
            "retain_all_attempts": True,
            "no_replacement_seeds": True,
            "each_seed_fit_independently": True,
        },
    }
    experiment_path = args.output_root / "experiment.json"
    with experiment_path.open("w", encoding="utf-8") as output_file:
        json.dump(experiment, output_file, indent=2)
    print("saved:", context_path)
    print("saved:", experiment_path)


if __name__ == "__main__":
    main()
