from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np
import torch

import plug_charger_causal_env  # noqa: F401 registers PlugChargerCausal-v1
from grids import isolated_grid, mixed_training_grid, smoke_grid


def to_numpy(value: Any):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (np.bool_, bool)):
        return np.asarray(value, dtype=np.bool_)
    if isinstance(value, (np.integer, int)):
        return np.asarray(value, dtype=np.int64)
    if isinstance(value, (np.floating, float)):
        return np.asarray(value, dtype=np.float64)
    return value


def write_h5(group: h5py.Group, key: str, value: Any) -> None:
    if isinstance(value, dict):
        sub = group.create_group(key)
        for sub_key, sub_value in value.items():
            write_h5(sub, str(sub_key), sub_value)
        return

    array = to_numpy(value)
    if isinstance(array, str):
        group.attrs[key] = array
    elif isinstance(array, np.ndarray):
        group.create_dataset(key, data=array)
    elif np.isscalar(array):
        group.create_dataset(key, data=array)
    else:
        group.attrs[key] = json.dumps(array)


def rows_for_split(args) -> list[dict]:
    if args.split == "smoke":
        return smoke_grid()
    if args.split == "isolated_grid":
        return isolated_grid()
    if args.split == "train_mixed":
        return mixed_training_grid(args.train_samples, args.seed)
    raise ValueError(args.split)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=["smoke", "isolated_grid", "train_mixed"],
        default="isolated_grid",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--train-samples", type=int, default=50)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("maniskill_spectrum/plug_charger_causal/reset_datasets"),
    )
    parser.add_argument("--name", default="reset_states")
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--render-backend", default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault(
        "MPLCONFIGDIR", "/home/rocos/sia/maniskill_spectrum/.matplotlib"
    )

    output_dir = args.output_root / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / f"{args.name}.h5"
    json_path = output_dir / f"{args.name}.json"
    manifest = []

    env = gym.make(
        "PlugChargerCausal-v1",
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
        render_mode=None,
        reward_mode="sparse",
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
    )

    base_rows = rows_for_split(args)
    sample_id = 0
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["env_id"] = "PlugChargerCausal-v1"
        h5.attrs["source_type"] = "controlled_reset_state"
        h5.attrs["obs_mode"] = "state_dict"
        h5.attrs["control_mode"] = "pd_joint_pos"

        for seed_offset in range(args.num_seeds):
            seed = args.seed + seed_offset
            for intervention_index, row in enumerate(base_rows):
                obs, info = env.reset(
                    seed=seed, options={"causal_delta": row["causal_delta"]}
                )
                state = env.unwrapped.get_state_dict()
                eval_info = env.unwrapped.evaluate()

                group = h5.create_group(f"sample_{sample_id}")
                group.attrs["sample_id"] = sample_id
                group.attrs["seed"] = seed
                group.attrs["seed_offset"] = seed_offset
                group.attrs["intervention_index"] = intervention_index
                group.attrs["generator"] = row["generator"]
                group.create_dataset(
                    "causal_delta", data=np.asarray(row["causal_delta"], dtype=np.float64)
                )
                write_h5(group, "obs", obs)
                write_h5(group, "env_state", state)
                write_h5(group, "reset_info", info)
                write_h5(group, "evaluate", eval_info)

                record = dict(
                    row,
                    sample_id=sample_id,
                    seed=seed,
                    seed_offset=seed_offset,
                    intervention_index=intervention_index,
                    success=bool(to_numpy(eval_info["success"])[0]),
                    obj_to_goal_dist=float(to_numpy(eval_info["obj_to_goal_dist"])[0]),
                    obj_to_goal_angle=float(to_numpy(eval_info["obj_to_goal_angle"])[0]),
                )
                manifest.append(record)
                print(json.dumps(record, sort_keys=True))
                sample_id += 1

    env.close()
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "env_id": "PlugChargerCausal-v1",
                "source_type": "controlled_reset_state",
                "split": args.split,
                "num_samples": sample_id,
                "samples": manifest,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    print(
        json.dumps(
            {
                "h5": str(h5_path),
                "json": str(json_path),
                "num_samples": sample_id,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
