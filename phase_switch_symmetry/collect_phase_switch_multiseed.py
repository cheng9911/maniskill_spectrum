from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py

from collect_phase_switch_rollouts import collect


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completed_dataset(path: Path, seed: int, manifest_hash: str) -> bool:
    try:
        with h5py.File(path, "r") as data_file:
            return (
                int(data_file.attrs["seed"]) == seed
                and str(data_file.attrs["context_manifest_sha256"])
                == manifest_hash
                and int(data_file.attrs["condition_count"]) == 39
                and int(data_file.attrs["episode_count"]) == len(data_file)
            )
    except (OSError, KeyError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/experiment.json"),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Optional registered seed subset; omitted means all five.",
    )
    parser.add_argument("--skip-complete", action="store_true")
    args = parser.parse_args()

    with args.experiment.open(encoding="utf-8") as experiment_file:
        experiment = json.load(experiment_file)
    registered_seeds = [int(seed) for seed in experiment["seeds"]]
    seeds = registered_seeds if args.seeds is None else args.seeds
    if any(seed not in registered_seeds for seed in seeds):
        raise ValueError("requested seeds must be preregistered in experiment.json")
    context_path = Path(experiment["context_manifest"])
    manifest_hash = sha256(context_path)
    if manifest_hash != experiment["context_manifest_sha256"]:
        raise RuntimeError("frozen context manifest hash changed after preregistration")
    with context_path.open(encoding="utf-8") as context_file:
        context_manifest = json.load(context_file)
    rows = [
        {
            "generator": str(row["generator"]),
            "causal_delta": list(row["causal_delta"]),
        }
        for row in context_manifest["conditions"]
    ]
    if len(rows) != 39:
        raise RuntimeError("fixed-context experiment requires exactly 39 conditions")

    output_root = args.experiment.parent / "rollouts"
    output_root.mkdir(parents=True, exist_ok=True)
    status = {
        "experiment": str(args.experiment.resolve()),
        "experiment_sha256": sha256(args.experiment),
        "context_manifest_sha256": manifest_hash,
        "datasets": {},
    }
    status_path = args.experiment.parent / "collection_status.json"
    for seed in seeds:
        output_path = output_root / f"seed_{seed}.h5"
        if output_path.exists():
            if args.skip_complete and completed_dataset(
                output_path, seed, manifest_hash
            ):
                print("skipping complete dataset:", output_path, flush=True)
            else:
                raise FileExistsError(
                    f"refusing to overwrite existing or incomplete dataset: {output_path}"
                )
        else:
            collect(
                rows,
                output_path,
                seed,
                int(experiment["retries_per_condition"]),
                robot_init_qpos_noise=float(
                    experiment["robot_init_qpos_noise_rad"]
                ),
                context_manifest_path=context_path,
            )
        status["datasets"][str(seed)] = {
            "path": str(output_path.resolve()),
            "sha256": sha256(output_path),
            "complete_container": completed_dataset(
                output_path, seed, manifest_hash
            ),
        }
        with status_path.open("w", encoding="utf-8") as status_file:
            json.dump(status, status_file, indent=2)
    print("saved:", status_path)


if __name__ == "__main__":
    main()
