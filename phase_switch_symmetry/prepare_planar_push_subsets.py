from __future__ import annotations

"""Freeze rank-6 planar-push mixed-context subsets (planar_push_subsets.json).

Mirror of prepare_se3_subsets.py for the push task. The mixed pool is the same
60 six-vectors as the frozen peg-in-hole manifest; qualified subsets must have
rank 6 and condition number < threshold. Sample sizes are {8, 15, 30}.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SAMPLE_SIZES = [8, 15, 30]
CONTEXT_SCALE = np.array(
    [0.012, 0.012, 0.012, np.deg2rad(15.0), np.deg2rad(15.0), np.deg2rad(30.0)]
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnostics(contexts, indices):
    selected = contexts[np.asarray(indices, dtype=int)] / CONTEXT_SCALE
    return {
        "rank": int(np.linalg.matrix_rank(selected)),
        "augmented_intercept_rank": int(
            np.linalg.matrix_rank(
                np.column_stack([np.ones(len(selected)), selected])
            )
        ),
        "condition_number": float(np.linalg.cond(selected)),
    }


def draw_unique(rng, sample_size, count, predicate=lambda _: True):
    selected = []
    seen = set()
    attempts = 0
    while len(selected) < count:
        attempts += 1
        if attempts > 100000:
            raise RuntimeError(
                f"could not draw {count} unique qualified subsets for N={sample_size}"
            )
        indices = tuple(sorted(rng.choice(60, sample_size, replace=False).tolist()))
        if indices in seen or not predicate(indices):
            continue
        seen.add(indices)
        selected.append(indices)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path(
            "phase_switch_symmetry_multiseed/planar_push/planar_push_experiment.json"
        ),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--selection-seed", type=int, default=20260818)
    parser.add_argument("--condition-threshold", type=float, default=10.0)
    args = parser.parse_args()

    with args.experiment.open(encoding="utf-8") as experiment_file:
        experiment = json.load(experiment_file)
    context_path = Path(experiment["context_manifest"])
    if sha256(context_path) != experiment["context_manifest_sha256"]:
        raise RuntimeError("context manifest changed after preregistration")
    with context_path.open(encoding="utf-8") as context_file:
        context_manifest = json.load(context_file)
    mixed_rows = [
        row for row in context_manifest["conditions"] if row["generator"] == "mixed"
    ]
    contexts = np.asarray([row["causal_delta"] for row in mixed_rows])
    source_condition_ids = [int(row["condition_id"]) for row in mixed_rows]
    if len(contexts) != 60:
        raise RuntimeError(
            f"planar-push manifest requires 60 fixed mixed contexts, found {len(contexts)}"
        )

    rng = np.random.default_rng(args.selection_seed)
    subsets = []
    subset_id = 0
    for sample_size in SAMPLE_SIZES:
        random_subsets = draw_unique(rng, sample_size, args.repeats)
        qualified_subsets = draw_unique(
            rng,
            sample_size,
            args.repeats,
            predicate=lambda indices: (
                diagnostics(contexts, indices)["rank"] == 6
                and diagnostics(contexts, indices)["condition_number"]
                < args.condition_threshold
            ),
        )
        for protocol, protocol_subsets in [
            ("random", random_subsets),
            ("qualified", qualified_subsets),
        ]:
            for repeat, indices in enumerate(protocol_subsets):
                subsets.append(
                    {
                        "subset_id": subset_id,
                        "protocol": protocol,
                        "sample_size": sample_size,
                        "repeat": repeat,
                        "mixed_indices": list(indices),
                        "source_condition_ids": [
                            source_condition_ids[index] for index in indices
                        ],
                        **diagnostics(contexts, indices),
                    }
                )
                subset_id += 1

    output = {
        "schema_version": 2,
        "status": "frozen_before_planar_push_fitting",
        "experiment": str(args.experiment.resolve()),
        "experiment_sha256": sha256(args.experiment),
        "context_manifest_sha256": experiment["context_manifest_sha256"],
        "selection_seed": args.selection_seed,
        "sample_sizes": SAMPLE_SIZES,
        "repeats_per_protocol": args.repeats,
        "context_scale": CONTEXT_SCALE.tolist(),
        "qualification": {
            "rank": 6,
            "condition_number_strictly_below": args.condition_threshold,
            "augmented_intercept_rank_is_reported_but_not_filtered": True,
        },
        "policies": {
            "same_subset_indices_across_execution_seeds": True,
            "same_subset_indices_across_tasks": True,
            "random_protocol_drops_no_subsets": True,
            "fit_failures_are_retained": True,
            "no_new_rollouts": True,
        },
        "subsets": subsets,
    }
    output_path = args.experiment.parent / "planar_push_subsets.json"
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(output, output_file, indent=2)
    print("saved:", output_path)
    print("subsets:", len(subsets))


if __name__ == "__main__":
    main()
