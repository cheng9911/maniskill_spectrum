from __future__ import annotations

"""One-shot geometry/relation transfer benchmark.

The question tested here is different from fitting alpha from scratch:

    Can a relevance law learned in one object/scene transfer to another
    object/scene from one target nominal trajectory?

For each transfer pair, the source law is the N=30 Pdiag-finite alpha profile
already learned in the controlled LIBERO relation suite.  The target task uses
exactly one nominal target trajectory (condition 0) to instantiate the target
trajectory frame/offset.  No target intervention is used to re-identify alpha for
the transferred method; target interventions are used only as probes for
validation and for comparing from-scratch baselines.
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from benchmark_libero_relation_suite import (
    ACTIVE_CODES,
    GENERATOR_NAMES,
    SEEDS,
    _task_file,
    load_dataset,
    nominal_frame,
    oracle_alpha,
    relation_correct,
    task_curve,
)
from benchmark_se3_transfer import json_ready, sha256
from phase_switch_se3_baselines import (
    SE3DiagonalOperatorModel,
    SE3FrameWeightedModel,
    SE3PhaseScalarGPModel,
    SE3SmoothFinitePDiagModel,
    SE3TPGMMModel,
    se3_to_metric,
    wrap_angle,
)


SAMPLE_SIZES = (3, 5, 8)
TRANSFER_SOURCE_SIZE = 30
METHOD_ORDER = (
    "Ours transfer: source Pdiag N=30 + target nominal N=1",
    "Frame-weighted target scratch",
    "Phase scalar GP target scratch",
    "Pdiag finite target scratch",
    "TP-GMM SE(3) target scratch",
)


@dataclass(frozen=True)
class TransferPair:
    pair_key: str
    source_task: str
    target_task: str
    family: str
    claim: str


TRANSFER_PAIRS = (
    TransferPair(
        pair_key="drawer_to_plate_push",
        source_task="drawer_middle_open",
        target_task="plate_front_push",
        family="one_axis_sliding",
        claim="rail/sliding du relevance transfers across articulated drawer and free plate push",
    ),
    TransferPair(
        pair_key="knob_to_microwave_door",
        source_task="stove_knob_turn",
        target_task="microwave_door_revolute",
        family="revolute_yaw",
        claim="yaw-only relevance transfers across knob and hinged door",
    ),
    TransferPair(
        pair_key="bowl_stove_to_bowl_plate",
        source_task="bowl_on_stove",
        target_task="bowl_on_plate",
        family="support_to_stacking",
        claim="upright xyz placement transfers across support surfaces",
    ),
    TransferPair(
        pair_key="bowl_stove_to_cream_cheese_bowl",
        source_task="bowl_on_stove",
        target_task="cream_cheese_in_bowl",
        family="support_to_container",
        claim="upright xyz placement transfers across object and container geometry",
    ),
    TransferPair(
        pair_key="bowl_stove_to_wine_cabinet",
        source_task="bowl_on_stove",
        target_task="wine_bottle_on_cabinet",
        family="cross_object_support",
        claim="upright xyz placement transfers across object identity and scene support",
    ),
)


def _source_profiles(profile_path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    data = np.load(profile_path, allow_pickle=False)
    profiles = data["profile"]
    task_keys = data["task_key"].astype(str)
    sample_sizes = data["sample_size"]
    progress = data["progress"]
    phase_codes = data["phase_codes"]
    out: dict[str, np.ndarray] = {}
    for task_key in sorted(set(task_keys)):
        mask = (task_keys == task_key) & (sample_sizes == TRANSFER_SOURCE_SIZE)
        if np.any(mask):
            out[task_key] = profiles[mask].mean(axis=0)
    return out, progress, phase_codes


def _nominal_curve(path: Path, bins: int) -> np.ndarray:
    with h5py.File(path, "r") as data_file:
        for key in data_file:
            if key.startswith("episode_") and int(data_file[key].attrs["condition_id"]) == 0:
                return task_curve(data_file[key], bins)
    raise RuntimeError(f"missing condition 0 nominal trajectory in {path}")


def _metric_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    residual = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    residual[..., 3:] = wrap_angle(residual[..., 3:])
    return float(np.mean(se3_to_metric(residual) ** 2))


def _transfer_predict(nominal_curve: np.ndarray, contexts: np.ndarray, profile: np.ndarray) -> np.ndarray:
    prediction = nominal_curve[None, :, :] + contexts[:, None, :] * profile[None, :, :]
    prediction[..., 3:] = wrap_angle(prediction[..., 3:])
    return prediction


def _make_target_scratch_models(
    nominal_pose: np.ndarray, pdiag_config: dict, include_tpgmm: bool
) -> list:
    models = [
        SE3FrameWeightedModel(),
        SE3PhaseScalarGPModel(),
        SE3SmoothFinitePDiagModel(nominal_frame_pose=nominal_pose, **pdiag_config),
    ]
    if include_tpgmm:
        models.append(
            SE3TPGMMModel(
                frame_mode="se3",
                nominal_frame_pose=nominal_pose,
                component_candidates=(1, 2),
                cv_n_init=1,
                final_n_init=1,
            )
        )
    return models


def _split_rows(rng: np.random.Generator, n_total: int, sample_size: int, repeats: int) -> list[np.ndarray]:
    rows = []
    for _ in range(repeats):
        rows.append(np.sort(rng.choice(n_total, size=sample_size, replace=False)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path(
            "phase_switch_symmetry_multiseed/libero_relation_suite/"
            "libero_relation_suite_experiment.json"
        ),
    )
    parser.add_argument(
        "--rollout-root",
        type=Path,
        default=Path("phase_switch_symmetry_rollouts_libero_relation_suite"),
    )
    parser.add_argument(
        "--relation-output-root",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/libero_relation_suite"),
    )
    parser.add_argument(
        "--profile-path",
        type=Path,
        default=Path(
            "phase_switch_symmetry_multiseed/libero_relation_suite/"
            "libero_relation_suite_profiles.npz"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/geometry_transfer"),
    )
    parser.add_argument("--bins", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--sample-sizes", type=int, nargs="*", default=list(SAMPLE_SIZES))
    parser.add_argument("--split-seed", type=int, default=20260831)
    parser.add_argument("--pdiag-alpha-max", type=float, default=1.25)
    parser.add_argument("--pdiag-basis-count", type=int, default=8)
    parser.add_argument("--pdiag-basis-width", type=float, default=0.12)
    parser.add_argument("--pdiag-smoothness", type=float, default=0.1)
    parser.add_argument("--pdiag-nominal-iterations", type=int, default=1)
    parser.add_argument("--include-tpgmm", action="store_true")
    args = parser.parse_args()

    with args.experiment.open(encoding="utf-8") as experiment_file:
        experiment = json.load(experiment_file)
    specs = {spec["task_key"]: spec for spec in experiment["task_specs"]}
    source_profiles, progress, phase_codes = _source_profiles(args.profile_path)
    active_mask = np.isin(phase_codes, ACTIVE_CODES)
    pdiag_config = {
        "alpha_max": args.pdiag_alpha_max,
        "n_basis": args.pdiag_basis_count,
        "basis_width": args.pdiag_basis_width,
        "smoothness_weight": args.pdiag_smoothness,
        "nominal_iterations": args.pdiag_nominal_iterations,
    }

    dataset_cache = {
        task_key: {
            seed: load_dataset(_task_file(args.rollout_root, task_key, seed), args.bins)
            for seed in SEEDS
        }
        for task_key in sorted({pair.target_task for pair in TRANSFER_PAIRS})
    }
    nominal_cache = {
        task_key: {
            seed: _nominal_curve(_task_file(args.rollout_root, task_key, seed), args.bins)
            for seed in SEEDS
        }
        for task_key in sorted({pair.target_task for pair in TRANSFER_PAIRS})
    }

    rows = []
    rng = np.random.default_rng(args.split_seed)
    for pair in TRANSFER_PAIRS:
        assert pair.source_task in source_profiles, pair.source_task
        source_profile = source_profiles[pair.source_task]
        source_active = source_profile[active_mask].max(axis=0)
        source_selector = (source_active > 0.5).astype(int)
        target_selector = np.asarray(specs[pair.target_task]["oracle_selector"], dtype=int)
        assert np.array_equal(source_selector, target_selector), (
            pair.pair_key,
            source_selector.tolist(),
            target_selector.tolist(),
        )
        oracle = oracle_alpha(target_selector, phase_codes)
        transfer_e_alpha = float(np.mean((source_profile - oracle) ** 2))
        transfer_correct = relation_correct(source_active, target_selector)
        for seed in SEEDS:
            ds = dataset_cache[pair.target_task][seed]
            nominal_curve = nominal_cache[pair.target_task][seed]
            contexts = ds["mixed_contexts"]
            curves = ds["mixed_curves"]
            all_indices = np.arange(len(contexts))
            cid_to_idx = {cid: i for i, cid in enumerate(ds["mixed_cids"])}
            target_path = _task_file(args.rollout_root, pair.target_task, seed)

            for sample_size in args.sample_sizes:
                for repeat, train_indices in enumerate(
                    _split_rows(rng, len(contexts), sample_size, args.repeats)
                ):
                    heldout = np.setdiff1d(all_indices, train_indices)
                    transfer_prediction = _transfer_predict(
                        nominal_curve, contexts[heldout], source_profile
                    )
                    rows.append(
                        dict(
                            pair_key=pair.pair_key,
                            family=pair.family,
                            claim=pair.claim,
                            source_task=pair.source_task,
                            target_task=pair.target_task,
                            seed=seed,
                            sample_size=sample_size,
                            repeat=repeat,
                            method=METHOD_ORDER[0],
                            target_nominal_trajectories=1,
                            target_interventions_for_fit=0,
                            target_interventions_for_validation=sample_size,
                            source_sample_size=TRANSFER_SOURCE_SIZE,
                            fit_success=True,
                            fit_error="",
                            fit_seconds=0.0,
                            m_transfer_correct=bool(transfer_correct),
                            e_alpha=transfer_e_alpha,
                            heldout_prediction_mse=_metric_mse(
                                transfer_prediction, curves[heldout]
                            ),
                            **{
                                f"alpha_active_{GENERATOR_NAMES[j]}": float(source_active[j])
                                for j in range(len(GENERATOR_NAMES))
                            },
                        )
                    )

                    train_contexts = contexts[train_indices]
                    train_curves = curves[train_indices]
                    source_cids = [ds["mixed_cids"][int(i)] for i in train_indices]
                    keys = [ds["mixed_keys"][cid_to_idx[cid]] for cid in source_cids]
                    try:
                        with h5py.File(target_path, "r") as data_file:
                            nominal_pose = nominal_frame(data_file, keys, train_contexts)
                        models = _make_target_scratch_models(
                            nominal_pose, pdiag_config, args.include_tpgmm
                        )
                    except Exception as exc:
                        models = []
                        rows.append(
                            dict(
                                pair_key=pair.pair_key,
                                family=pair.family,
                                claim=pair.claim,
                                source_task=pair.source_task,
                                target_task=pair.target_task,
                                seed=seed,
                                sample_size=sample_size,
                                repeat=repeat,
                                method="target scratch setup",
                                target_nominal_trajectories=0,
                                target_interventions_for_fit=sample_size,
                                target_interventions_for_validation=0,
                                source_sample_size=0,
                                fit_success=False,
                                fit_error=repr(exc),
                                fit_seconds=0.0,
                                m_transfer_correct=False,
                                e_alpha=np.nan,
                                heldout_prediction_mse=np.nan,
                                **{
                                    f"alpha_active_{name}": np.nan
                                    for name in GENERATOR_NAMES
                                },
                            )
                        )
                    for model in models:
                        started = time.perf_counter()
                        try:
                            model.fit(train_contexts, train_curves, progress, phase_codes)
                            profile = model.jacobian_diag()
                            active = profile[active_mask].max(axis=0)
                            e_alpha = float(np.mean((profile - oracle) ** 2))
                            prediction = model.predict(contexts[heldout])
                            fit_success = True
                            fit_error = ""
                            m_correct = relation_correct(active, target_selector)
                            heldout_mse = _metric_mse(prediction, curves[heldout])
                        except Exception as exc:
                            active = np.full(len(GENERATOR_NAMES), np.nan)
                            e_alpha = np.nan
                            fit_success = False
                            fit_error = repr(exc)
                            m_correct = False
                            heldout_mse = np.nan
                        method_name = (
                            "TP-GMM SE(3) target scratch"
                            if model.name == "TP-GMM SE(3)"
                            else model.name.replace(" (SE(3))", " target scratch")
                        )
                        rows.append(
                            dict(
                                pair_key=pair.pair_key,
                                family=pair.family,
                                claim=pair.claim,
                                source_task=pair.source_task,
                                target_task=pair.target_task,
                                seed=seed,
                                sample_size=sample_size,
                                repeat=repeat,
                                method=method_name,
                                target_nominal_trajectories=0,
                                target_interventions_for_fit=sample_size,
                                target_interventions_for_validation=0,
                                source_sample_size=0,
                                fit_success=fit_success,
                                fit_error=fit_error,
                                fit_seconds=time.perf_counter() - started,
                                m_transfer_correct=bool(m_correct),
                                e_alpha=e_alpha,
                                heldout_prediction_mse=heldout_mse,
                                **{
                                    f"alpha_active_{GENERATOR_NAMES[j]}": float(active[j])
                                    for j in range(len(GENERATOR_NAMES))
                                },
                            )
                        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    fits = pd.DataFrame(rows)
    ok = fits[fits.fit_success].copy()
    summary = (
        ok.groupby(["method", "sample_size"], sort=False)
        .agg(
            m_transfer_accuracy=("m_transfer_correct", "mean"),
            e_alpha_mean=("e_alpha", "mean"),
            heldout_prediction_mse_mean=("heldout_prediction_mse", "mean"),
            count=("m_transfer_correct", "count"),
        )
        .reset_index()
    )
    by_pair = (
        ok.groupby(["method", "pair_key", "family", "sample_size"], sort=False)
        .agg(
            m_transfer_accuracy=("m_transfer_correct", "mean"),
            e_alpha_mean=("e_alpha", "mean"),
            heldout_prediction_mse_mean=("heldout_prediction_mse", "mean"),
            count=("m_transfer_correct", "count"),
        )
        .reset_index()
    )
    fits.to_csv(args.output_root / "geometry_transfer_fits.csv", index=False)
    summary.to_csv(args.output_root / "geometry_transfer_summary.csv", index=False)
    by_pair.to_csv(args.output_root / "geometry_transfer_by_pair.csv", index=False)

    validation = {
        "schema_version": 1,
        "experiment": str(args.experiment.resolve()),
        "experiment_sha256": sha256(args.experiment),
        "relation_output_root": str(args.relation_output_root.resolve()),
        "profile_path": str(args.profile_path.resolve()),
        "profile_sha256": sha256(args.profile_path),
        "rollout_root": str(args.rollout_root.resolve()),
        "pairs": [pair.__dict__ for pair in TRANSFER_PAIRS],
        "method_order": METHOD_ORDER,
        "sample_sizes": list(args.sample_sizes),
        "repeats": args.repeats,
        "split_seed": args.split_seed,
        "pdiag_target_scratch_config": pdiag_config,
        "include_tpgmm": bool(args.include_tpgmm),
        "definition": {
            "ours_transfer": (
                "source N=30 Pdiag-finite alpha profile is frozen; target condition "
                "0 is the only target nominal trajectory; target interventions are "
                "validation probes and are not used to re-identify alpha."
            ),
            "m_transfer_correct": (
                "source/fit active-phase alpha is thresholded at 0.5 and compared "
                "with the target task oracle selector."
            ),
            "heldout_prediction_mse": (
                "metric-scaled pose-trajectory MSE on target mixed contexts not used "
                "by target-scratch fits; ours uses no target intervention for fitting."
            ),
        },
        "fit_count": int(len(fits)),
        "fit_failure_count": int((~fits.fit_success).sum()),
        "summary": json_ready(summary.to_dict(orient="records")),
        "by_pair": json_ready(by_pair.to_dict(orient="records")),
    }
    with (args.output_root / "geometry_transfer_validation.json").open(
        "w", encoding="utf-8"
    ) as output_file:
        json.dump(json_ready(validation), output_file, indent=2, allow_nan=False)

    print("saved:", args.output_root)
    print(summary.to_string(index=False))
    failures = fits.loc[~fits.fit_success, ["pair_key", "method", "fit_error"]]
    if len(failures):
        print("fit failures:")
        print(failures.to_string(index=False))


if __name__ == "__main__":
    main()
