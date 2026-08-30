from __future__ import annotations

"""Planar-push SE(3) benchmark (supplement).

Rebuts the reviewer concern that the SE(3) relevance model was "tailored to
circular symmetry": the planar-push task has a genuinely different physical
structure. A rectangular cuboid is pushed across a table to a ghost 6-DOF
target. The planar table constraint SUPPRESSES three generators (dw, roll,
pitch — the block never leaves the table or tilts) while du/dv (in-plane
translation) and, in the heading arm, yaw (heading about the table normal) are
TRACKED. Selectivity here is the COMPLEMENT of peg-in-hole: instead of one
selective generator among six always-on ones, three generators are always-off.

Two arms, one env (PlanarPush-v1), distinguished by ``goal_heading``:

    heading_push  : block aligns in-plane position AND heading -> [1,1,0,0,0,1]
    free_yaw_push : block aligns in-plane position only       -> [1,1,0,0,0,0]

Oracle over the four solver phases (reach=3, push=4, align=5, retract=6),
mapped to normalized phase codes 0..3:

    heading_push:  du   [0,1,1,1]     free_yaw_push: du [0,1,1,1]
                   dv   [0,1,1,1]                    dv [0,1,1,1]
                   dw   [0,0,0,0]                    dw [0,0,0,0]
                   roll [0,0,0,0]                    roll [0,0,0,0]
                   pitch[0,0,0,0]                    pitch[0,0,0,0]
                   yaw  [0,0,1,1]                    yaw  [0,0,0,0]

Headline metric M-oop (out-of-plane suppression, heading_push): correct iff the
MAX over active phases (push + align, normalized codes 1..2) of alpha_j is
  < 0.5 for all of {dw, roll, pitch}   AND
  > 0.5 for all of {du, dv, yaw}.
The complementary per-arm correctness is also scored for free_yaw_push (where
yaw joins the suppressed set).

M-yaw (heading vs free-yaw contrast): correct iff
  alpha_yaw_active_max(heading_push) > 0.5  AND
  alpha_yaw_active_max(free_yaw_push) < 0.5.

Because success terminates the episode as soon as the block is at the target,
episodes end during push (small |yaw|) or align (large |yaw|): align/retract
phases are absent for many conditions. The block curve is therefore extracted
per phase and empty phases are HELD at the episode's final pose (the block does
not move after success). This keeps the fixed 4-phase oracle while honouring the
real, variable-length trajectories.
"""

import argparse
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from transforms3d.quaternions import quat2mat

from analyze_phase_switch_rollouts import resample
from benchmark_phase_switch_baselines import PHASE_CODES, progress_grid
from benchmark_se3_transfer import (
    SE3_MODEL_ORDER,
    build_models,
    json_ready,
    pose_to_pose6,
    sha256,
)
from phase_switch_se3_baselines import (
    SE3_DIM,
    euler_from_matrix,
    pose6_from_se3,
    se3_from_pose6,
    se3_inverse,
)

SEEDS = [20260818, 20270818, 20280818]
TASK_ORDER = ("heading_push", "free_yaw_push")
TASK_FILES = {
    "heading_push": "phase_switch_symmetry_rollouts_planar_push/heading_push_seed_{seed}.h5",
    "free_yaw_push": "phase_switch_symmetry_rollouts_planar_push/free_yaw_push_seed_{seed}.h5",
}
GENERATOR_NAMES = ("du", "dv", "dw", "roll", "pitch", "yaw")
DU_INDEX, DV_INDEX, DW_INDEX = 0, 1, 2
ROLL_INDEX, PITCH_INDEX, YAW_INDEX = 3, 4, 5

# Normalized phase codes for the active (scored) phases: push=1, align=2.
ACTIVE_CODES = (1, 2)
OOP_THRESHOLD = 0.5


def usable_planar_push(group):
    """An episode is usable iff it succeeded (block reached the target) and was
    not truncated. Unlike peg-in-hole, align/retract are optional (success fires
    early), so completeness over all four phases is NOT required."""
    return (
        bool(np.asarray(group["success"])[-1])
        and not bool(np.asarray(group["truncated"]).any())
    )


def _phase_block_pose6(group, phase, bins, final_pose6):
    phases = np.asarray(group["solver_phase"])
    indices = np.flatnonzero(phases == phase)
    if len(indices) == 0:
        return np.tile(final_pose6, (bins, 1))
    block = np.asarray(group["block_pose"])[indices]
    rpy = np.stack([euler_from_matrix(quat2mat(q)) for q in block[:, 3:]])
    rpy = np.unwrap(rpy, axis=0)
    pose6 = np.column_stack([block[:, 0], block[:, 1], block[:, 2], rpy])
    return resample(pose6, bins)


def planar_push_curve(group, bins):
    """Block pose6 curve over the four phases; empty phases are held at the
    episode's final pose (the block does not move after success)."""
    block = np.asarray(group["block_pose"])
    last = block[-1]
    final_pose6 = np.concatenate(
        [last[:3], np.asarray(euler_from_matrix(quat2mat(last[3:])))]
    )
    return np.concatenate(
        [_phase_block_pose6(group, phase, bins, final_pose6) for phase in PHASE_CODES],
        axis=0,
    )


def nominal_frame_planar_push(data_file, keys, contexts):
    """Recover the nominal ghost-target frame pose6 from mixed target poses by
    removing each known SE(3) intervention: C0 = target_world o intervention^{-1}."""
    pose6s = []
    for key, context in zip(keys, contexts):
        target_pose = np.asarray(data_file[key]["target_pose"])[0]
        target_pose6 = pose_to_pose6(target_pose)
        target_T = se3_from_pose6(target_pose6)
        intervention_T = se3_from_pose6(np.asarray(context, dtype=np.float64))
        nominal_T = target_T @ se3_inverse(intervention_T)
        pose6s.append(pose6_from_se3(nominal_T))
    pose6s = np.asarray(pose6s)
    nominal = np.empty(SE3_DIM, dtype=np.float64)
    nominal[:3] = pose6s[:, :3].mean(axis=0)
    for generator in range(3):
        angle = pose6s[:, 3 + generator]
        nominal[3 + generator] = float(
            np.arctan2(np.sin(angle).mean(), np.cos(angle).mean())
        )
    return nominal


def oracle_alpha_planar_push(arm, phase_codes):
    """6 x T generator-law oracle (phasewise, normalized codes 0..3)."""
    phase_codes = np.asarray(phase_codes, dtype=int)
    oracle = np.zeros((len(phase_codes), SE3_DIM), dtype=np.float64)
    oracle[:, DU_INDEX] = (phase_codes >= 1).astype(np.float64)
    oracle[:, DV_INDEX] = (phase_codes >= 1).astype(np.float64)
    if arm == "heading_push":
        oracle[:, YAW_INDEX] = (phase_codes >= 2).astype(np.float64)
    return oracle


def m_oop_correct(alpha_active_max, arm):
    """Correct iff suppressed generators are all < 0.5 over active phases AND
    relevant generators are all > 0.5. Relevant/suppressed sets differ per arm."""
    relevant = {DU_INDEX, DV_INDEX, YAW_INDEX} if arm == "heading_push" else {DU_INDEX, DV_INDEX}
    suppressed = {DW_INDEX, ROLL_INDEX, PITCH_INDEX} if arm == "heading_push" else {
        DW_INDEX, ROLL_INDEX, PITCH_INDEX, YAW_INDEX
    }
    return bool(
        all(alpha_active_max[j] < OOP_THRESHOLD for j in suppressed)
        and all(alpha_active_max[j] > OOP_THRESHOLD for j in relevant)
    )


def load_planar_push_dataset(path, bins):
    with h5py.File(path, "r") as data_file:
        usable_groups = {
            int(data_file[key].attrs["condition_id"]): key
            for key in data_file
            if key.startswith("episode_") and usable_planar_push(data_file[key])
        }
        mixed_cond = {
            cid: key
            for cid, key in usable_groups.items()
            if str(data_file[key].attrs["generator"]) == "mixed"
        }
        mixed_cids = sorted(mixed_cond)
        mixed_keys = [mixed_cond[cid] for cid in mixed_cids]
        mixed_contexts = np.asarray(
            [np.asarray(data_file[key]["causal_delta"]) for key in mixed_keys]
        )
        mixed_curves = np.asarray(
            [planar_push_curve(data_file[key], bins) for key in mixed_keys]
        )
    return {
        "mixed_cids": mixed_cids,
        "mixed_keys": mixed_keys,
        "mixed_contexts": mixed_contexts,
        "mixed_curves": mixed_curves,
    }


_DATASETS = None
_PDIAG_CONFIG = None
_PROGRESS = None
_PHASE_CODES = None
_ACTIVE_MASK = None


def _fit_subset(job):
    arm, seed, subset = job
    ds = _DATASETS[arm][seed]
    cid_to_idx = {cid: i for i, cid in enumerate(ds["mixed_cids"])}
    source_cids = subset["source_condition_ids"]
    missing = [cid for cid in source_cids if cid not in cid_to_idx]
    placeholder = lambda model, error: dict(
        arm=arm, seed=seed, subset_id=subset["subset_id"],
        protocol=subset["protocol"], sample_size=subset["sample_size"],
        repeat=subset["repeat"], model=model,
        fit_success=False, fit_error=error, e_alpha=np.nan,
        m_oop_correct=False,
        **{f"alpha_active_{GENERATOR_NAMES[j]}": np.nan for j in range(SE3_DIM)},
    )
    if missing:
        return [placeholder(model, "missing_source_condition") for model in SE3_MODEL_ORDER], []
    indices = [cid_to_idx[cid] for cid in source_cids]
    contexts = ds["mixed_contexts"][indices]
    curves = ds["mixed_curves"][indices]
    keys = [ds["mixed_keys"][i] for i in indices]
    try:
        with h5py.File(Path(TASK_FILES[arm].format(seed=seed)), "r") as data_file:
            nominal_frame_pose = nominal_frame_planar_push(data_file, keys, contexts)
        models = build_models(nominal_frame_pose, _PDIAG_CONFIG)
    except Exception as exc:
        return [placeholder(model, f"nominal_frame:{repr(exc)}") for model in SE3_MODEL_ORDER], []

    oracle = oracle_alpha_planar_push(arm, _PHASE_CODES)
    rows = []
    profiles = []
    for model in models:
        started = time.perf_counter()
        try:
            model.fit(contexts, curves, _PROGRESS, _PHASE_CODES)
            profile = model.jacobian_diag()  # (n_steps, 6)
            e_alpha = float(np.mean((profile - oracle) ** 2))
            active = profile[_ACTIVE_MASK]
            alpha_active_max = active.max(axis=0)
            fit_success = True
            fit_error = ""
            if model.name == "Pdiag finite (SE(3))":
                profiles.append(
                    dict(arm=arm, seed=seed, subset_id=subset["subset_id"],
                         sample_size=subset["sample_size"], profile=profile)
                )
        except Exception as exc:
            fit_success, fit_error = False, repr(exc)
            e_alpha = float("nan")
            alpha_active_max = np.full(SE3_DIM, np.nan)
        rows.append(
            dict(
                arm=arm, seed=seed, subset_id=subset["subset_id"],
                protocol=subset["protocol"], sample_size=subset["sample_size"],
                repeat=subset["repeat"], model=model.name,
                fit_success=fit_success, fit_error=fit_error,
                fit_seconds=time.perf_counter() - started,
                e_alpha=e_alpha,
                m_oop_correct=bool(m_oop_correct(alpha_active_max, arm)) if fit_success else False,
                **{f"alpha_active_{GENERATOR_NAMES[j]}": float(alpha_active_max[j]) for j in range(SE3_DIM)},
            )
        )
    return rows, profiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/planar_push/planar_push_experiment.json"),
    )
    parser.add_argument(
        "--subsets",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/planar_push/planar_push_subsets.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/planar_push"),
    )
    parser.add_argument("--bins", type=int, default=25)
    parser.add_argument("--pdiag-alpha-max", type=float, default=1.25)
    parser.add_argument("--pdiag-basis-count", type=int, default=24)
    parser.add_argument("--pdiag-basis-width", type=float, default=0.065)
    parser.add_argument("--pdiag-smoothness", type=float, default=0.1)
    parser.add_argument("--pdiag-nominal-iterations", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=0)
    args = parser.parse_args()

    pdiag_config = {
        "alpha_max": args.pdiag_alpha_max,
        "n_basis": args.pdiag_basis_count,
        "basis_width": args.pdiag_basis_width,
        "smoothness_weight": args.pdiag_smoothness,
        "nominal_iterations": args.pdiag_nominal_iterations,
    }
    with args.experiment.open(encoding="utf-8") as experiment_file:
        experiment = json.load(experiment_file)
    with args.subsets.open(encoding="utf-8") as subset_file:
        subset_manifest = json.load(subset_file)
    if subset_manifest["experiment_sha256"] != sha256(args.experiment):
        raise RuntimeError("subset manifest does not match experiment")

    progress, phase_codes = progress_grid(args.bins)
    active_mask = np.isin(phase_codes, ACTIVE_CODES)

    datasets = {}
    for arm in TASK_ORDER:
        datasets[arm] = {}
        for seed in SEEDS:
            path = Path(TASK_FILES[arm].format(seed=seed))
            if not path.exists():
                raise FileNotFoundError(f"missing dataset: {path}")
            datasets[arm][seed] = load_planar_push_dataset(path, args.bins)

    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        (arm, seed, subset)
        for arm in TASK_ORDER
        for seed in SEEDS
        for subset in subset_manifest["subsets"]
    ]

    global _DATASETS, _PDIAG_CONFIG, _PROGRESS, _PHASE_CODES, _ACTIVE_MASK
    _DATASETS = datasets
    _PDIAG_CONFIG = pdiag_config
    _PROGRESS = progress
    _PHASE_CODES = phase_codes
    _ACTIVE_MASK = active_mask

    n_jobs = args.jobs or min(16, os.cpu_count() or 1)
    fit_rows = []
    profile_accum = []
    if n_jobs > 1:
        import multiprocessing
        ctx = multiprocessing.get_context("fork")
        with ctx.Pool(processes=n_jobs) as pool:
            for rows, profiles in pool.imap_unordered(_fit_subset, jobs):
                fit_rows.extend(rows)
                profile_accum.extend(profiles)
    else:
        for job in jobs:
            rows, profiles = _fit_subset(job)
            fit_rows.extend(rows)
            profile_accum.extend(profiles)

    fits = pd.DataFrame(fit_rows)
    ok = fits[fits.fit_success]

    # ---- headline: M-oop out-of-plane suppression accuracy ----
    m_oop = (
        ok.groupby(["model", "arm", "sample_size"], sort=False)["m_oop_correct"]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "accuracy"})
        .reset_index()
    )

    # ---- per-generator active-phase max alpha_j (means) ----
    alpha_cols = [f"alpha_active_{g}" for g in GENERATOR_NAMES]
    alpha_means = (
        ok.groupby(["model", "arm", "sample_size"], sort=False)[alpha_cols]
        .mean()
        .reset_index()
    )

    # ---- E_alpha vs the planar-push oracle ----
    e_alpha = (
        ok.groupby(["model", "arm", "sample_size"], sort=False)["e_alpha"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # ---- M-yaw: heading vs free-yaw yaw contrast ----
    yaw_piv = ok.pivot_table(
        index=["model", "sample_size", "seed", "subset_id"],
        columns="arm", values="alpha_active_yaw",
    ).reset_index()
    m_yaw_rows = []
    if "heading_push" in yaw_piv.columns and "free_yaw_push" in yaw_piv.columns:
        for _, row in yaw_piv.iterrows():
            if not np.isfinite(row["heading_push"]) or not np.isfinite(row["free_yaw_push"]):
                continue
            m_yaw_rows.append(
                dict(
                    model=row["model"], sample_size=row["sample_size"],
                    seed=row["seed"], subset_id=row["subset_id"],
                    yaw_active_heading=float(row["heading_push"]),
                    yaw_active_free=float(row["free_yaw_push"]),
                    m_yaw_correct=bool(
                        row["heading_push"] > OOP_THRESHOLD
                        and row["free_yaw_push"] < OOP_THRESHOLD
                    ),
                )
            )
    m_yaw_df = pd.DataFrame(m_yaw_rows)
    m_yaw = (
        m_yaw_df.groupby(["model", "sample_size"], sort=False)
        .agg(
            m_yaw_accuracy=("m_yaw_correct", "mean"),
            yaw_active_heading_mean=("yaw_active_heading", "mean"),
            yaw_active_free_mean=("yaw_active_free", "mean"),
        )
        .reset_index()
    ) if len(m_yaw_df) else pd.DataFrame()

    # ---- write outputs ----
    fits.to_csv(args.output_root / "planar_push_fits.csv", index=False)
    m_oop.to_csv(args.output_root / "planar_push_M_oop.csv", index=False)
    alpha_means.to_csv(args.output_root / "planar_push_alpha_active_max.csv", index=False)
    e_alpha.to_csv(args.output_root / "planar_push_e_alpha.csv", index=False)
    if len(m_yaw):
        m_yaw.to_csv(args.output_root / "planar_push_M_yaw.csv", index=False)
    if profile_accum:
        np.savez(
            args.output_root / "planar_push_profiles.npz",
            profile=np.stack([p["profile"] for p in profile_accum]),
            arm=np.asarray([p["arm"] for p in profile_accum]),
            seed=np.asarray([p["seed"] for p in profile_accum]),
            subset_id=np.asarray([p["subset_id"] for p in profile_accum]),
            sample_size=np.asarray([p["sample_size"] for p in profile_accum]),
            progress=progress,
            phase_codes=phase_codes,
        )

    summary = {
        "schema_version": 1,
        "experiment_sha256": sha256(args.experiment),
        "subset_manifest_sha256": sha256(args.subsets),
        "source_sha256": {
            "benchmark_planar_push.py": sha256(Path(__file__)),
            "benchmark_se3_transfer.py": sha256(
                Path(__file__).with_name("benchmark_se3_transfer.py")
            ),
            "phase_switch_se3_baselines.py": sha256(
                Path(__file__).with_name("phase_switch_se3_baselines.py")
            ),
        },
        "oracle": {
            "heading_push": "du/dv = 1 from push on (code>=1); yaw = 1 from align on (code>=2); dw/roll/pitch = 0",
            "free_yaw_push": "du/dv = 1 from push on; dw/roll/pitch/yaw = 0",
        },
        "active_phases": {"push": 1, "align": 2},
        "curve_extraction": (
            "block pose6 per phase (reach/push/align/retract), empty phases held at "
            "the episode final pose; success terminates the episode early so "
            "align/retract are absent for small-|yaw| conditions."
        ),
        "M_oop_out_of_plane_suppression": {
            "definition": (
                "max over active phases (push+align) of alpha_j; heading_push correct "
                "iff dw/roll/pitch all < 0.5 AND du/dv/yaw all > 0.5; free_yaw_push "
                "correct iff dw/roll/pitch/yaw all < 0.5 AND du/dv all > 0.5."
            ),
            "threshold": OOP_THRESHOLD,
        },
        "M_yaw_heading_vs_free_yaw": {
            "definition": (
                "correct iff alpha_yaw_active_max(heading_push) > 0.5 AND "
                "alpha_yaw_active_max(free_yaw_push) < 0.5."
            ),
            "threshold": OOP_THRESHOLD,
        },
        "M_oop_accuracy": m_oop.to_dict(orient="records"),
        "alpha_active_max": alpha_means.to_dict(orient="records"),
        "e_alpha": e_alpha.to_dict(orient="records"),
        "M_yaw_accuracy": m_yaw.to_dict(orient="records") if len(m_yaw) else [],
        "yaw_residual_note": (
            "The heading arm's align phase is interrupted by success when the block "
            "heading enters the success tolerance (|heading_err| < 0.10 rad), so the "
            "block stops ~0.08-0.10 rad short of the target heading rather than at the "
            "align loop's 0.02 rad target. This is a constant ~5 deg offset, small vs "
            "the +-30 deg yaw signal, and lowers alpha_yaw_active(heading_push) to "
            "~0.7-0.8 (still > 0.5). free_yaw_push never rotates the block."
        ),
        "fit_count": int(len(fits)),
        "fit_failure_count": int((~fits.fit_success).sum()),
        "bins_per_phase": args.bins,
    }
    with (args.output_root / "planar_push_summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(json_ready(summary), output_file, indent=2, allow_nan=False)

    print("\n=== M-oop out-of-plane suppression accuracy ===")
    print(m_oop.to_string(index=False))
    print("\n=== per-generator active-phase max alpha_j (means) ===")
    print(alpha_means.to_string(index=False))
    print("\n=== E_alpha vs planar-push oracle ===")
    print(e_alpha.to_string(index=False))
    print("\n=== M-yaw heading vs free-yaw contrast ===")
    print(m_yaw.to_string(index=False) if len(m_yaw) else "(empty)")
    print("\nsaved:", args.output_root / "planar_push_summary.json")


if __name__ == "__main__":
    main()
