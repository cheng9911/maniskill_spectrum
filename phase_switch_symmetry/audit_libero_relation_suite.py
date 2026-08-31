from __future__ import annotations

"""Audit generated LIBERO relation-suite artifacts.

This script is intentionally read-only: it validates manifests, rollout files,
benchmark rows, and the headline metric relationships used in the manuscript
supplement.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import pandas as pd


SEEDS = (20260818, 20270818, 20280818)
PHASE_CODES = [3, 4, 5, 6]
SAMPLE_SIZES = {8, 15, 30}
FULL_SE3_MODELS = (
    "Frame-weighted (SE(3))",
    "Phase scalar GP (SE(3))",
    "TP-GMM additive (SE(3))",
    "TP-GMM SE(3)",
    "Full operator (SE(3))",
    "Pdiag pointwise (SE(3))",
    "Pdiag finite (SE(3))",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(path: Path) -> Path:
    assert path.exists(), f"missing required artifact: {path}"
    return path


def check_rollout_file(path: Path, expected_task_key: str | None = None) -> tuple[int, int]:
    require(path)
    with h5py.File(path, "r") as data_file:
        if expected_task_key is not None:
            assert data_file.attrs["task_key"] == expected_task_key, (
                path,
                data_file.attrs.get("task_key"),
            )
        keys = sorted(key for key in data_file if key.startswith("episode_"))
        phases = sorted(
            set(
                int(phase)
                for key in keys
                for phase in data_file[key]["solver_phase"][:]
            )
        )
        usable = sum(
            bool(data_file[key]["success"][-1])
            and not bool(data_file[key]["truncated"][:].any())
            for key in keys
        )
    assert len(keys) == 75, (path, "episode_count", len(keys))
    assert usable == 75, (path, "usable_count", usable)
    assert phases == PHASE_CODES, (path, "phase_codes", phases)
    return len(keys), usable


def assert_model_sample_rows(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    rows = frame.loc[frame["model"] == model].copy()
    assert set(rows["sample_size"]) == SAMPLE_SIZES, (model, rows)
    return rows


def check_drawer(args: argparse.Namespace) -> dict[str, int]:
    root = args.drawer_output_root
    rollout_root = args.drawer_rollout_root
    require(root / "libero_drawer_summary.json")
    require(root / "VALIDATION_libero_drawer.md")
    require(root / "libero_drawer_experiment.json")
    require(root / "libero_drawer_subsets.json")

    episodes = 0
    usable = 0
    for seed in SEEDS:
        count, good = check_rollout_file(rollout_root / f"drawer_seed_{seed}.h5")
        episodes += count
        usable += good

    m_prismatic = pd.read_csv(require(root / "libero_drawer_M_prismatic.csv"))
    pdiag_rows = m_prismatic.loc[
        m_prismatic["model"] == "Pdiag finite (SE(3))"
    ].copy()
    assert set(pdiag_rows["sample_size"]) == SAMPLE_SIZES
    assert (pdiag_rows["accuracy"] == 1.0).all(), pdiag_rows

    tpgmm_projected = pd.read_csv(
        require(root / "libero_drawer_tpgmm_M_prismatic_projected.csv")
    )
    assert len(tpgmm_projected) == 6, tpgmm_projected
    assert set(tpgmm_projected["sample_size"]) == SAMPLE_SIZES
    assert (tpgmm_projected["accuracy"] == 1.0).all(), tpgmm_projected
    return {"drawer_files": len(SEEDS), "drawer_episodes": episodes, "drawer_usable": usable}


def check_relation_suite(args: argparse.Namespace) -> dict[str, int]:
    root = args.relation_output_root
    rollout_root = args.relation_rollout_root
    required_files = (
        "libero_relation_suite_summary.json",
        "libero_relation_suite_experiment.json",
        "libero_relation_suite_subsets.json",
        "libero_relation_suite_M_relation.csv",
        "libero_relation_suite_M_relation_by_task.csv",
        "libero_relation_suite_e_alpha.csv",
        "libero_relation_suite_e_alpha_by_task.csv",
        "libero_relation_suite_alpha_active_max.csv",
        "libero_relation_suite_tpgmm_M_relation_projected.csv",
        "libero_relation_suite_tpgmm_M_relation_projected_by_task.csv",
        "libero_relation_suite_tpgmm_alpha_active_max_projected.csv",
        "libero_relation_suite_profiles.npz",
        "VALIDATION_libero_relation_suite.md",
    )
    for name in required_files:
        require(root / name)

    experiment = json.loads((root / "libero_relation_suite_experiment.json").read_text())
    subset_manifest = json.loads((root / "libero_relation_suite_subsets.json").read_text())
    manifest_path = Path(experiment["context_manifest"])
    assert sha256(manifest_path) == experiment["context_manifest_sha256"]
    assert subset_manifest["experiment_sha256"] == sha256(
        root / "libero_relation_suite_experiment.json"
    )

    task_keys = [spec["task_key"] for spec in experiment["task_specs"]]
    families = {spec["family"] for spec in experiment["task_specs"]}
    assert len(task_keys) == 10, task_keys
    assert len(families) >= 6, families

    episodes = 0
    usable = 0
    for task_key in task_keys:
        for seed in SEEDS:
            count, good = check_rollout_file(
                rollout_root / task_key / f"{task_key}_seed_{seed}.h5",
                expected_task_key=task_key,
            )
            episodes += count
            usable += good

    m_relation = pd.read_csv(root / "libero_relation_suite_M_relation.csv")
    assert set(m_relation["model"]) == set(FULL_SE3_MODELS), sorted(m_relation["model"].unique())
    for model in FULL_SE3_MODELS:
        rows = assert_model_sample_rows(m_relation, model)
        assert set(rows["count"]) == {180}, (model, rows[["sample_size", "count"]])

    assert (assert_model_sample_rows(m_relation, "Pdiag finite (SE(3))")["accuracy"] == 1.0).all()
    assert (assert_model_sample_rows(m_relation, "Pdiag pointwise (SE(3))")["accuracy"] == 1.0).all()
    assert (assert_model_sample_rows(m_relation, "Full operator (SE(3))")["accuracy"] == 1.0).all()
    assert (assert_model_sample_rows(m_relation, "TP-GMM additive (SE(3))")["accuracy"] == 1.0).all()
    assert (assert_model_sample_rows(m_relation, "TP-GMM SE(3)")["accuracy"] == 1.0).all()
    assert (assert_model_sample_rows(m_relation, "Frame-weighted (SE(3))")["accuracy"] == 0.0).all()
    assert (assert_model_sample_rows(m_relation, "Phase scalar GP (SE(3))")["accuracy"] == 0.0).all()

    by_task = pd.read_csv(root / "libero_relation_suite_M_relation_by_task.csv")
    pdiag_by_task = by_task.loc[by_task["model"] == "Pdiag finite (SE(3))"]
    assert len(pdiag_by_task) == 10 * len(SAMPLE_SIZES), pdiag_by_task
    assert (pdiag_by_task["accuracy"] == 1.0).all(), pdiag_by_task

    e_alpha = pd.read_csv(root / "libero_relation_suite_e_alpha.csv")
    pdiag_n30 = float(
        e_alpha.loc[
            (e_alpha["model"] == "Pdiag finite (SE(3))")
            & (e_alpha["sample_size"] == 30),
            "mean",
        ].iloc[0]
    )
    tpgmm_n30 = float(
        e_alpha.loc[
            (e_alpha["model"] == "TP-GMM SE(3)")
            & (e_alpha["sample_size"] == 30),
            "mean",
        ].iloc[0]
    )
    scalar_gp_n30 = float(
        e_alpha.loc[
            (e_alpha["model"] == "Phase scalar GP (SE(3))")
            & (e_alpha["sample_size"] == 30),
            "mean",
        ].iloc[0]
    )
    assert pdiag_n30 < tpgmm_n30 < scalar_gp_n30, (pdiag_n30, tpgmm_n30, scalar_gp_n30)

    projected = pd.read_csv(root / "libero_relation_suite_tpgmm_M_relation_projected.csv")
    assert len(projected) == 6, projected
    assert set(projected["sample_size"]) == SAMPLE_SIZES
    assert (projected["accuracy"] == 1.0).all(), projected

    summary = json.loads((root / "libero_relation_suite_summary.json").read_text())
    assert summary["fit_failure_count"] == 0
    assert summary["tpgmm_fit_failure_count"] == 0
    assert summary["pdiag_config"]["n_basis"] == 8
    assert summary["pdiag_config"]["nominal_iterations"] == 1

    return {
        "relation_tasks": len(task_keys),
        "relation_families": len(families),
        "relation_files": len(task_keys) * len(SEEDS),
        "relation_episodes": episodes,
        "relation_usable": usable,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drawer-output-root",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/libero_drawer"),
    )
    parser.add_argument(
        "--drawer-rollout-root",
        type=Path,
        default=Path("phase_switch_symmetry_rollouts_libero_drawer"),
    )
    parser.add_argument(
        "--relation-output-root",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/libero_relation_suite"),
    )
    parser.add_argument(
        "--relation-rollout-root",
        type=Path,
        default=Path("phase_switch_symmetry_rollouts_libero_relation_suite"),
    )
    args = parser.parse_args()

    drawer = check_drawer(args)
    relation = check_relation_suite(args)
    print("AUDIT_OK")
    print(json.dumps({**drawer, **relation}, indent=2, sort_keys=True))
    print(
        "CAVEAT: relation suite uses controlled state-level LIBERO/robosuite "
        "episodes and a lightweight Pdiag breadth config."
    )
    print(
        "CAVEAT: SE(3) TP-GMM/GP baseline extension currently modifies the "
        "previous SE(3) benchmark source files."
    )


if __name__ == "__main__":
    main()
