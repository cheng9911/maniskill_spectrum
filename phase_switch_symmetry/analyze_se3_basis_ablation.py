from __future__ import annotations

"""Basis-ablation analysis (supplement, additive).

Reads the frozen basis-ablation fits (``basis_ablation_fits.csv``) and scores
the ablation at FULL precision. The ablation's own summary tables round to 4
decimals, which flattens the whole effect (local e_data ~ 1e-5 and rotated
~ 5e-5 both print as 0.0000); the structure lives in the relative degradation,
which this script extracts:

1. **e_data degradation ratio** per model/task/N: in-sample prediction error
   in metric units (basis-independent). Expected: Pdiag finite degrades
   (its diagonal prior only holds in the task basis) while the Full operator
   and the Frame-weighted scalar stay invariant. The Frame-weighted family
   ``X = C0 Exp(w(s) t) C0^-1 X0`` is PROVABLY basis-free: the scalar w(s)
   commutes with the adjoint, so the same family expresses the same responses
   in any basis — a built-in negative control.

2. **Analytic rescaling floor**: the rotated-basis model absorbs the basis
   rotation through its recovered nominal frame (C0' ~ C0), so its response
   operator on the local twist is Ad(A) diag(alpha') Ad(R^-1) with
   A = C0'^-1 C0_true ~ I (validated numerically on a probe subset). The
   best-achievable diagonal is the diagonal projection of Ad(A) diag(d) Ad(R);
   for A = I this is the RESCALED law d_k(s) * R[k,k] — the true generator
   relevance times the diagonal of the basis rotation. floor = mean squared
   deviation of the rescaled law from the true oracle. The fitted
   e_alpha(rotated) exceeds this floor by a large margin: the pose6-weighted
   objective makes the fit contort the diagonal to chase the off-diagonal
   response (which in the rotated basis carries most of the empirical
   operator's norm), trading law-recovery against data-fit — a tension that
   does not exist in the task basis, where the empirical response IS diagonal
   (probe: enter-phase diag ~ [1,1,1,1,1,1], off-diagonal norm 0.15 local vs
   2.77 rotated).

3. **Off-diagonal mass** of the Full operator per basis (it absorbs the
   conjugation by becoming MORE off-diagonal, 0.89 -> 0.93).

All inputs are the frozen SE(3) data and the frozen ablation run; nothing is
re-fit here.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark_phase_switch_baselines import progress_grid
from benchmark_se3_basis_ablation import (
    BASIS_ORDER,
    ROTATION_EULERS_DEG,
)
from benchmark_se3_transfer import (
    TASK_ORDER,
    json_ready,
    oracle_alpha_se3,
    sha256,
)
from phase_switch_se3_baselines import SE3_DIM, matrix_from_euler

GENERATOR_NAMES = ("du", "dv", "dw", "roll", "pitch", "yaw")


def analytic_floor(task, basis_name, R, bins):
    """Best-achievable e_alpha for ANY diagonal model in the rotated basis.

    The rotated-basis Pdiag model composes the rotated context twist through
    its recovered nominal frame C0' ~ C0 (the basis rotation is absorbed into
    the recovery, not into the response); its response operator on the local
    twist is therefore Ad(A) diag(alpha') Ad(R^-1) with A = C0'^-1 C0_true,
    which is ~ I (first order in the context mean; A = I exactly for the
    ideal model, numerically a few degrees). The best-achievable diagonal is
    the diagonal projection of Ad(A) diag(d) Ad(R):

        proj_k(s) = sum_j A[k,j] R[j,k] d_j(s)  ->  d_k(s) * R[k,k] for A = I

    i.e. the RESCALED law: the true generator relevance times the diagonal of
    the basis rotation (Ad(R) = blkdiag(R, R), so each triple rescales by the
    same diag(R)). floor = mean_s ||proj(s) - d(s)||^2 against the task-basis
    oracle, exactly like the fitted e_alpha. The rescaling preserves WHICH
    generators drop (the yaw drop survives as 0.814 -> 0) but distorts their
    magnitudes — the principled-basis reading alpha_j ~ 1 is what the task
    frame buys.
    """
    _, phase_codes = progress_grid(bins)
    oracle = oracle_alpha_se3(task, phase_codes)  # (n_steps, SE3_DIM)
    proj = np.empty_like(oracle)
    for block in range(2):
        for k in range(3):
            kk = block * 3 + k
            proj[:, kk] = R[k, k] * oracle[:, kk]
    return float(np.mean((proj - oracle) ** 2))


def ratio_table(ok, value):
    """Long-format means of `value` per model/task/N/basis + degradation ratio.

    Returns (mean_frame, ratio_frame) with rows model x task x sample_size,
    columns: local, rotated-1, rotated-2, r1/local, r2/local.
    """
    means = (
        ok.groupby(["model", "task", "sample_size", "basis"], sort=False)[value]
        .mean()
        .reset_index()
    )
    local = means[means.basis == "local"].set_index(
        ["model", "task", "sample_size"]
    )[value]
    pivot = means.pivot_table(
        index=["model", "task", "sample_size"],
        columns="basis",
        values=value,
    )[list(BASIS_ORDER)]
    pivot["r1/local"] = pivot["rotated-1"] / np.maximum(local, 1e-15)
    pivot["r2/local"] = pivot["rotated-2"] / np.maximum(local, 1e-15)
    return pivot.reset_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fits",
        type=Path,
        default=Path(
            "phase_switch_symmetry_multiseed/se3_transfer_basis_ablation/"
            "basis_ablation_fits.csv"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "phase_switch_symmetry_multiseed/se3_transfer_basis_ablation"
        ),
    )
    parser.add_argument("--bins", type=int, default=25)
    args = parser.parse_args()

    fits = pd.read_csv(args.fits)
    ok = fits[fits.fit_success].copy()

    e_data = ratio_table(ok, "e_data")
    e_alpha = ratio_table(ok, "e_alpha")

    off_diag = (
        ok[ok.model == "Full operator (SE(3))"]
        .groupby(["task", "sample_size", "basis"], sort=False)["off_diag_mass"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # ---- analytic smearing floors + comparison with fitted e_alpha ----
    floors = {}
    for task in TASK_ORDER:
        floors[task] = {}
        for basis_name in BASIS_ORDER:
            if basis_name == "local":
                floors[task][basis_name] = 0.0
                continue
            R = matrix_from_euler(
                *(
                    np.deg2rad(deg)
                    for deg in ROTATION_EULERS_DEG[basis_name]
                )
            )
            floors[task][basis_name] = analytic_floor(
                task, basis_name, R, args.bins
            )

    floor_rows = []
    for task in TASK_ORDER:
        row = {"task": task}
        row.update(
            {f"floor_{b}": floors[task][b] for b in ("rotated-1", "rotated-2")}
        )
        floor_rows.append(row)
    floor_table = pd.DataFrame(floor_rows)

    # fitted-vs-floor: Pdiag finite and Full operator (both recover a
    # diagonal; the dense operator's fitted diagonal should also sit at the
    # floor since its best fit is the conjugated true operator)
    fitted = (
        ok[ok.model.isin(
            ["Pdiag finite (SE(3))", "Full operator (SE(3))"]
        )]
        .groupby(["model", "task", "sample_size", "basis"], sort=False)["e_alpha"]
        .mean()
        .reset_index()
    )
    fitted = fitted[fitted.basis != "local"].copy()
    fitted["floor"] = fitted.apply(
        lambda r: floors[r.task][r.basis], axis=1
    )
    fitted["fitted/floor"] = fitted["e_alpha"] / np.maximum(
        fitted["floor"], 1e-15
    )

    # ---- rescaling distortion of the recovered law in each rotated basis ----
    rescale = {}
    for basis_name in ("rotated-1", "rotated-2"):
        R = matrix_from_euler(
            *(
                np.deg2rad(deg)
                for deg in ROTATION_EULERS_DEG[basis_name]
            )
        )
        rescale[basis_name] = {
            "note": (
                "Best-achievable rotated-basis diagonal = the rescaled law "
                "d_k * R[k,k] (A = C0'^-1 C0_true ~ I; see analytic_floor). "
                "The selectivity structure (which generators drop) survives "
                "the rescaling; the MAGNITUDES are under-read by 1 - R[k,k]. "
                "diag(R) applies to both the translation and rotation triples "
                "(Ad(R) = blkdiag(R, R))."
            ),
            "diag_R": [float(R[k, k]) for k in range(3)],
            "relevance_underread": {
                GENERATOR_NAMES[k]: float(1.0 - R[k, k]) for k in range(3)
            },
        }

    args.output_root.mkdir(parents=True, exist_ok=True)
    e_data.to_csv(
        args.output_root / "basis_ablation_e_data_ratio.csv", index=False
    )
    e_alpha.to_csv(
        args.output_root / "basis_ablation_e_alpha_ratio.csv", index=False
    )
    fitted.to_csv(
        args.output_root / "basis_ablation_fitted_vs_floor.csv", index=False
    )

    summary = {
        "schema_version": 1,
        "fits_source": str(args.fits),
        "source_sha256": {
            "analyze_se3_basis_ablation.py": sha256(Path(__file__)),
            "benchmark_se3_basis_ablation.py": sha256(
                Path(__file__).with_name("benchmark_se3_basis_ablation.py")
            ),
            "benchmark_se3_transfer.py": sha256(
                Path(__file__).with_name("benchmark_se3_transfer.py")
            ),
        },
        "rotations_deg": ROTATION_EULERS_DEG,
        "design": {
            "e_data": (
                "mean squared in-sample prediction error in metric units; "
                "basis-independent, scored at full precision. Pdiag finite "
                "must degrade in rotated bases; the Full operator and the "
                "provably Ad-invariant Frame-weighted scalar must not."
            ),
            "rescaling_floor": (
                "closed-form best-achievable e_alpha for any diagonal model "
                "in a rotated basis: proj_k(s) = sum_j A[k,j] R[j,k] d_j(s) "
                "with A = C0'^-1 C0_true ~ I, i.e. the rescaled law "
                "d_k(s) * R[k,k]. The fitted e_alpha(rotated) exceeds it "
                "because the pose6-weighted objective contorts the diagonal "
                "to chase the off-diagonal response (a law-vs-data tension "
                "that only exists off the task basis)."
            ),
        },
        "e_data_means_and_ratios": e_data.to_dict(orient="records"),
        "e_alpha_means_and_ratios": e_alpha.to_dict(orient="records"),
        "analytic_smearing_floor": floor_table.to_dict(orient="records"),
        "fitted_vs_floor": fitted.to_dict(orient="records"),
        "rescaling_distortion": rescale,
        "off_diag_mass_full_operator": off_diag.to_dict(orient="records"),
    }
    with (args.output_root / "basis_ablation_analysis.json").open("w", encoding="utf-8") as out_file:
        json.dump(json_ready(summary), out_file, indent=2, allow_nan=False)

    pd.set_option("display.float_format", lambda v: f"{v:.4e}")
    print("=== e_data: mean per basis + degradation ratio (full precision) ===")
    print(e_data.to_string(index=False))
    print("\n=== e_alpha (recovered diagonal vs task-basis oracle): same layout ===")
    print(e_alpha.to_string(index=False))
    print("\n=== analytic rescaling floor (closed form) ===")
    print(floor_table.to_string(index=False))
    print("\n=== fitted e_alpha vs floor (rotated bases) ===")
    print(fitted.to_string(index=False))
    print("\n=== rescaling distortion (best-achievable law = d_k * R[k,k]) ===")
    for basis_name, entry in rescale.items():
        print(f"{basis_name}: diag(R) = {[f'{v:.3f}' for v in entry['diag_R']]}  "
              f"underread = {entry['relevance_underread']}")
    print("\n=== off-diagonal mass, Full operator ===")
    print(off_diag.to_string(index=False))
    print("\nsaved:", args.output_root / "basis_ablation_analysis.json")


if __name__ == "__main__":
    main()
