from __future__ import annotations

"""Rotated-axis EXPLORATORY analysis on the common-success intersection.

The pilot collection had 4 mixed conditions (14, 16, 27, 36) that failed in at
least one file (condition 27 consistently in Q2). This is systematic
context-dependent missingness, NOT MCAR, so it must not be silently dropped for
the formal result. This script is a complete-case SENSITIVITY analysis only: fit
both orientations on the identical subset of mixed contexts that succeeded in
every orientation and seed, to check whether the invariance signal already
exists. It is exploratory, not the preregistered confirmatory result.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from analyze_phase_switch_rotated import transform_curve
from benchmark_phase_switch_baselines import PHASE_CODES, progress_grid, usable
from phase_switch_baselines import SmoothFinitePDiagModel


def common_usable_mixed(files):
    common = None
    for path in files:
        usable_set = set()
        with h5py.File(path, "r") as f:
            cond = {}
            for k in f:
                g = f[k]
                cid = int(g.attrs.get("condition_id", -1))
                ok = usable(g)
                cond[cid] = cond.get(cid, False) or ok
            for cid, s in cond.items():
                if s:
                    usable_set.add(cid)
        common = usable_set if common is None else (common & usable_set)
    return sorted(common)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/rotated_axis_experiment.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/fixed_contexts.json"),
    )
    parser.add_argument("--bins", type=int, default=25)
    args = parser.parse_args()

    with args.experiment.open() as f:
        experiment = json.load(f)
    config = {
        "alpha_max": experiment["frozen_model"]["alpha_max"],
        "n_basis": experiment["frozen_model"]["n_basis"],
        "basis_width": experiment["frozen_model"]["basis_width"],
        "smoothness_weight": experiment["frozen_model"]["smoothness_weight"],
        "nominal_iterations": experiment["frozen_model"]["nominal_iterations"],
    }
    with args.manifest.open() as f:
        manifest = json.load(f)
    gen_by_id = {c["condition_id"]: c["generator"] for c in manifest["conditions"]}

    common = common_usable_mixed(args.files)
    mixed_ids = [c for c in common if gen_by_id[c] == "mixed"]
    isolated_ids = [c for c in common if gen_by_id[c] in {"yaw", "translation"} and
                    np.linalg.norm(np.asarray(manifest["conditions"][c]["causal_delta"])) > 1e-12]
    baseline_ids = [c for c in common if gen_by_id[c] == "yaw" and
                    np.linalg.norm(np.asarray(manifest["conditions"][c]["causal_delta"])) <= 1e-12]
    print(f"common mixed={len(mixed_ids)}, isolated={len(isolated_ids)}, baseline={len(baseline_ids)}")
    if len(mixed_ids) < 10:
        raise RuntimeError("too few common mixed contexts for a meaningful fit")

    progress, phase_codes = progress_grid(args.bins)
    preclear = 2 * args.bins - 1
    profiles = {}
    rows = []
    for path in args.files:
        q = "Q1" if "_Q1_" in path.stem else "Q2" if "_Q2_" in path.stem else path.stem
        seed = int(path.stem.split("seed_")[-1])
        with h5py.File(path, "r") as f:
            orientation = np.asarray(f.attrs["orientation"], dtype=np.float64)
            task_anchor = np.asarray(f.attrs["task_anchor"], dtype=np.float64)
            from phase_switch_symmetry_env import SOCKET_CENTER

            def curve(cid):
                for k in f:
                    if int(f[k].attrs.get("condition_id", -1)) == cid and usable(f[k]):
                        return transform_curve(f[k], args.bins, orientation, task_anchor, SOCKET_CENTER)
                raise RuntimeError(f"condition {cid} not usable in {path}")

            contexts = np.asarray([manifest["conditions"][c]["causal_delta"] for c in mixed_ids])
            curves = np.asarray([curve(c) for c in mixed_ids])
            pdiag = SmoothFinitePDiagModel(
                nominal_frame_xy=SOCKET_CENTER[:2].copy(), nominal_frame_yaw=0.0, **config
            ).fit(contexts, curves, progress, phase_codes)
            profile = pdiag.jacobian_diag()
            profiles[(q, seed)] = profile
            rows.append({
                "orientation": q, "seed": seed,
                "g_translation_final": float(profile[-1, :2].mean()),
                "g_axial_preclear": float(profile[preclear, 2]),
                "g_axial_final": float(profile[-1, 2]),
            })

    frame = pd.DataFrame(rows)
    frame["law_pass"] = ((frame.g_translation_final > 0.9) & (frame.g_axial_preclear > 0.9)
                         & (frame.g_axial_final.abs() < 0.1))
    seeds = sorted({s for q, s in profiles if q == "Q1"} & {s for q, s in profiles if q == "Q2"})
    layer2 = []
    for seed in seeds:
        a = profiles[("Q1", seed)]; b = profiles[("Q2", seed)]
        layer2.append({
            "seed": seed,
            "D_orient": float(np.sqrt(np.mean((a - b) ** 2))),
            "rho_axial": float(np.corrcoef(a[:, 2], b[:, 2])[0, 1]),
            "abs_delta_g_trans_final": float(abs(a[-1, :2].mean() - b[-1, :2].mean())),
            "abs_delta_g_axial_final": float(abs(a[-1, 2] - b[-1, 2])),
        })
    layer2 = pd.DataFrame(layer2)

    def cross_seed(q):
        ss = [s for qq, s in profiles if qq == q]
        vals = [np.sqrt(np.mean((profiles[(q, ss[i])] - profiles[(q, ss[j])]) ** 2))
                for i in range(len(ss)) for j in range(i + 1, len(ss))]
        return float(np.mean(vals))
    d_seed = float(np.mean([cross_seed("Q1"), cross_seed("Q2")]))
    d_orient = float(layer2.D_orient.mean())

    print("\n=== Layer 1 (common-success set) ===")
    print(frame.to_string(index=False))
    print("\n=== Layer 2 (matched seed) ===")
    print(layer2.to_string(index=False))
    print(f"\n=== Layer 3 ===\nD_orient = {d_orient:.4f}, D_seed = {d_seed:.4f}, "
          f"D_orient - D_seed = {d_orient - d_seed:.4f}")


if __name__ == "__main__":
    main()
