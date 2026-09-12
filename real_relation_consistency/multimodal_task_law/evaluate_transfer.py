from __future__ import annotations

"""Offline cross-mode prediction matrix (plan section 7.1).

For every source mode A and target mode B, freeze A's identified pitch response
``alpha_A(s)`` and predict B's held-out trajectory from B's 0-degree reference:
``Xhat_B,c(s) = Rot_y(alpha(s) * theta_c) applied to X_B,0(s)``.  Baselines share
the identical target reference and progress:

  no_adapt : alpha = 0        (P = 0, ignore the intervention)
  rigid    : alpha = 1        (P = I, full rigid pitch)
  ramp     : alpha = s        (pre-registered smooth scalar transition)
  source   : alpha = alpha_A  (frozen source-mode law)
  target   : alpha = alpha_B  (target-mode own law, upper-bound reference)

Errors are reported per geometric segment (free approach / enter / insert) and at
the terminal state: position RMSE, full-pose RMSE, insertion-axis angle RMSE,
lateral error and axial error (decomposed along the target socket axis).
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "phase_switch_symmetry"))
sys.path.insert(0, str(HERE))

from transforms3d.quaternions import quat2mat  # noqa: E402
from generate_contexts import load_protocol  # noqa: E402
from phase_switch_se3_baselines import euler_from_matrix, matrix_from_euler_batched  # noqa: E402
from identify_modes import pose7_to_pose6  # noqa: E402

PITCH_CHANNEL = 4


def rotate_about_y(pose6: np.ndarray, ang: np.ndarray, center: np.ndarray):
    """Rotate pose6 (n,6) about the world y-axis through ``center`` by ang (n,)."""
    cp, sp = np.cos(ang), np.sin(ang)
    R = np.zeros((len(ang), 3, 3))
    R[:, 0, 0] = cp
    R[:, 0, 2] = sp
    R[:, 1, 1] = 1.0
    R[:, 2, 0] = -sp
    R[:, 2, 2] = cp
    pos = np.einsum("sij,sj->si", R, pose6[:, :3] - center) + center
    R_nom = matrix_from_euler_batched(pose6[:, 3:])
    R_out = np.einsum("sij,sjk->sik", R, R_nom)
    rot = np.array([euler_from_matrix(R_out[s]) for s in range(len(ang))])
    return np.concatenate([pos, rot], axis=1)


def predict_pose6(nominal_pose7: np.ndarray, theta: float, alpha: np.ndarray, center: np.ndarray):
    nominal = np.array([pose7_to_pose6(p) for p in nominal_pose7])
    return rotate_about_y(nominal, np.asarray(alpha) * theta, center)


def segment_error(pred: np.ndarray, actual: np.ndarray, seg_id: np.ndarray,
                  socket_axis: np.ndarray, socket_center: np.ndarray):
    """pred, actual: (n,6) pose6. Return dict of metrics per segment + terminal."""
    pos_err = pred[:, :3] - actual[:, :3]
    axis_err = np.array([
        np.arccos(np.clip(
            np.dot(quat2mat(_q_from_pose6(pred[s, 3:])) @ np.array([0, 0, 1.0]),
                   quat2mat(_q_from_pose6(actual[s, 3:])) @ np.array([0, 0, 1.0])),
            -1.0, 1.0))
        for s in range(len(pred))
    ])
    # lateral/axial decomposition of position error
    axial = pos_err @ socket_axis
    lateral = np.linalg.norm(pos_err - axial[:, None] * socket_axis[None, :], axis=1)
    pos_rmse = np.sqrt(np.mean(np.sum(pos_err ** 2, axis=1)))
    full_pose = np.concatenate([pos_err, _wrap_rot(pred[:, 3:] - actual[:, 3:])], axis=1)
    full_rmse = np.sqrt(np.mean(np.sum(full_pose ** 2, axis=1)))
    out = {
        "overall": dict(pos_rmse_mm=pos_rmse * 1000, full_rmse=full_rmse,
                        axis_rmse_deg=np.rad2deg(np.sqrt(np.mean(axis_err ** 2))),
                        lateral_mm=np.mean(lateral) * 1000, axial_mm=np.mean(np.abs(axial)) * 1000),
    }
    for s in range(3):
        m = seg_id == s
        if not m.any():
            continue
        pe = pos_err[m]; ae = axis_err[m]; lt = lateral[m]; ax = axial[m]
        fpe = full_pose[m]
        out[f"seg{s}"] = dict(
            pos_rmse_mm=np.sqrt(np.mean(np.sum(pe ** 2, axis=1))) * 1000,
            full_rmse=np.sqrt(np.mean(np.sum(fpe ** 2, axis=1))),
            axis_rmse_deg=np.rad2deg(np.sqrt(np.mean(ae ** 2))),
            lateral_mm=np.mean(lt) * 1000, axial_mm=np.mean(np.abs(ax)) * 1000,
        )
    # terminal = last 10 points
    pe = pos_err[-10:]; ae = axis_err[-10:]; lt = lateral[-10:]; ax = axial[-10:]
    fpe = full_pose[-10:]
    out["terminal"] = dict(
        pos_rmse_mm=np.sqrt(np.mean(np.sum(pe ** 2, axis=1))) * 1000,
        full_rmse=np.sqrt(np.mean(np.sum(fpe ** 2, axis=1))),
        axis_rmse_deg=np.rad2deg(np.sqrt(np.mean(ae ** 2))),
        lateral_mm=np.mean(lt) * 1000, axial_mm=np.mean(np.abs(ax)) * 1000,
    )
    return out


def _wrap_rot(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def _q_from_pose6(rpy):
    from transforms3d.euler import euler2quat
    return euler2quat(*rpy)


def evaluate(protocol, resampled_h5, events_json, id_json, out_prefix):
    p = protocol
    center = np.asarray(p["task"]["task_anchor"], dtype=np.float64)
    test_pitch = p["test_pitch_deg"]
    id_data = json.load(open(id_json))
    models = {k: np.array(v["alpha"]) for k, v in id_data["models"].items()}
    events = json.load(open(events_json))["episodes"]

    with h5py.File(resampled_h5, "r") as rf:
        # index episodes: (mode, block, pitch) -> episode
        index = {}
        for e in events:
            index[(e["mode_id"], e["block_id"], e["pitch_deg"])] = e
        # socket axis from the nominal socket (pitch=0): from the source h5 we can
        # use a reference episode's socket_pose via its group attr? Use analytic:
        # socket at pitch=0 has identity orientation (task orientation is identity)
        socket_axis = np.array([0.0, 0.0, 1.0])  # orientation=[1,0,0,0]

        records = []
        for source in p["mode_ids"]:
            for target in p["mode_ids"]:
                for blk in range(p["core_blocks"]):
                    ref = index.get((target, blk, 0.0))
                    if ref is None:
                        continue
                    X0 = rf[ref["episode_id"]]["peg_pose"][:]
                    seg_id = rf[ref["episode_id"]]["segment_id"][:]
                    alpha_source = models.get(f"{source}_{blk}")
                    alpha_target = models.get(f"{target}_{blk}")
                    for theta_deg in test_pitch:
                        actual_ep = index.get((target, blk, theta_deg))
                        if actual_ep is None:
                            continue
                        actual = rf[actual_ep["episode_id"]]["peg_pose"][:]
                        actual6 = np.array([pose7_to_pose6(q) for q in actual])
                        theta = np.deg2rad(theta_deg)
                        methods = {
                            "no_adapt": np.zeros(len(X0)),
                            "rigid": np.ones(len(X0)),
                            "ramp": np.linspace(0.0, 1.0, len(X0)),
                            "source": alpha_source if alpha_source is not None else None,
                            "target": alpha_target if alpha_target is not None else None,
                        }
                        for mname, alpha in methods.items():
                            if alpha is None:
                                continue
                            pred6 = predict_pose6(X0, theta, alpha, center)
                            err = segment_error(pred6, actual6, seg_id, socket_axis, center)
                            rec = dict(source=source, target=target, block=blk,
                                       theta_deg=theta_deg, method=mname)
                            for seg, metrics in err.items():
                                for metric, value in metrics.items():
                                    rec[f"{seg}__{metric}"] = float(value)
                            records.append(rec)

    with open(out_prefix + ".json", "w", encoding="utf-8") as mf:
        json.dump(dict(protocol_sha256=p["content_sha256"], records=records), mf, indent=2)

    # summary: 4x4 source->target matrix of mean terminal axis RMSE (deg) for method=source
    print("records:", len(records))
    summarize(records, out_prefix)


def summarize(records, out_prefix):
    import csv
    methods = sorted({r["method"] for r in records})
    metrics = ["overall__pos_rmse_mm", "overall__axis_rmse_deg", "terminal__pos_rmse_mm",
               "terminal__axis_rmse_deg", "seg0__pos_rmse_mm", "seg2__pos_rmse_mm"]
    sources = sorted({r["source"] for r in records})
    targets = sorted({r["target"] for r in records})
    with open(out_prefix + ".csv", "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(["method", "metric", "source", "target", "mean", "std", "n"])
        for method in methods:
            for metric in metrics:
                for source in sources:
                    for target in targets:
                        vals = [r[metric] for r in records
                                if r["method"] == method and r["source"] == source
                                and r["target"] == target and metric in r]
                        if vals:
                            w.writerow([method, metric, source, target,
                                        np.mean(vals), np.std(vals), len(vals)])
    # print the source-method terminal axis RMSE matrix
    print("\nterminal axis RMSE (deg) for method=source (source->target):")
    print("       " + " ".join(f"{t:>14s}" for t in targets))
    for source in sources:
        row = []
        for target in targets:
            vals = [r["terminal__axis_rmse_deg"] for r in records
                    if r["method"] == "source" and r["source"] == source and r["target"] == target]
            row.append(f"{np.mean(vals):>14.2f}" if vals else " "*14)
        print(f"{source:>6s} " + " ".join(row))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--resampled", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--id", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    evaluate(protocol, args.resampled, args.events, args.id, str(args.out_prefix))


if __name__ == "__main__":
    main()
