from __future__ import annotations

"""Geometric event extraction + two progress schemes for the multimodal rollouts.

Events are recovered from the RAW peg/socket poses (no solver labels, no fitted
law), following plan section 5.  All quantities are expressed in the socket
frame: ``axial`` = projection of a point onto the tilted socket axis, ``lateral``
= residual norm.  The peg tip (lowest geometric reference point) is the shaft
bottom-centre = peg_centre - PEG_HALF_LENGTH * peg_axis_world.

  E0  common post-grasp start  (first step of the shared 'approach' phase; all
      modes share the identical reach/grasp/lift code up to this point)
  E1  peg centre first PERSISTENTLY inside the hole-opening neighbourhood
      (lateral < 0.020 while still above the gate plane, axial > 0.065)
  E2  peg tip crosses the gate plane inside the pre-registered lateral channel
      (tip_axial <= 0.065 AND tip_lateral < CIRCULAR_GATE_INNER_RADIUS=0.015)
  E3  axial depth reaches the achievable terminal depth (axial <= 0.072)

Two progress schemes are produced per episode:
  (A) geometric-event progress: each segment E0->E1, E1->E2, E2->E3 resampled
      to N points by elapsed-time fraction (segment_progress in [0,1]).
  (B) raw time + cumulative TCP-translation arc-length (sensitivity check).
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

PEG_HALF_LENGTH = 0.045
GATE_Z_MAX = 0.065
CIRCULAR_GATE_INNER_RADIUS = 0.015
N_PER_SEGMENT = 50


def socket_axis_from_pose(socket_pose7: np.ndarray) -> np.ndarray:
    return quat2mat(socket_pose7[3:]) @ np.array([0.0, 0.0, 1.0])


def peg_axis_from_pose(peg_pose7: np.ndarray) -> np.ndarray:
    return quat2mat(peg_pose7[3:]) @ np.array([0.0, 0.0, 1.0])


def geometry(peg: np.ndarray, socket: np.ndarray):
    """peg, socket: (T,7) pose arrays [x,y,z,w,x,y,z]. Returns per-step dict."""
    axis = socket_axis_from_pose(socket[0])
    sock_center = socket[:, :3]
    rel = peg[:, :3] - sock_center
    axial = np.einsum("ij,j->i", rel, axis)
    lat = rel - axial[:, None] * axis[None, :]
    lateral = np.linalg.norm(lat, axis=1)
    pax = np.array([peg_axis_from_pose(p) for p in peg])
    tip = peg[:, :3] - PEG_HALF_LENGTH * pax
    tip_rel = tip - sock_center
    tip_axial = np.einsum("ij,j->i", tip_rel, axis)
    tip_lat = tip_rel - tip_axial[:, None] * axis[None, :]
    tip_lateral = np.linalg.norm(tip_lat, axis=1)
    return dict(
        axial=axial, lateral=lateral,
        tip_axial=tip_axial, tip_lateral=tip_lateral,
        peg_axis=pax,
    )


def detect_events(geom, solver_phase, ev):
    """Return (E0, E1, E2, E3) step indices, or -1 when missing."""
    T = len(geom["axial"])
    # E0: first approach step (solver_phase == 3). Fallback to step 0.
    E0 = int(np.argmax(solver_phase == 3)) if (solver_phase == 3).any() else 0
    lateral_lt = ev["E1"]["lateral_lt"]
    axial_gt = ev["E1"]["axial_gt"]
    persist = ev["E1"]["persist_steps"]

    # E1: persistent lateral entry while still above the gate
    in_nbhd = (geom["lateral"] < lateral_lt) & (geom["axial"] > axial_gt)
    E1 = -1
    run = 0
    for i in range(E0, T):
        if in_nbhd[i]:
            run += 1
            if run >= persist:
                E1 = i - persist + 1
                break
        else:
            run = 0

    # E2: peg tip crosses gate plane inside the lateral channel
    tip_lt = ev["E2"]["lateral_lt"]
    tip_axial_le = ev["E2"]["tip_axial_le"]
    crossed = (geom["tip_axial"] <= tip_axial_le) & (geom["tip_lateral"] < tip_lt)
    E2 = int(np.argmax(crossed)) if crossed.any() else -1
    if E2 < E1:
        E2 = -1

    # E3: axial depth reaches achievable terminal depth
    axial_le = ev["E3"]["axial_le"]
    reached = geom["axial"] <= axial_le
    E3 = int(np.argmax(reached)) if reached.any() else -1

    return E0, E1, E2, E3


def resample_segments(geom, events, n=N_PER_SEGMENT):
    """Resample axial/lateral/peg/tcp (passed via geom dict extra) onto the
    3-segment grid. Returns dict of (3*n,) or (3*n,D) arrays + segment_id +
    segment_progress. Each segment [E_k, E_{k+1}] (inclusive) is resampled by
    time fraction."""
    E0, E1, E2, E3 = events
    bounds = [E0, E1, E2, E3]
    seg_id = np.repeat(np.arange(3), n)
    seg_prog = np.tile(np.linspace(0.0, 1.0, n), 3)
    out = {"segment_id": seg_id, "segment_progress": seg_prog}
    for key, arr in geom.items():
        arr = np.asarray(arr, dtype=np.float64)
        # only resample per-step quantities; broadcast constant vectors untouched
        T = len(arr) if arr.ndim >= 1 else 1
        shape = (3 * n,) + (arr.shape[1:] if arr.ndim > 1 else ())
        res = np.full(shape, np.nan, dtype=np.float64)
        for k in range(3):
            a, b = bounds[k], bounds[k + 1]
            if a < 0 or b < 0 or b < a:
                continue
            idx = np.linspace(a, b, n)
            if arr.ndim == 1 and arr.shape[0] == T:
                # per-step scalar/vector whose length equals n_steps: interpolate
                res[k * n:(k + 1) * n] = np.interp(idx, np.arange(T), arr)
            elif arr.ndim == 2 and arr.shape[0] == T:
                res[k * n:(k + 1) * n] = np.column_stack([
                    np.interp(idx, np.arange(T), arr[:, d]) for d in range(arr.shape[1])
                ])
            elif arr.ndim == 1 and arr.shape[0] != T:
                # constant vector (e.g. socket_axis): broadcast
                res[k * n:(k + 1) * n] = arr
        out[key] = res
    return out


def process_stage(protocol, stage_path, out_prefix):
    p = protocol
    ev = p["events"]
    with h5py.File(stage_path, "r") as f:
        ep_ids = [k for k in f.keys() if k.startswith("episode")]
        rows = []
        resampled_groups = {}
        for k in ep_ids:
            g = f[k]
            peg = g["peg_pose"][:]
            sock = g["socket_pose"][:]
            tcp = g["tcp_pose"][:]
            solver_phase = np.asarray(g["solver_phase"][:])
            success = np.asarray(g["success"][:], bool)
            geom = geometry(peg, sock)
            geom["tcp_pose"] = tcp
            geom["peg_pose"] = peg
            E0, E1, E2, E3 = detect_events(geom, solver_phase, ev)
            # trailing consecutive success (hold diagnostic)
            hold = 0
            for s in reversed(success):
                if s:
                    hold += 1
                else:
                    break
            # arc-length progress up to E2 (sensitivity)
            if E2 > E0:
                d = np.linalg.norm(np.diff(tcp[E0:E2 + 1, :3], axis=0), axis=1)
                arc = np.concatenate([[0.0], np.cumsum(d)])
            else:
                arc = np.zeros(max(0, len(tcp) - E0) + 1)
            row = dict(
                episode_id=k,
                mode_id=str(g.attrs.get("mode_id", "")),
                yaw_mode=str(g.attrs.get("yaw_mode", "")),
                block_id=int(g.attrs["block_id"]),
                condition_index=int(g.attrs["condition_index"]),
                split=str(g.attrs.get("split", "")),
                pitch_deg=float(g.attrs["pitch_deg"]),
                E0=int(E0), E1=int(E1), E2=int(E2), E3=int(E3),
                missing=[int(x) for x in (E0, E1, E2, E3) if x < 0],
                order_ok=bool(E0 <= E1 <= E2 <= E3),
                n_steps=int(len(peg)),
                success_terminal=bool(success[-1]),
                hold_steps=int(hold),
                terminal_depth=float(geom["axial"][-1]),
                terminal_lateral=float(geom["lateral"][-1]),
                terminal_tip_axial=float(geom["tip_axial"][-1]),
                arc_to_E2=float(arc[-1]) if len(arc) else 0.0,
            )
            rows.append(row)
            resampled_groups[k] = resample_segments(geom, (E0, E1, E2, E3))

    with h5py.File(out_prefix + ".h5", "w") as rf:
        rf.attrs["protocol_sha256"] = p["content_sha256"]
        rf.attrs["n_per_segment"] = N_PER_SEGMENT
        for k, d in resampled_groups.items():
            grp = rf.create_group(k)
            for key, arr in d.items():
                grp.create_dataset(key, data=np.asarray(arr))
    with open(out_prefix + ".json", "w", encoding="utf-8") as mf:
        json.dump(dict(protocol_sha256=p["content_sha256"], episodes=rows), mf, indent=2)
    n_ok = sum(r["success_terminal"] for r in rows)
    n_order = sum(r["order_ok"] for r in rows)
    print(f"{stage_path.name}: {len(rows)} episodes, {n_ok} terminal-success, "
          f"{n_order} ordered events")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--stage-h5", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    process_stage(protocol, args.stage_h5, str(args.out_prefix))


if __name__ == "__main__":
    main()
