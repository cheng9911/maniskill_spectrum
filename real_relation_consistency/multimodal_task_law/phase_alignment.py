from __future__ import annotations

"""Task-phase re-alignment analysis (answering: is the 0.72 vs 0.32 enter-segment
pitch-response difference a genuine strategy dependence, or a progress-coordinate
misalignment?).

For each demonstration we recover the alignment fraction
``alpha(t) = 1 - axis_angle_err(t)/|theta|`` (0 = not yet aligned, 1 = aligned to
the tilted socket) directly from the raw axis-angle error, then locate the
alignment window and compare, across the 4 modes, THREE coordinates:

  (a) wall-clock time (raw steps)          -> shows the timing shift
  (b) physical depth (peg-centre axial depth along the socket axis) -> shows the anchoring
  (c) task-phase coordinate (free / align / insert, resampled)      -> shows the shared structure

Output: fig5_alignment_phases.{png,pdf,svg} + phase_alignment_summary.csv
"""

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import cm  # noqa: E402
from transforms3d.quaternions import quat2mat  # noqa: E402

HERE = Path(__file__).resolve().parent

MODES = ["lift_early", "lift_late", "diagonal_early", "diagonal_late"]
MODE_LABEL = {
    "lift_early": "lift / early",
    "lift_late": "lift / late",
    "diagonal_early": "diagonal / early",
    "diagonal_late": "diagonal / late",
}
COLORS = ["tab:blue", "tab:cyan", "tab:red", "tab:orange"]
N = 50


def sock_axis(s7):
    return quat2mat(s7[3:]) @ np.array([0.0, 0.0, 1.0])


def load_episode_index(protocol, ev):
    idx = {}
    for e in ev:
        idx[(e["mode_id"], e["block_id"], round(e["pitch_deg"], 6))] = e
    return idx


def alignment_trace(g, theta_deg, center):
    """Return (alpha, depth) time series for one episode."""
    aae = np.asarray(g["axis_angle_err"][:])
    alpha = 1.0 - aae / np.deg2rad(abs(theta_deg))
    peg = np.asarray(g["peg_pose"][:])
    sock = np.asarray(g["socket_pose"][:])
    ax = sock_axis(sock[0])
    rel = peg[:, :3] - center
    depth = np.einsum("ij,j->i", rel, ax)
    return alpha, depth


def phase_curve(alpha, depth, E0, E3):
    """Resample alpha onto the 3-phase coordinate (free/align/insert)."""
    t_s = int(np.argmax(alpha > 0.05))
    t_e = int(np.argmax(alpha > 0.95))
    if t_s == 0 or t_e <= t_s or E3 <= t_e:
        return None, (t_s, t_e)
    out = np.concatenate([
        np.interp(np.linspace(E0, t_s, N), np.arange(len(alpha)), alpha),   # free
        np.interp(np.linspace(t_s, t_e, N), np.arange(len(alpha)), alpha),  # align
        np.interp(np.linspace(t_e, E3, N), np.arange(len(alpha)), alpha),   # insert
    ])
    return out, (t_s, t_e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    ap.add_argument("--h5", type=Path, default=HERE / "core_260.h5")
    ap.add_argument("--events", type=Path, default=HERE / "ev_core.json")
    ap.add_argument("--outdir", type=Path, default=HERE / "figures")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(HERE))
    from generate_contexts import load_protocol

    protocol = load_protocol(args.protocol)
    center = np.asarray(protocol["task"]["task_anchor"], dtype=np.float64)
    ev = json.load(open(args.events))["episodes"]
    index = load_episode_index(protocol, ev)

    cmap = cm.get_cmap("tab10")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    summary_rows = []

    with h5py.File(args.h5, "r") as f:
        for mi, mode in enumerate(MODES):
            # collect block-0 +10deg traces (and a couple blocks for spread)
            alphas_wc, depths_wc, phases = [], [], []
            half_depths, end_depths, starts, ends = [], [], [], []
            for b in range(protocol["core_blocks"]):
                for th in (10.0, -10.0):
                    e = index.get((mode, b, th))
                    if e is None:
                        continue
                    g = f[e["episode_id"]]
                    alpha, depth = alignment_trace(g, th, center)
                    t_s = int(np.argmax(alpha > 0.05))
                    t_e = int(np.argmax(alpha > 0.95))
                    if t_s == 0 or t_e <= t_s:
                        continue
                    ph, _ = phase_curve(alpha, depth, e["E0"], e["E3"])
                    if ph is not None:
                        phases.append(ph)
                    half = int(np.argmax(alpha > 0.5))
                    half_depths.append(depth[half])
                    end_depths.append(depth[t_e])
                    starts.append(t_s)
                    ends.append(t_e)
                    if b == 0 and th == 10.0:
                        alphas_wc.append(alpha)
                        depths_wc.append(depth)

            color = COLORS[mi]
            # (a) wall-clock
            if alphas_wc:
                ax = axes[0]
                T = len(alphas_wc[0])
                ax.plot(np.arange(T), alphas_wc[0], color=color, lw=2.0,
                        label=MODE_LABEL[mode])
            # (b) depth
            if depths_wc:
                ax = axes[1]
                ax.plot(depths_wc[0], alphas_wc[0], color=color, lw=2.0,
                        label=MODE_LABEL[mode])
            # (c) task-phase (mean over blocks/signs)
            if phases:
                ax = axes[2]
                mu = np.mean(phases, axis=0)
                std = np.std(phases, axis=0)
                phi = np.linspace(0, 1, 3 * N)
                ax.plot(phi, mu, color=color, lw=2.0, label=MODE_LABEL[mode])
                ax.fill_between(phi, mu - std, mu + std, color=color, alpha=0.12)

            summary_rows.append(dict(
                mode=mode,
                half_align_depth_m=float(np.mean(half_depths)),
                align_end_depth_m=float(np.mean(end_depths)),
                ramp_len_steps=float(np.mean(np.subtract(ends, starts))),
            ))

    axes[0].set_xlabel("raw step (wall-clock)")
    axes[0].set_ylabel(r"$\alpha_{\mathrm{pitch}}(t)$")
    axes[0].set_title("(a) vs wall-clock time — timing shift")
    axes[1].set_xlabel("peg-centre axial depth (m)")
    axes[1].set_ylabel(r"$\alpha_{\mathrm{pitch}}$")
    axes[1].set_title("(b) vs physical depth — anchoring difference")
    axes[1].invert_xaxis()
    axes[2].set_xlabel("task-phase coordinate φ (free / align / insert)")
    axes[2].set_ylabel(r"$\alpha_{\mathrm{pitch}}$")
    axes[2].set_title("(c) vs task phase — shared structure")
    for s in (1 / 3, 2 / 3):
        axes[2].axvline(s, color="gray", ls=":", lw=0.8)
    for ax in axes:
        ax.legend(fontsize=8, ncol=2, loc="lower right")
        ax.grid(alpha=0.25)
    fig.suptitle("Figure 5 — task-phase re-alignment of the pitch response across modes")
    fig.tight_layout()
    args.outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(args.outdir / f"fig5_alignment_phases.{ext}", bbox_inches="tight")
    plt.close(fig)

    with open(args.outdir / "phase_alignment_summary.csv", "w", newline="") as cf:
        w = csv.DictWriter(cf, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print("wrote fig5_alignment_phases.* + phase_alignment_summary.csv")
    for r in summary_rows:
        print(f"  {r['mode']:16s} half-align depth={r['half_align_depth_m']:.4f} m "
              f"end depth={r['align_end_depth_m']:.4f} m  ramp_len={r['ramp_len_steps']:.0f} steps")


if __name__ == "__main__":
    main()
