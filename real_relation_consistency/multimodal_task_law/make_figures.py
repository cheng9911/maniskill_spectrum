from __future__ import annotations

"""Generate the 4 paper figures + source CSVs (plan section 9).

Figure 1  four modes' 3D actual peg trajectories (0 deg and +10 deg) with the
          hole geometry and event points.
Figure 2  raw pitch response + identified scalar response law alpha(s) per mode
          (all 5 blocks), segmented into free / enter / insert.
Figure 3  4x4 source->target held-out error matrix (enter vs insert segment).
Figure 4  physical-execution success (numerator/denominator) + insertion depth /
          axis error per method.

Every figure is exported to PNG + PDF + SVG; every table to a source CSV.
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
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib import cm  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

PEG_HALF_LENGTH = 0.045
MODES = ["lift_early", "lift_late", "diagonal_early", "diagonal_late"]
MODE_LABEL = {
    "lift_early": "lift / early",
    "lift_late": "lift / late",
    "diagonal_early": "diagonal / early",
    "diagonal_late": "diagonal / late",
}
SEG_LABEL = ["free approach", "enter", "insert"]


def peg_axis(pq):
    """pq: (T,7) peg pose [x,y,z,w,x,y,z] -> (T,3) peg z-axis."""
    from transforms3d.quaternions import quat2mat
    return np.array([quat2mat(q[3:]) @ np.array([0.0, 0.0, 1.0]) for q in pq])


def peg_tip(pq):
    return pq[:, :3] - PEG_HALF_LENGTH * peg_axis(pq)


def load_events():
    ev = json.load(open(HERE / "ev_core.json"))["episodes"]
    return {e["episode_id"]: e for e in ev}


def load_models():
    return json.load(open(HERE / "id_core.json"))


def load_transfer():
    return json.load(open(HERE / "tr_core.json"))["records"]


def load_execution():
    return json.load(open(HERE / "ex_core.json"))["results"]


def load_yaw():
    return json.load(open(HERE / "yaw_50.json"))["episodes"]


# --------------------------------------------------------------------------- #
# Figure 1
# --------------------------------------------------------------------------- #
def fig1(protocol, outdir):
    center = np.asarray(protocol["task"]["task_anchor"], dtype=np.float64)
    ev = load_events()
    by_key = {}
    for e in ev.values():
        by_key[(e["mode_id"], e["block_id"], e["pitch_deg"])] = e

    fig, axes = plt.subplots(2, 2, figsize=(9, 8), subplot_kw={"projection": "3d"})
    with h5py.File(HERE / "core_260.h5", "r") as f:
        for ax, mode in zip(axes.ravel(), MODES):
            for pitch, color, lw in [(0.0, "tab:blue", 1.6), (10.0, "tab:red", 1.6)]:
                e = by_key[(mode, 0, pitch)]
                pq = f[e["episode_id"]]["peg_pose"][:]
                tip = peg_tip(pq)
                ax.plot(tip[:, 0], tip[:, 1], tip[:, 2], color=color, lw=lw,
                        label=f"{pitch:+.0f}°")
                # event markers
                for name, mk in [("E1", "o"), ("E2", "s"), ("E3", "^")]:
                    s = e[name]
                    if s >= 0:
                        ax.scatter(tip[s, 0], tip[s, 1], tip[s, 2], marker=mk,
                                   color=color, s=28)
            # socket axis (vertical, tilted by +10 deg about y)
            th = np.deg2rad(10.0)
            R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0],
                          [-np.sin(th), 0, np.cos(th)]])
            z = np.linspace(-0.06, 0.12, 20)
            axis_line = center[None, :] + z[:, None] * (R @ np.array([0, 0, 1]))[None, :]
            ax.plot(axis_line[:, 0], axis_line[:, 1], axis_line[:, 2], "k--", lw=1.0)
            ax.set_title(MODE_LABEL[mode], fontsize=9)
            ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
            ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Figure 1 — actual peg-tip trajectories (block 0): 0° vs +10° pitch")
    fig.tight_layout()
    save(fig, outdir / "fig1_mode_trajectories")


# --------------------------------------------------------------------------- #
# Figure 2
# --------------------------------------------------------------------------- #
def fig2(protocol, outdir):
    d = load_models()
    models = d["models"]; responses = d["responses"]
    n = 150
    prog = np.arange(n) / (n - 1)
    seg_bound = [50 / 150, 100 / 150]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    # (a) identified alpha(s)
    ax = axes[0]
    cmap = cm.get_cmap("tab10")
    for mi, mode in enumerate(MODES):
        alphas = []
        for b in range(protocol["core_blocks"]):
            k = f"{mode}_{b}"
            if k in models:
                alphas.append(np.array(models[k]["alpha"]))
        for a in alphas:
            ax.plot(prog, a, color=cmap(mi), lw=0.6, alpha=0.35)
        ax.plot(prog, np.mean(alphas, axis=0), color=cmap(mi), lw=2.0,
                label=MODE_LABEL[mode])
    for s in seg_bound:
        ax.axvline(s, color="gray", ls=":", lw=0.8)
    ax.set_ylabel(r"identified $\alpha_{\mathrm{pitch}}(s)$")
    ax.set_ylim(0, 1.3)
    ax.set_title("(a) frozen scalar pitch response law per mode (5 blocks each)")
    ax.legend(fontsize=8, ncol=2)

    # (b) raw central-difference-like k_pitch = r_pitch / pitch (deg/deg)
    ax = axes[1]
    for mi, mode in enumerate(MODES):
        rows = [r for r in responses if r["mode_id"] == mode and abs(r["pitch_deg"]) > 1e-9]
        curves = [np.array(r["r_pitch_deg"]) / r["pitch_deg"] for r in rows]
        ax.plot(prog, np.mean(curves, axis=0), color=cmap(mi), lw=2.0,
                label=MODE_LABEL[mode])
        ax.fill_between(prog, np.mean(curves, axis=0) - np.std(curves, axis=0),
                        np.mean(curves, axis=0) + np.std(curves, axis=0),
                        color=cmap(mi), alpha=0.15)
    for s in seg_bound:
        ax.axvline(s, color="gray", ls=":", lw=0.8)
    for txt, x0, x1 in zip(SEG_LABEL, [0, 1 / 3, 2 / 3], [1 / 3, 2 / 3, 1]):
        ax.text((x0 + x1) / 2, 0.05, txt, ha="center", fontsize=8, color="dimgray")
    ax.set_ylabel(r"raw $k_{\mathrm{pitch}}$ (deg / deg)")
    ax.set_xlabel("normalized progress (geometric events)")
    ax.set_ylim(-0.4, 1.3)
    ax.set_title("(b) raw pitch response per degree of intervention")
    ax.legend(fontsize=8, ncol=2)

    fig.tight_layout()
    save(fig, outdir / "fig2_response_law")


# --------------------------------------------------------------------------- #
# Figure 3
# --------------------------------------------------------------------------- #
def fig3(protocol, outdir):
    recs = load_transfer()
    metrics = [("seg1__axis_rmse_deg", "enter axis RMSE (°)"),
               ("seg2__axis_rmse_deg", "insert axis RMSE (°)"),
               ("terminal__pos_rmse_mm", "terminal pos RMSE (mm)")]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (metric, title) in zip(axes, metrics):
        mat = np.full((4, 4), np.nan)
        for i, s in enumerate(MODES):
            for j, t in enumerate(MODES):
                vals = [r[metric] for r in recs if r["method"] == "source"
                        and r["source"] == s and r["target"] == t and metric in r]
                mat[i, j] = np.mean(vals) if vals else np.nan
        vmax = np.nanmax(mat)
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=vmax)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels([m.replace("_", "\n") for m in MODES], fontsize=7)
        ax.set_yticklabels([m.replace("_", "\n") for m in MODES], fontsize=7)
        ax.set_xlabel("target mode"); ax.set_ylabel("source mode")
        for i in range(4):
            for j in range(4):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            fontsize=8, color="white" if mat[i, j] > vmax * 0.5 else "black")
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Figure 3 — 4×4 source→target held-out error (method = frozen source law)")
    fig.tight_layout()
    save(fig, outdir / "fig3_transfer_matrix")


# --------------------------------------------------------------------------- #
# Figure 4
# --------------------------------------------------------------------------- #
def fig4(protocol, outdir):
    rs = load_execution()
    methods = ["no_adapt", "rigid", "source", "target"]
    mlabels = {"no_adapt": "P=0", "rigid": "P=I", "source": "source\nlaw", "target": "target\nlaw"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ax = axes[0]
    for i, m in enumerate(methods):
        rr = [r for r in rs if r["method"] == m]
        succ = sum(r["success"] for r in rr)
        ax.bar(i, succ / len(rr), color="tab:blue" if m != "no_adapt" else "tab:gray")
        ax.text(i, succ / len(rr) + 0.01, f"{succ}/{len(rr)}", ha="center", fontsize=9)
    ax.set_xticks(range(len(methods))); ax.set_xticklabels([mlabels[m] for m in methods])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("success rate (first attempt)")
    ax.set_title("(a) executed-transfer success")

    ax = axes[1]
    for m, color in zip(methods, ["tab:gray", "tab:orange", "tab:green", "tab:purple"]):
        rr = [r for r in rs if r["method"] == m]
        ax.scatter([r["depth"] for r in rr], [r["axis_err_deg"] for r in rr],
                   color=color, alpha=0.6, label=mlabels[m], s=20)
    ax.axvline(0.072, color="r", ls="--", lw=1, label="depth success threshold")
    ax.set_xlabel("terminal insertion depth (m)")
    ax.set_ylabel("terminal axis error (°)")
    ax.set_title("(b) terminal depth vs axis error")
    ax.legend(fontsize=8)
    fig.suptitle("Figure 4 — physical execution of generated transfer trajectories")
    fig.tight_layout()
    save(fig, outdir / "fig4_execution")


# --------------------------------------------------------------------------- #
# Source CSVs (tables 1-3)
# --------------------------------------------------------------------------- #
def write_csvs(protocol, outdir):
    # Table 1: mode differences + collection success
    ev = load_events()
    with open(outdir / "table1_mode_collection.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode_id", "block_id", "pitch_deg", "success_terminal",
                    "terminal_depth", "terminal_lateral", "E1", "E2", "E3", "n_steps"])
        for e in ev.values():
            w.writerow([e["mode_id"], e["block_id"], e["pitch_deg"],
                        e["success_terminal"], f"{e['terminal_depth']:.4f}",
                        f"{e['terminal_lateral']:.4f}", e["E1"], e["E2"], e["E3"],
                        e["n_steps"]])

    # Table 2: response differences per segment
    d = load_models()
    responses = d["responses"]
    with open(outdir / "table2_response_k.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode_id", "segment", "k_pitch_mean", "k_pitch_std",
                    "k_spin_abs_mean", "n_conditions"])
        for mode in MODES:
            rows = [r for r in responses if r["mode_id"] == mode and abs(r["pitch_deg"]) > 1e-9]
            for seg in range(3):
                ks = [np.array(r["r_pitch_deg"])[seg * 50:(seg + 1) * 50] / r["pitch_deg"]
                      for r in rows]
                ks_all = np.concatenate(ks)
                sp = np.concatenate([np.abs(np.array(r["r_spin_deg"])[seg * 50:(seg + 1) * 50])
                                     for r in rows])
                w.writerow([mode, SEG_LABEL[seg], f"{np.mean(ks_all):.4f}",
                            f"{np.std(ks_all):.4f}", f"{np.mean(sp):.4f}", len(rows)])

    # Table 3: cross-mode execution results
    rs = load_execution()
    with open(outdir / "table3_execution.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "target", "theta_deg", "success", "depth",
                    "axis_err_deg", "lateral_mm"])
        for r in rs:
            w.writerow([r["method"], r["target"], r["theta_deg"], r["success"],
                        f"{r['depth']:.4f}", f"{r['axis_err_deg']:.3f}",
                        f"{r['lateral_mm']:.3f}"])

    # Yaw challenge table
    yaw = load_yaw()
    yaw_deg = protocol["yaw_deg"]
    with open(outdir / "table4_yaw.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["yaw_mode", "yaw_deg", "success", "assembly_complete", "n"])
        for ym in ["yaw_fixed", "yaw_follow"]:
            for ci, th in enumerate(yaw_deg):
                rr = [e for e in yaw if e["yaw_mode"] == ym and e["condition_index"] == ci]
                if rr:
                    w.writerow([ym, th, sum(e["success"] for e in rr),
                                sum(e["assembly_complete"] for e in rr), len(rr)])


def save(fig, base):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(f"{base}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {base}.png/.pdf/.svg")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=HERE / "figures")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, str(HERE))
    from generate_contexts import load_protocol

    protocol = load_protocol(HERE / "protocol.json")
    fig1(protocol, args.outdir)
    fig2(protocol, args.outdir)
    fig3(protocol, args.outdir)
    fig4(protocol, args.outdir)
    write_csvs(protocol, args.outdir)
    print("done")


if __name__ == "__main__":
    main()
