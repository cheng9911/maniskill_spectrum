from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D

# ---- Task geometry (mirrors phase_switch_symmetry_env.py) -----------------
KEY_HALF_X = 0.021
KEY_HALF_Y = 0.014
SHAFT_RADIUS = 0.013
BORE_INNER_RADIUS = 0.030
SOCKET_OUTER_RADIUS = 0.070
GATE_CLEARANCE = 0.002

# ---- Few-shot models (label -> (display, color)) ---------------------------
MODELS = {
    "Full operator": ("Full operator", "#D55E00"),
    "Generic RBF": ("Generic RBF", "#0072B2"),
    "Pdiag finite": ("Pdiag finite", "#009E73"),
}
GT_COLOR = "#000000"
GT_LABEL = "Ground truth"

PHASE_NAMES = ["Align keyed", "Enter key", "Unlock yaw", "Circular insert"]

# Which protocol each sample size uses (N<30 uses `random` subsets, N=30 = all).
def protocol_for(sample_size: int) -> str:
    return "all" if sample_size == 30 else "random"


def load_fewshot(base: Path, sample_sizes=(5, 30), n_shadow=12):
    """Pool per-seed fits into mean profile + a deterministic shadow sample."""
    seeds = sorted(
        d.name for d in base.glob("seed_*") if (d / "fewshot_results.csv").exists()
    )
    progress = None
    gt = None
    mean = {}
    shadow = {}
    err = {}
    for ss in sample_sizes:
        pr = protocol_for(ss)
        # model -> list of (K,100,3) profile blocks and (K,) error blocks
        prof_blocks = {}
        err_blocks = {}
        for sd in seeds:
            df = pd.read_csv(base / sd / "fewshot_results.csv")
            arr = np.load(base / sd / "fewshot_profiles.npz")
            if progress is None:
                progress = arr["progress"]
                gt = arr["target_profile"]
            P = arr["profiles"]
            for m in MODELS:
                sub = df[
                    (df.sample_size == ss) & (df.protocol == pr) & (df.model == m)
                ]
                if len(sub) == 0:
                    continue
                prof_blocks.setdefault(m, []).append(P[sub.profile_index.to_numpy()])
                err_blocks.setdefault(m, []).append(sub.task_error_mean_mm_equiv.to_numpy())
        mean[ss] = {}
        shadow[ss] = {}
        err[ss] = {}
        for m in MODELS:
            if m not in prof_blocks:
                continue
            K = np.concatenate(prof_blocks[m], axis=0)  # (K,100,3)
            E = np.concatenate(err_blocks[m])
            mean[ss][m] = K.mean(axis=0)
            # deterministic, evenly-spaced shadow sample (spread across fits)
            k = K.shape[0]
            idx = np.linspace(0, k - 1, min(n_shadow, k)).astype(int)
            shadow[ss][m] = K[idx]
            err[ss][m] = (float(E.mean()), float(E.std()))
    return progress, gt, mean, shadow, err


def add_key(ax, cx, cy, yaw, color, alpha=1.0, lw=1.6, ls="-", zorder=6):
    t = Affine2D().rotate(yaw).translate(cx, cy) + ax.transData
    r = Rectangle(
        (-KEY_HALF_X, -KEY_HALF_Y),
        2 * KEY_HALF_X,
        2 * KEY_HALF_Y,
        facecolor="none",
        edgecolor=color,
        lw=lw,
        ls=ls,
        alpha=alpha,
        transform=t,
        zorder=zorder,
    )
    ax.add_patch(r)


def add_peg(ax, cx, cy, yaw, color, alpha=1.0, lw=1.6, ls="-", zorder=6):
    ax.add_patch(
        Circle(
            (cx, cy),
            SHAFT_RADIUS,
            facecolor=color,
            edgecolor=color,
            alpha=0.45 * alpha if ls == "-" else 0.12 * alpha,
            lw=0.6,
            zorder=zorder - 1,
        )
    )
    add_key(ax, cx, cy, yaw, color, alpha=alpha, lw=lw, ls=ls, zorder=zorder)


def draw_socket(ax, dpsi):
    ax.add_patch(
        Circle(
            (0, 0), SOCKET_OUTER_RADIUS,
            facecolor="#e8e8e8", edgecolor="#888888", lw=1.2, zorder=1,
        )
    )
    ax.add_patch(
        Circle(
            (0, 0), BORE_INNER_RADIUS,
            facecolor="#cfe8e4", edgecolor="#1b9e77", lw=1.2, zorder=2,
        )
    )
    kh_x = KEY_HALF_X + GATE_CLEARANCE
    kh_y = KEY_HALF_Y + GATE_CLEARANCE
    t = Affine2D().rotate(dpsi) + ax.transData
    ax.add_patch(
        Rectangle(
            (-kh_x, -kh_y), 2 * kh_x, 2 * kh_y,
            facecolor="#b3d9ff", edgecolor="#1f77b4", lw=1.8, transform=t, zorder=3,
        )
    )


def _alpha_to_pose(alpha, du, dv, dpsi):
    """First-order response: peg pose = diag(alpha) . intervention."""
    return alpha[0] * du, alpha[1] * dv, alpha[2] * dpsi


def make_fewshot_animation(
    progress, gt, mean, shadow, err, intervention, out_path: Path, fps: float = 12.0
):
    du, dv, dpsi = intervention
    deg = np.degrees(dpsi)
    n = len(progress)
    ncol = 3
    nrow = 2
    sample_sizes = [5, 30]
    trans_extent = max(abs(du), abs(dv), 0.004)
    lim = max(0.10, trans_extent + SOCKET_OUTER_RADIUS + 0.02)

    fig = plt.figure(figsize=(15.0, 8.8))
    axes = []  # (ax, model, ss)
    for r, ss in enumerate(sample_sizes):
        for c, m in enumerate(MODELS):
            ax = fig.add_axes(
                [0.035 + c * 0.323, 0.20 + (1 - r) * 0.395, 0.285, 0.345]
            )
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            axes.append((ax, m, ss))

    fig.suptitle(
        "Few-shot sample size: N=5 vs N=30 predicted peg response (socket yaw "
        f"{deg:+.0f}°, Δx {du*1e3:+.0f} mm) — shadow = spread across random subsets",
        fontsize=13, fontweight="bold", y=0.995,
    )
    prog_txt = fig.text(0.5, 0.935, "", ha="center", fontsize=12, fontweight="bold",
                        color="#333333")

    # legend strip
    handles = [plt.Line2D([], [], color=GT_COLOR, lw=3, label=GT_LABEL)]
    for m, (label, color) in MODELS.items():
        handles.append(plt.Line2D([], [], color=color, lw=3, label=label))
    handles.append(
        plt.Line2D([], [], color="0.55", lw=1.0, ls="-", label="individual fits (spread)")
    )
    fig.legend(handles=handles, fontsize=9, frameon=False, loc="lower center",
               ncol=5, bbox_to_anchor=(0.5, 0.02))

    def update(i):
        s = progress[i]
        phase = PHASE_NAMES[int(i * 4 // n)]
        for ax, m, ss in axes:
            ax.cla()
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            draw_socket(ax, dpsi)
            # spread shadow: individual fits behind the mean peg
            if m in shadow[ss]:
                for alpha in shadow[ss][m]:
                    cx, cy, yaw = _alpha_to_pose(alpha[i], du, dv, dpsi)
                    add_key(ax, cx, cy, yaw, MODELS[m][1], alpha=0.22, lw=0.9, zorder=4)
            # ground-truth peg
            gx, gy, gyaw = _alpha_to_pose(gt[i], du, dv, dpsi)
            add_peg(ax, gx, gy, gyaw, GT_COLOR, alpha=1.0, lw=2.8, zorder=6)
            # mean model peg
            cx, cy, yaw = _alpha_to_pose(mean[ss][m][i], du, dv, dpsi)
            add_peg(ax, cx, cy, yaw, MODELS[m][1], alpha=0.9, lw=2.0, zorder=7)
            # title
            em, es = err[ss][m]
            title = f"{MODELS[m][0]} — N={ss}  (err {em:.1f}±{es:.1f} mm)"
            ax.set_title(title, fontsize=10.5, fontweight="bold", pad=4)
        prog_txt.set_text(f"phase s = {s:.2f}   —   {phase}")
        return []

    anim = animation.FuncAnimation(fig, update, frames=n, interval=1000.0 / fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(
        fps=fps, codec="libx264", bitrate=3500,
        extra_args=["-pix_fmt", "yuv420p"],
    )
    anim.save(str(out_path), writer=writer)
    plt.close(fig)
    return dict(frames=n, out=str(out_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fewshot-dir",
        type=Path,
        default=Path("phase_switch_symmetry_multiseed/fewshot"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("phase_switch_symmetry_videos"))
    parser.add_argument("--fps", type=float, default=12.0)
    args = parser.parse_args()

    progress, gt, mean, shadow, err = load_fewshot(args.fewshot_dir)

    # Mixed intervention (Δx=-15 mm + yaw +30°): exercises both translation
    # (RBF under-fits alpha_x) and yaw (RBF under-rotates alpha_yaw).
    intervention = [-0.015, 0.0, np.deg2rad(30.0)]
    info = make_fewshot_animation(
        progress, gt, mean, shadow, err, intervention,
        args.out_dir / "20_fewshot_N5_vs_N30.mp4", fps=args.fps,
    )
    print(f"wrote {info['out']}  frames={info['frames']}", flush=True)


if __name__ == "__main__":
    main()
