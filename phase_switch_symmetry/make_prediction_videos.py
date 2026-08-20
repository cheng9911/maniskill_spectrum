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

# ---- Models to overlay (label -> (display, color)) ------------------------
MODELS = {
    "Empirical isolated": ("Ground truth", "#000000"),
    "Frame-weighted": ("Frame-weighted  P=w(s)I", "#E69F00"),
    "Phase scalar GP": ("Phase scalar GP", "#0072B2"),
    "TP-GMM SE(2)": ("TP-GMM SE(2)", "#CC79A7"),
    "Pdiag finite": ("Pdiag finite", "#009E73"),
}

PHASE_NAMES = ["Align keyed", "Enter key", "Unlock yaw", "Circular insert"]

# Zero-response reference (not a fitted model): the peg ignoring the intervention.
NOMINAL = ("Nominal (ignore)", "#888888")


def load_profiles(csv_path: Path):
    df = pd.read_csv(csv_path)
    # All models share the same progress grid; take one model's sorted axis.
    first = list(MODELS)[0]
    progress = df[df["model"] == first].sort_values("global_progress")["global_progress"].to_numpy()
    g_trans = {}
    g_yaw = {}
    for model in MODELS:
        sub = df[df["model"] == model].sort_values("global_progress")
        g_trans[model] = sub["g_translation"].to_numpy()
        g_yaw[model] = sub["g_yaw"].to_numpy()
    return progress, g_trans, g_yaw


def add_key(ax, cx, cy, yaw, color, alpha=1.0, lw=1.6, ls="-", zorder=6):
    t = Affine2D().rotate(yaw).translate(cx, cy) + ax.transData
    r = Rectangle(
        (-KEY_HALF_X, -KEY_HALF_Y),
        2 * KEY_HALF_X,
        2 * KEY_HALF_Y,
        facecolor="none" if ls != "-" else color,
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
    # Outer ring
    ax.add_patch(
        Circle(
            (0, 0),
            SOCKET_OUTER_RADIUS,
            facecolor="#e8e8e8",
            edgecolor="#888888",
            lw=1.2,
            zorder=1,
        )
    )
    # Circular bore
    ax.add_patch(
        Circle(
            (0, 0),
            BORE_INNER_RADIUS,
            facecolor="#cfe8e4",
            edgecolor="#1b9e77",
            lw=1.2,
            zorder=2,
        )
    )
    # Rectangular keyed gate opening, rotated by the socket-yaw intervention
    kh_x = KEY_HALF_X + GATE_CLEARANCE
    kh_y = KEY_HALF_Y + GATE_CLEARANCE
    t = Affine2D().rotate(dpsi) + ax.transData
    ax.add_patch(
        Rectangle(
            (-kh_x, -kh_y),
            2 * kh_x,
            2 * kh_y,
            facecolor="#b3d9ff",
            edgecolor="#1f77b4",
            lw=1.8,
            transform=t,
            zorder=3,
        )
    )


def _draw_peg_overlay(ax, du, dv, dpsi, s_idx, progress, g_trans, g_yaw, nom_color):
    """Draw socket + nominal + all model pegs for one context at progress index."""
    draw_socket(ax, dpsi)
    add_peg(ax, 0.0, 0.0, 0.0, nom_color, alpha=0.9, lw=1.6, ls="--", zorder=4)
    for model, (label, color) in MODELS.items():
        cx = g_trans[model][s_idx] * du
        cy = g_trans[model][s_idx] * dv
        yaw = g_yaw[model][s_idx] * dpsi
        if model == "Empirical isolated":
            add_peg(ax, cx, cy, yaw, color, alpha=1.0, lw=3.0)
        else:
            add_peg(ax, cx, cy, yaw, color, alpha=0.85, lw=1.6)


def make_animation(
    progress,
    g_trans,
    g_yaw,
    intervention,
    out_path: Path,
    fps: float = 12.0,
):
    du, dv, dpsi = intervention  # meters, meters, radians
    deg = np.degrees(dpsi)
    # Spatial view limits: fit translation + socket radius.
    trans_extent = max(abs(du), abs(dv), 0.004)
    lim = max(0.055, trans_extent + SOCKET_OUTER_RADIUS + 0.008)

    fig = plt.figure(figsize=(11.6, 5.4))
    ax_sp = fig.add_axes([0.015, 0.07, 0.40, 0.88])
    ax_yaw = fig.add_axes([0.53, 0.57, 0.45, 0.32])
    ax_tr = fig.add_axes([0.53, 0.10, 0.45, 0.32])

    # ---- Curve panels (static curves, drawn once) -------------------------
    n = len(progress)
    nom_label, nom_color = NOMINAL
    zeros = np.zeros_like(progress)
    ax_yaw.plot(progress, zeros, color=nom_color, lw=1.4, ls="--", label=nom_label, zorder=2)
    ax_tr.plot(progress, zeros, color=nom_color, lw=1.4, ls="--", zorder=2)
    for model, (label, color) in MODELS.items():
        lw = 2.6 if model == "Empirical isolated" else 1.4
        ax_yaw.plot(progress, g_yaw[model], color=color, lw=lw, label=label, zorder=3)
        ax_tr.plot(progress, g_trans[model], color=color, lw=lw, zorder=3)
    # phase shading
    for a in (ax_yaw, ax_tr):
        for i in range(4):
            a.axvspan(i / 4, (i + 1) / 4, color="0.90", alpha=0.35, zorder=0)
        a.set_xlim(0, 1)
        a.set_ylim(-0.15, 1.25)
        a.set_xticks([0.125, 0.375, 0.625, 0.875])
        a.set_xticklabels(PHASE_NAMES, fontsize=7)
        a.tick_params(labelsize=7)
    ax_yaw.set_ylabel(r"$g_{\mathrm{yaw}}(s)$", fontsize=9)
    ax_tr.set_ylabel(r"$g_{\mathrm{trans}}(s)$", fontsize=9)
    ax_yaw.set_title("Axial-yaw relevance", loc="left", fontsize=8, fontweight="bold")
    ax_tr.set_title("Translation relevance", loc="left", fontsize=8, fontweight="bold")
    ax_yaw.legend(fontsize=6.2, frameon=False, loc="lower right", ncol=2)
    # moving markers
    mk_yaw = ax_yaw.axvline(progress[0], color="k", lw=1.0, ls="--", alpha=0.6)
    mk_tr = ax_tr.axvline(progress[0], color="k", lw=1.0, ls="--", alpha=0.6)
    # clearance boundary marker
    ax_yaw.axvline(0.5, color="0.55", lw=0.9, ls=":", alpha=0.7)
    ax_tr.axvline(0.5, color="0.55", lw=0.9, ls=":", alpha=0.7)
    ax_yaw.text(
        0.5, 1.16, "key clearance", color="0.45", fontsize=6.5, ha="center"
    )

    # ---- Spatial panel -----------------------------------------------------
    ax_sp.set_xlim(-lim, lim)
    ax_sp.set_ylim(-lim, lim)
    ax_sp.set_aspect("equal")
    ax_sp.set_xlabel("x (m)", fontsize=8)
    ax_sp.set_ylabel("y (m)", fontsize=8)
    ax_sp.tick_params(labelsize=7)
    ax_sp.grid(False)
    ax_sp.set_axisbelow(True)

    # hand-built legend swatches in spatial panel
    handles = []
    handles.append(plt.Line2D([], [], color=nom_color, lw=2, ls="--", label=nom_label))
    for model, (label, color) in MODELS.items():
        handles.append(
            plt.Line2D([], [], color=color, lw=3, label=label)
        )
    ax_sp.legend(handles=handles, fontsize=6.2, frameon=False, loc="lower left", ncol=2)

    def update(i):
        ax_sp.cla()
        ax_sp.set_xlim(-lim, lim)
        ax_sp.set_ylim(-lim, lim)
        ax_sp.set_aspect("equal")
        ax_sp.set_xlabel("x (m)", fontsize=8)
        ax_sp.set_ylabel("y (m)", fontsize=8)
        ax_sp.tick_params(labelsize=7)
        ax_sp.set_title(
            f"Predicted peg response — socket intervention "
            f"yaw={deg:+.0f}°, Δxy=({du*1e3:+.0f},{dv*1e3:+.0f}) mm",
            loc="left",
            fontsize=8.5,
            fontweight="bold",
        )
        _draw_peg_overlay(ax_sp, du, dv, dpsi, i, progress, g_trans, g_yaw, nom_color)
        s = progress[i]
        phase = PHASE_NAMES[int(i * 4 // n)]
        ax_sp.text(
            0.0, -lim + 0.006, phase, fontsize=9, fontweight="bold",
            color="#555555", ha="center",
        )
        mk_yaw.set_xdata([s, s])
        mk_tr.set_xdata([s, s])
        return []

    anim = animation.FuncAnimation(fig, update, frames=n, interval=1000.0 / fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Force 4:2:0 chroma subsampling: libx264 defaults to yuv444p here, which many
    # players (PowerPoint / WMP / QuickTime / hw decoders) render as a black frame.
    writer = animation.FFMpegWriter(
        fps=fps, codec="libx264", bitrate=2500,
        extra_args=["-pix_fmt", "yuv420p"],
    )
    anim.save(str(out_path), writer=writer)
    plt.close(fig)
    return dict(frames=n, out=str(out_path))


def context_label(du, dv, dpsi):
    if abs(dpsi) > 1e-9:
        return f"yaw {np.degrees(dpsi):+.0f}°"
    if abs(du) > 1e-9:
        return f"Δx {du*1e3:+.0f} mm"
    return f"Δy {dv*1e3:+.0f} mm"


def make_grid_animation(
    progress, g_trans, g_yaw, contexts, labels, out_path: Path, fps: float = 12.0
):
    """All held-out interventions side by side (2x4 grid of top-down panels)."""
    n = len(progress)
    lim = 0.055
    nom_label, nom_color = NOMINAL
    ncol = 4
    fig = plt.figure(figsize=(16, 8.0))
    axes = []
    for k, label in enumerate(labels):
        r, c = divmod(k, ncol)
        ax = fig.add_axes([0.02 + c * 0.243, 0.20 + (1 - r) * 0.40, 0.225, 0.34])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        axes.append(ax)

    fig.suptitle(
        "Predicted peg response across all 8 held-out interventions",
        fontsize=13, fontweight="bold", y=0.99,
    )
    prog_txt = fig.text(
        0.5, 0.955, "", ha="center", fontsize=11, fontweight="bold", color="#333333"
    )
    handles = [plt.Line2D([], [], color=nom_color, lw=2, ls="--", label=nom_label)]
    for model, (label, color) in MODELS.items():
        handles.append(plt.Line2D([], [], color=color, lw=3, label=label))
    fig.legend(
        handles=handles, fontsize=8.5, frameon=False, loc="lower center", ncol=6,
        bbox_to_anchor=(0.5, 0.02),
    )

    def update(i):
        s = progress[i]
        phase = PHASE_NAMES[int(i * 4 // n)]
        for ax, label, (du, dv, dpsi) in zip(axes, labels, contexts):
            ax.cla()
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(label, fontsize=11, fontweight="bold", pad=5)
            _draw_peg_overlay(ax, du, dv, dpsi, i, progress, g_trans, g_yaw, nom_color)
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
        "--profiles",
        type=Path,
        default=Path(
            "phase_switch_symmetry_baselines/phase_switch_baseline_profiles.csv"
        ),
    )
    parser.add_argument(
        "--arrays",
        type=Path,
        default=Path("phase_switch_symmetry_baselines/phase_switch_baseline_arrays.npz"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("phase_switch_symmetry_videos"))
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument(
        "--preset",
        type=str,
        default="yaw_p30",
        choices=[
            "yaw_p30", "yaw_m30", "yaw_p15", "yaw_m15",
            "trans_n15", "trans_p15_x", "trans_m15_y", "trans_p15_y",
            "mixed",
        ],
    )
    parser.add_argument("--grid", action="store_true", help="Render the 8-context grid.")
    args = parser.parse_args()

    progress, g_trans, g_yaw = load_profiles(args.profiles)

    if args.grid:
        arr = np.load(args.arrays)
        contexts = [tuple(map(float, c)) for c in arr["test_contexts"]]
        labels = [context_label(*c) for c in contexts]
        info = make_grid_animation(
            progress, g_trans, g_yaw, contexts, labels,
            args.out_dir / "19_grid_all8.mp4", fps=args.fps,
        )
        print(f"wrote {info['out']}  frames={info['frames']}", flush=True)
        return

    presets = {
        "yaw_p30": ([0.0, 0.0, np.deg2rad(30.0)], "10_model_prediction_yaw_p30.mp4"),
        "yaw_m30": ([0.0, 0.0, np.deg2rad(-30.0)], "11_model_prediction_yaw_m30.mp4"),
        "trans_n15": ([-0.015, 0.0, 0.0], "12_model_prediction_trans_n15.mp4"),
        "mixed": ([-0.015, 0.0, np.deg2rad(30.0)], "13_model_prediction_mixed.mp4"),
        "yaw_p15": ([0.0, 0.0, np.deg2rad(15.0)], "14_model_prediction_yaw_p15.mp4"),
        "yaw_m15": ([0.0, 0.0, np.deg2rad(-15.0)], "15_model_prediction_yaw_m15.mp4"),
        "trans_p15_x": ([0.015, 0.0, 0.0], "16_model_prediction_trans_p15_x.mp4"),
        "trans_m15_y": ([0.0, -0.015, 0.0], "17_model_prediction_trans_m15_y.mp4"),
        "trans_p15_y": ([0.0, 0.015, 0.0], "18_model_prediction_trans_p15_y.mp4"),
    }
    intervention, filename = presets[args.preset]

    info = make_animation(
        progress,
        g_trans,
        g_yaw,
        intervention,
        args.out_dir / filename,
        fps=args.fps,
    )
    print(f"wrote {info['out']}  frames={info['frames']}", flush=True)


if __name__ == "__main__":
    main()
