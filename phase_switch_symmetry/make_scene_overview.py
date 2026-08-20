from __future__ import annotations

"""Real-simulation-scene overview figure for the paper.

The user asked for a single figure showing how many simulation experiments were
run, made of REAL rendered simulation frames (not the abstract schematic). This
script:

  1. replays stored qpos + peg/socket poses from the .h5 rollouts into a fresh
     ManiSkill env and renders rgb_array frames with a camera re-aimed at the
     task (so the peg/socket are centered and large);
  2. composes those frames into a 2x4 grid: the keyed phase-switch mechanism
     (row 1) and the geometry variants behind each experiment group (row 2);
  3. annotates every panel with the experiment name + frozen counts from
     main.tex Table 1, and writes experiment_overview_scenes.{png,pdf}.

Run from phase_switch_symmetry/ (so phase_switch_symmetry_env is importable).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_videos import load_episode, make_env  # noqa: E402
from make_symmetry_transfer_videos import first_success_condition  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import torch  # noqa: E402
import sapien  # noqa: E402
from mani_skill.utils import sapien_utils  # noqa: E402
from mani_skill.utils.structs.pose import Pose  # noqa: E402

PHASE_NAME = {2: "lift", 3: "align_keyed", 4: "enter_key", 5: "unlock_yaw", 6: "circular_insert"}

# Frozen numbers from main.tex Table 1 (tab:protocol).
TOTAL_PHYSICS = 700
TOTAL_FITS = 1895

C_DARK = "#1b4f72"
C_MID = "#2471a3"
C_AMBER = "#b9770e"
C_TXT = "#1c1c1c"
C_SUB = "#555555"


def to_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return np.asarray(x, dtype=np.float64).reshape(-1)


def aim(env, eye, target):
    cam = env.unwrapped.scene.human_render_cameras["render_camera"]
    pose = sapien_utils.look_at(np.asarray(eye, np.float32), np.asarray(target, np.float32))
    cam.camera.set_local_pose(
        sapien.Pose(to_np(pose.p).astype(np.float32), to_np(pose.q).astype(np.float32))
    )


def render_at(env, data, i):
    base = env.unwrapped
    device = base.device
    base.agent.robot.set_qpos(
        torch.tensor(data["qpos"][i], dtype=torch.float32, device=device).reshape(1, -1)
    )
    base.peg.set_pose(
        Pose.create_from_pq(
            torch.tensor(data["peg_pose"][i][:3], dtype=torch.float32, device=device).reshape(1, 3),
            torch.tensor(data["peg_pose"][i][3:], dtype=torch.float32, device=device).reshape(1, 4),
        )
    )
    base.socket.set_pose(
        Pose.create_from_pq(
            torch.tensor(data["socket_pose"][i][:3], dtype=torch.float32, device=device).reshape(1, 3),
            torch.tensor(data["socket_pose"][i][3:], dtype=torch.float32, device=device).reshape(1, 4),
        )
    )
    fr = env.render()
    if hasattr(fr, "cpu"):
        fr = fr.cpu().numpy()
    return np.squeeze(np.asarray(fr))


def last_step_of(data, phase):
    hit = np.where(data["solver_phase"] == phase)[0]
    return int(hit[-1]) if len(hit) else None


def render_scene_frames(h5_path, episode_id, phases, task_anchor):
    """Return {phase: frame} for the given episode, camera aimed at task_anchor."""
    data, ea, ra = load_episode(Path(h5_path), episode_id)
    orientation = np.asarray(ra["orientation"], dtype=np.float64)
    env_id = ra.get("env_id", "KeyedCircularPhaseSwitch-v1")
    if isinstance(env_id, bytes):
        env_id = env_id.decode()
    env = make_env(orientation, task_anchor, env_id=env_id)
    env.reset(seed=int(ea.get("episode_seed", 20260818)),
              options={"causal_delta": data["causal_delta"].tolist()})
    eye = task_anchor + np.array([0.34, -0.40, 0.36])
    target = task_anchor + np.array([0.0, 0.0, 0.01])
    aim(env, eye, target)
    out = {}
    for p in phases:
        i = last_step_of(data, p)
        if i is not None:
            out[p] = render_at(env, data, i)
    env.close()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, default=4)
    parser.add_argument("--out-png", type=Path, default=Path("../experiment_overview_scenes.png"))
    parser.add_argument("--out-pdf", type=Path, default=Path("../experiment_overview_scenes.pdf"))
    args = parser.parse_args()

    rr = Path("../phase_switch_symmetry_rollouts_rotated")
    keyed_q1 = rr / "rotated_Q1_seed_20260818.h5"
    keyed_q2 = rr / "rotated_Q2_seed_20260818.h5"
    honest = rr / "circular_honest_seed_20260818.h5"
    placebo = rr / "circular_placebo_seed_20260818.h5"

    # task_anchor is identical across the rotated arms; read it once from Q1.
    _, _, ra = load_episode(keyed_q1, 0)
    task_anchor = np.asarray(ra["task_anchor"], dtype=np.float64)

    ep_q1 = first_success_condition(keyed_q1, args.cid)
    ep_q2 = first_success_condition(keyed_q2, args.cid)
    ep_hn = first_success_condition(honest, args.cid)
    ep_pl = first_success_condition(placebo, args.cid)
    print(f"episodes: keyedQ1={ep_q1} keyedQ2={ep_q2} honest={ep_hn} placebo={ep_pl}", flush=True)

    # Row 1: the keyed phase-switch mechanism (single-seed / five-seed scene).
    ph_frames = render_scene_frames(keyed_q1, ep_q1, [2, 3, 4, 5, 6], task_anchor)
    # Row 2: geometry variants. keyedQ1 uses a full-scene lift frame; Q2 uses insert.
    q2_frames = render_scene_frames(keyed_q2, ep_q2, [6], task_anchor)
    hn_frames = render_scene_frames(honest, ep_hn, [6], task_anchor)
    pl_frames = render_scene_frames(placebo, ep_pl, [6], task_anchor)

    frames = {
        "align": ph_frames.get(3),
        "enter": ph_frames.get(4),
        "unlock": ph_frames.get(5),
        "insert": ph_frames.get(6),
        "q1_lift": ph_frames.get(2),
        "q2": q2_frames.get(6),
        "honest": hn_frames.get(6),
        "placebo": pl_frames.get(6),
    }
    missing = [k for k, v in frames.items() if v is None]
    if missing:
        raise RuntimeError(f"missing frames: {missing}")

    # ---- Compose -----------------------------------------------------------
    fig = plt.figure(figsize=(13.2, 7.6))
    gs = fig.add_gridspec(2, 4, left=0.01, right=0.99, top=0.80, bottom=0.16,
                          wspace=0.05, hspace=0.16)

    def panel(ax, img, title, sub, edge):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(edge)
            s.set_linewidth(2.2)
        ax.set_title(title, fontsize=10.2, fontweight="bold", color=C_DARK, pad=5)
        ax.text(0.5, -0.12, sub, transform=ax.transAxes, ha="center", va="top",
                fontsize=7.6, color=C_SUB)

    # Row 1 — phase switch
    row1 = [
        ("align", "align_keyed", "key aligns to rotated gate"),
        ("enter", "enter_key", "key passes through keyed gate"),
        ("unlock", "unlock_yaw", "yaw unlocks in circular bore"),
        ("insert", "circular_insert", "shaft seats into bore"),
    ]
    for col, (key, title, sub) in enumerate(row1):
        panel(fig.add_subplot(gs[0, col]), frames[key], title, sub, C_MID)

    # Row 2 — geometry variants behind each experiment group
    row2 = [
        ("q1_lift", "Keyed Q₁ (vertical)",
         "single-seed 39  ·  five-seed 195\nfew-shot 1,815 fits  ·  TP-GMM 80 fits", C_MID),
        ("q2", "Keyed Q₂ (horizontal)",
         "rotated axis 232 usable", C_AMBER),
        ("honest", "Circular honest",
         "circular symmetry 117 usable", C_AMBER),
        ("placebo", "Circular placebo",
         "circular symmetry 117 usable", C_AMBER),
    ]
    for col, (key, title, sub, edge) in enumerate(row2):
        panel(fig.add_subplot(gs[1, col]), frames[key], title, sub, edge)

    # Row labels
    fig.text(0.004, 0.615, "Phase switch\n(keyed task)", rotation=90, ha="center",
             va="center", fontsize=11, fontweight="bold", color=C_MID)
    fig.text(0.004, 0.30, "Geometry variants\n(experiment arms)", rotation=90, ha="center",
             va="center", fontsize=10, fontweight="bold", color=C_AMBER)

    # Header
    fig.text(0.5, 0.945, "Keyed-to-circular peg insertion — real simulation scenes",
             ha="center", va="center", fontsize=16, fontweight="bold", color=C_DARK)
    fig.text(0.5, 0.885, "ManiSkill / PhysX  ·  Panda pd_joint_pos  ·  intervention c = [Δx, Δy, Δψ]\n"
                         "frozen 39-condition manifest (30 mixed + 8 isolated + 1 zero)  ·  held-out socket yaw +30° shown",
             ha="center", va="center", fontsize=9.0, color=C_TXT)

    # Footer
    fig.text(0.5, 0.075,
             f"6 experiments   ·   {TOTAL_PHYSICS} usable physics rollouts   ·   {TOTAL_FITS} offline model fits\n"
             "all rollouts executed in simulation (ManiSkill/PhysX); failed attempts retained with condition id, attempt id, seed, stop reason.",
             ha="center", va="center", fontsize=10.5, color=C_DARK, fontweight="bold")

    fig.savefig(args.out_png, dpi=150, facecolor="white", bbox_inches="tight")
    fig.savefig(args.out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {args.out_png.resolve()}  and  {args.out_pdf.resolve()}")


if __name__ == "__main__":
    main()
