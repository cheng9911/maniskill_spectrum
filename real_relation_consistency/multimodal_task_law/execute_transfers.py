from __future__ import annotations

"""Execute generated transfer trajectories in PhysX (plan section 7.2).

For a target mode B, held-out condition c, block b and method, the target peg
trajectory is generated from B's 0-degree reference and the frozen response law:

    Xhat_B,c(t) = Rot_y( alpha(progress(t)) * theta_c , socket centre ) applied to X_B,0(t)

  no_adapt : alpha = 0      rigid : alpha = 1      ramp : alpha = progress
  source   : alpha = alpha_A (frozen source-mode law)
  target   : alpha = alpha_B (target-mode own law, upper-bound reference)

The generated PEG waypoints are lifted to TCP via the execution's OWN grasp
transform H (fixed after grasp, NOT the target demonstration's H), IK + joint
continuity + pd_joint_pos tracking is done by the SAME planner primitive as the
solver (``move_to_pose``), and the episode launches from the common post-grasp
state (shared reach/grasp/lift).  No solve_se3 / RRT re-planning of the assembly
from the hole target is used.
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "phase_switch_symmetry"))
sys.path.insert(0, str(HERE))

import mani_skill.envs  # noqa: F401, E402
import phase_switch_symmetry_env  # noqa: F401, E402
from collect_phase_switch_rollouts import EpisodeFinished  # noqa: E402
from generate_contexts import load_protocol  # noqa: E402
from mani_skill.agents.robots.panda import Panda  # noqa: E402
from phase_switch_se3_baselines import euler_from_matrix, matrix_from_euler_batched  # noqa: E402
from solver_modes import ModeTraceWrapper, _make_planner  # noqa: E402
from transforms3d.quaternions import quat2mat  # noqa: E402

import sapien  # noqa: E402
from transforms3d.quaternions import qmult  # noqa: E402

Panda.gripper_stiffness = 2.5e3
Panda.gripper_force_limit = 150.0


def rotate_pose7_about_y(pose7: np.ndarray, ang: float, center: np.ndarray) -> np.ndarray:
    """Rotate a 7-vec peg pose about the world y-axis through ``center`` by ``ang``."""
    R = matrix_from_euler_batched(np.array([[0.0, ang, 0.0]]))[0]
    pos = R @ (pose7[:3] - center) + center
    R_peg = quat2mat(pose7[3:])
    R_out = R @ R_peg
    # back to quaternion (wxyz)
    from transforms3d.quaternions import mat2quat
    q = mat2quat(R_out)
    return np.concatenate([pos, np.array([q[0], q[1], q[2], q[3]])])


def progress_of_steps(events: dict, T: int) -> np.ndarray:
    """Map raw step index -> (segment, progress in [0,1]) for a 0-degree episode."""
    E0, E1, E2, E3 = events["E0"], events["E1"], events["E2"], events["E3"]
    seg = np.full(T, -1.0)
    prog = np.zeros(T)
    bounds = [E0, E1, E2, E3]
    for k in range(3):
        a, b = bounds[k], bounds[k + 1]
        if a < 0 or b < 0 or b <= a:
            continue
        mask = (np.arange(T) >= a) & (np.arange(T) <= b)
        seg[mask] = k
        prog[mask] = np.clip((np.arange(T)[mask] - a) / (b - a), 0.0, 1.0)
    return seg, prog


def alpha_at(alpha_grid: np.ndarray, seg: np.ndarray, prog: np.ndarray) -> np.ndarray:
    """Interpolate the 150-point alpha_grid (3x50) onto raw steps."""
    out = np.zeros(len(seg))
    for k in range(3):
        m = seg == k
        if not m.any():
            continue
        idx = k * 50 + (prog[m] * 49).astype(int)
        out[m] = alpha_grid[np.clip(idx, 0, 149)]
    return out


def build_env(protocol):
    p = protocol
    base = gym.make(
        p["task"]["env_id"], num_envs=1, obs_mode="state_dict",
        control_mode=p["task"]["control_mode"], reward_mode="sparse",
        sim_backend=p["task"]["sim_backend"], render_mode=None,
        robot_init_qpos_noise=p["task"]["robot_init_qpos_noise"],
        orientation=np.asarray(p["task"]["orientation"]),
        task_anchor=p["task"]["task_anchor"],
    )
    return ModeTraceWrapper(base)


def execute_one(env, protocol, target_mode, theta_deg, alpha_grid, B0_peg, B0_events,
                initial_state_seed, planner_seed):
    p = protocol
    center = np.asarray(p["task"]["task_anchor"], dtype=np.float64)
    theta = np.deg2rad(theta_deg)
    pickup = np.asarray(p["task"]["pickup_position"])
    T = len(B0_peg)
    seg, prog = progress_of_steps(B0_events, T)
    alpha = alpha_at(alpha_grid, seg, prog)

    # generate target peg waypoints (downsample)
    K = 40
    step_idx = np.linspace(B0_events["E0"] + 2, T - 2, K).astype(int)
    waypoints = []
    for i in step_idx:
        wp = rotate_pose7_about_y(B0_peg[i], float(alpha[i]) * theta, center)
        waypoints.append(wp)

    base = env.unwrapped
    axis = base.insertion_axis
    planner = _make_planner(env)
    np.random.seed(initial_state_seed)
    torch.manual_seed(planner_seed)
    causal_delta = [0.0, 0.0, 0.0, 0.0, theta, 0.0]
    env.reset(seed=planner_seed, options={
        "causal_delta": np.asarray(causal_delta, dtype=np.float64),
        "pickup_position": pickup,
    })
    env.start_trace()
    solver_error = None
    try:
        peg_position = base.peg.pose.sp.p
        grasp_pose = base.agent.build_grasp_pose(
            approaching=-axis,
            closing=base.Q_mat @ np.array([0.0, 1.0, 0.0]),
            center=peg_position + 0.020 * axis,
        )

        def move_pose(target_pose, refine_steps=0):
            result = planner.move_to_pose_with_screw(target_pose, refine_steps=refine_steps)
            if result == -1:
                result = planner.move_to_pose_with_RRTConnect(target_pose, refine_steps=refine_steps)
            if result == -1:
                raise RuntimeError("motion planning failed")

        # shared reach/grasp/lift (identical to solver)
        move_pose(grasp_pose * sapien.Pose([0.0, 0.0, -0.070]))
        move_pose(grasp_pose)
        planner.close_gripper()
        move_pose(grasp_pose * sapien.Pose([0.0, 0.0, -0.095]))
        # fixed grasp transform H (execution's own, not the demonstration's)
        H = base.peg.pose.sp.inv() * base.agent.tcp.pose.sp

        for wi, wp in enumerate(waypoints):
            peg_pose = sapien.Pose(p=np.asarray(wp[:3]), q=np.asarray(wp[3:]))
            refine = 8 if wi >= len(waypoints) - 3 else 2
            move_pose(peg_pose * H, refine_steps=refine)
        # shared terminal insertion control (identical across methods): descend to
        # the tilted goal pose, exactly as the demonstration solver does.
        goal_sp = base.goal_pose.sp
        move_pose(goal_sp * H, refine_steps=10)
        move_pose(goal_sp * H, refine_steps=10)
    except EpisodeFinished:
        solver_error = "EpisodeFinished"
    except Exception as exc:  # noqa: BLE001
        solver_error = repr(exc)
    finally:
        planner.close()

    states = env.trace["states"]
    success = bool(states[-1]["success"]) if states else False
    sock_q = np.asarray(states[0]["socket_pose"])[3:] if states else np.array([1, 0, 0, 0])
    sock_axis = quat2mat(sock_q) @ np.array([0.0, 0.0, 1.0])
    peg = np.asarray(states[-1]["peg_pose"])[:3] if states else np.zeros(3)
    sock = np.asarray(states[-1]["socket_pose"])[:3] if states else np.zeros(3)
    depth = float(np.dot(peg - sock, sock_axis))
    lateral = float(np.linalg.norm((peg - sock) - depth * sock_axis))
    axis_err = float(np.asarray(states[-1]["axis_angle_err"])) if states else np.nan
    env.close()
    return dict(success=success, depth=depth, lateral_mm=lateral * 1000,
                axis_err_deg=np.rad2deg(axis_err), steps=len(states), solver_error=solver_error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--resampled", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--id", type=Path, required=True)
    parser.add_argument("--source-h5", type=Path, required=True)
    parser.add_argument("--source-mode", default="lift_early")
    parser.add_argument("--methods", default="no_adapt,rigid,source,target")
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="max executions (0 = all)")
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    p = protocol
    id_data = json.load(open(args.id))
    models = {k: np.array(v["alpha"]) for k, v in id_data["models"].items()}
    events = json.load(open(args.events))["episodes"]
    by_ep = {e["episode_id"]: e for e in events}
    methods = args.methods.split(",")

    # index: (mode, block, pitch) -> episode_id; and raw peg via source h5
    index = {(e["mode_id"], e["block_id"], e["pitch_deg"]): e for e in events}
    target_modes = [m for m in p["mode_ids"] if m != args.source_mode]
    seed_base = p["seed"]["base"]
    is_stride = p["seed"]["initial_state_stride"]
    pl_stride = p["seed"]["planner_stride"]

    results = []
    with h5py.File(args.source_h5, "r") as sf, h5py.File(args.resampled, "r") as rf:
        for target in target_modes:
            for blk in range(p["core_blocks"]):
                ref = index.get((target, blk, 0.0))
                if ref is None:
                    continue
                B0_peg = sf[ref["episode_id"]]["peg_pose"][:]
                B0_ev = dict(E0=ref["E0"], E1=ref["E1"], E2=ref["E2"], E3=ref["E3"])
                for theta_deg in p["test_pitch_deg"]:
                    for method in methods:
                        if len(results) >= args.limit and args.limit > 0:
                            break
                        if method in ("source",):
                            alpha = models.get(f"{args.source_mode}_{blk}")
                        elif method == "target":
                            alpha = models.get(f"{target}_{blk}")
                        elif method == "rigid":
                            alpha = np.ones(150)
                        elif method == "no_adapt":
                            alpha = np.zeros(150)
                        elif method == "ramp":
                            alpha = np.linspace(0.0, 1.0, 150)
                        else:
                            alpha = None
                        if alpha is None:
                            continue
                        env = build_env(protocol)
                        is_seed = seed_base + blk * is_stride
                        planner_seed = is_seed + pl_stride * (p["mode_ids"].index(target) * len(p["pitch_deg"]) + p["pitch_deg"].index(theta_deg))
                        r = execute_one(env, protocol, target, theta_deg, alpha, B0_peg, B0_ev,
                                        is_seed, planner_seed)
                        r.update(source=args.source_mode, target=target, block=blk,
                                 theta_deg=theta_deg, method=method)
                        results.append(r)
                        print(f"[{args.source_mode}->{target}] blk={blk} theta={theta_deg:+g} "
                              f"{method}: success={r['success']} depth={r['depth']:.3f} "
                              f"axis={r['axis_err_deg']:.2f}deg", flush=True)
                    if args.limit > 0 and len(results) >= args.limit:
                        break
                if args.limit > 0 and len(results) >= args.limit:
                    break
            if args.limit > 0 and len(results) >= args.limit:
                break

    with open(str(args.out_prefix) + ".json", "w", encoding="utf-8") as mf:
        json.dump(dict(protocol_sha256=p["content_sha256"], results=results), mf, indent=2)
    n_succ = sum(r["success"] for r in results)
    print(f"executed {len(results)} transfers, {n_succ}/{len(results)} success")


if __name__ == "__main__":
    main()
