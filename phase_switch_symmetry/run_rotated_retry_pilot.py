from __future__ import annotations

"""Retry pilot: do the 13 first-round-failed cells recover under R_max=5?

Fixes orientation, anchor, solver, noise=0.01, outer seed, and frozen contexts;
only the stochastic realization (attempt seed) varies. Each failed
(Q, seed, condition) cell is retried 5 additional independent attempts, and the
survival curve N_unresolved(r), R_fail(r), R_all(r) is reported.
"""

import argparse
import json
import sqlite3  # noqa: F401 - load conda sqlite/libstdc++ before torch/sapien
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import pandas as pd

import mani_skill.envs  # noqa: F401
import phase_switch_symmetry_env  # noqa: F401
from collect_phase_switch_rollouts import (
    EpisodeFinished,
    PHASES,
    PhaseSwitchTraceWrapper,
)
from collect_phase_switch_rotated import solve_rotated
from mani_skill.agents.robots.panda import Panda

Panda.gripper_stiffness = 2.5e3
Panda.gripper_force_limit = 150.0

PHASE_NAME = {code: name for name, code in PHASES.items()}
QUAT = {"Q1": [1.0, 0.0, 0.0, 0.0], "Q2": [0.707107, 0.0, 0.707107, 0.0]}
ANCHOR = np.array([-0.35, 0.10, 0.08])


def identify_failed(files, manifest):
    conds = {c["condition_id"]: c for c in manifest["conditions"]}
    failed = []
    for path in files:
        q = "Q1" if "_Q1_" in path.stem else "Q2"
        seed = int(path.stem.split("seed_")[-1])
        usable = set()
        with h5py.File(path, "r") as f:
            for k in f:
                g = f[k]
                if bool(np.asarray(g["success"])[-1]):
                    usable.add(int(g.attrs.get("condition_id", -1)))
        for cid in range(39):
            if cid not in usable:
                failed.append((q, seed, cid, conds[cid]["generator"],
                               np.asarray(conds[cid]["causal_delta"])))
    return failed


def run_retry(q, seed, cid, causal_delta, attempts):
    env = gym.make(
        "KeyedCircularPhaseSwitch-v1",
        num_envs=1, obs_mode="state_dict", control_mode="pd_joint_pos",
        reward_mode="sparse", sim_backend="physx_cpu", render_mode=None,
        robot_init_qpos_noise=0.01, orientation=QUAT[q], task_anchor=ANCHOR,
    )
    w = PhaseSwitchTraceWrapper(env)
    results = []
    for a in range(attempts):
        episode_seed = seed + cid * 1009 + (100 + a)  # fresh range distinct from first round
        np.random.seed(episode_seed)
        import torch
        torch.manual_seed(episode_seed)
        w.reset(seed=episode_seed, options={"causal_delta": causal_delta})
        w.start_trace()
        reachable = False
        phase = -1
        try:
            solve_rotated(w)
            reachable = True
        except EpisodeFinished as exc:
            reachable = bool(exc.terminated)
        except Exception:
            reachable = False
        phase = int(w.trace["states"][-1]["solver_phase"])
        states = w.trace["states"]
        force = np.linalg.norm(np.asarray([s["contact_force"] for s in states]), axis=1)
        final_dist = float(states[-1]["obj_to_goal_dist"])
        results.append({
            "attempt": a, "reachable": reachable,
            "last_phase": PHASE_NAME.get(phase, "?"),
            "max_contact": float(force.max()), "final_dist": final_dist,
            "steps": len(w.trace["actions"]),
        })
    env.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("phase_switch_symmetry_multiseed/fixed_contexts.json"),
    )
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument(
        "--out", type=Path,
        default=Path("phase_switch_symmetry_rollouts_rotated/retry_pilot.csv"),
    )
    args = parser.parse_args()

    with args.manifest.open() as f:
        manifest = json.load(f)
    failed = identify_failed(args.files, manifest)
    print(f"identified {len(failed)} failed cells")

    rows = []
    unresolved_by_r = {r: 0 for r in range(args.attempts + 1)}
    for q, seed, cid, gen, causal_delta in failed:
        res = run_retry(q, seed, cid, causal_delta, args.attempts)
        recovered = False
        for a, r in enumerate(res, start=1):
            if r["reachable"]:
                recovered = True
            else:
                unresolved_by_r[a] = unresolved_by_r.get(a, 0) + 1
            rows.append({
                "orientation": q, "seed": seed, "condition_id": cid,
                "generator": gen, "attempt": a, **r,
            })
        if not recovered:
            unresolved_by_r[args.attempts] = unresolved_by_r.get(args.attempts, 0) + 1
        print(f"  {q} seed {seed} cond {cid} ({gen}): "
              f"recovered={recovered} attempts={[int(r['reachable']) for r in res]}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    # Survival curve over the 180 mixed cells (167 first-round successes).
    n_failed = len(failed)
    n_all = 180
    per_cell = df.groupby(["orientation", "seed", "condition_id"]).reachable.max()
    unresolved = {0: n_failed}
    for r in range(1, args.attempts + 1):
        sub = df[df.attempt <= r].groupby(
            ["orientation", "seed", "condition_id"]).reachable.max()
        still = int((sub == 0).sum()) if len(sub) else n_failed
        unresolved[r] = still

    print("\n=== survival curve (unresolved cells) ===")
    for r in range(args.attempts + 1):
        n_ok = n_all - unresolved.get(r, n_failed)
        print(f"  r={r}: unresolved={unresolved.get(r, n_failed)}, "
              f"R_all={n_ok}/{n_all} = {n_ok/n_all:.3f}, "
              f"R_fail={1 - unresolved.get(r, n_failed)/n_failed:.3f}")

    still5 = [k for k, v in per_cell.items() if v == 0]
    print(f"\ncells still failed after {args.attempts} attempts: {len(still5)}")
    for q, seed, cid in still5:
        print(f"  {q} seed {seed} cond {cid}")


if __name__ == "__main__":
    main()
