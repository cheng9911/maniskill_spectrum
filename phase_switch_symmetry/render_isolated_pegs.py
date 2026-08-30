from __future__ import annotations

"""Render isolated peg geometry from the ManiSkill phase-switch tasks.

This creates clean 3/4-view PNGs on a white background:
  - circular/keyed peg only
  - circular/keyed peg inserted into its socket

Run from the repository root, for example:
  conda run -n maniskill_download python phase_switch_symmetry/render_isolated_pegs.py
"""

import argparse
import sqlite3  # noqa: F401 - load conda sqlite/libstdc++ before torch/sapien
import sys
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

import mani_skill.envs  # noqa: F401
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.pose import Pose

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase_switch_symmetry_env as pse  # noqa: E402,F401


PEG_SPECS = {
    "circular": {
        "env_id": "CircularPhaseSwitch-v1",
        "filename": "isolated_circular_peg.png",
        "socket_filename": "circular_peg_socket.png",
        "center_z": -pse.SHAFT_Z_MIN,
        "socket_center_z": 0.069,
    },
    "keyed": {
        "env_id": "KeyedCircularPhaseSwitch-v1",
        "filename": "isolated_keyed_peg.png",
        "socket_filename": "keyed_peg_socket.png",
        "center_z": pse.PEG_HALF_LENGTH,
        "socket_center_z": 0.085,
    },
}

SOCKET_BASE_RADIUS = 0.085
SOCKET_BASE_HEIGHT = 0.040
SOCKET_HOLE_RADIUS = pse.SHAFT_RADIUS + 0.006
SOCKET_SLOT_HALF_SIZE = np.array(
    [pse.KEY_HALF_X + 0.003, pse.KEY_HALF_Y + 0.003, 0.001],
    dtype=np.float64,
)


def to_np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu()
    return np.asarray(x, dtype=np.float64).reshape(-1)


def make_env(env_id: str, width: int, height: int, shader_pack: str):
    return gym.make(
        env_id,
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
        reward_mode="sparse",
        sim_backend="physx_cpu",
        render_mode="rgb_array",
        human_render_camera_configs={
            "render_camera": {
                "width": width,
                "height": height,
                "shader_pack": shader_pack,
            }
        },
    )


def aim_camera(env, eye, target):
    cam = env.unwrapped.scene.human_render_cameras["render_camera"]
    pose = sapien_utils.look_at(
        np.asarray(eye, dtype=np.float32), np.asarray(target, dtype=np.float32)
    )
    cam.camera.set_local_pose(
        sapien.Pose(to_np(pose.p).astype(np.float32), to_np(pose.q).astype(np.float32))
    )


def set_entity_visibility(entity, visibility: float):
    body = entity.find_component_by_type(sapien.render.RenderBodyComponent)
    if body is not None:
        body.visibility = visibility


def set_actor_visibility(actor, visibility: float):
    for obj in actor._objs:
        set_entity_visibility(obj, visibility)


def set_articulation_visibility(articulation, visibility: float):
    for link in articulation.links:
        for obj in link._objs:
            set_entity_visibility(obj.entity, visibility)


def material(color):
    return sapien.render.RenderMaterial(
        base_color=color, roughness=0.68, specular=0.25
    )


def build_clean_socket(base, kind: str):
    builder = base.scene.create_actor_builder()
    cylinder_to_z = sapien.Pose(q=euler2quat(0.0, -np.pi / 2, 0.0))
    builder.add_cylinder_visual(
        pose=sapien.Pose(
            [0.0, 0.0, 0.5 * SOCKET_BASE_HEIGHT], q=cylinder_to_z.q
        ),
        radius=SOCKET_BASE_RADIUS,
        half_length=0.5 * SOCKET_BASE_HEIGHT,
        material=material([0.62, 0.64, 0.63, 1.0]),
    )
    hole_z = SOCKET_BASE_HEIGHT + 0.001
    if kind == "circular":
        builder.add_cylinder_visual(
            pose=sapien.Pose([0.0, 0.0, hole_z], q=cylinder_to_z.q),
            radius=SOCKET_HOLE_RADIUS,
            half_length=0.001,
            material=material([0.18, 0.20, 0.20, 1.0]),
        )
    else:
        builder.add_box_visual(
            pose=sapien.Pose([0.0, 0.0, hole_z]),
            half_size=SOCKET_SLOT_HALF_SIZE.tolist(),
            material=material([0.18, 0.20, 0.20, 1.0]),
        )
    builder.initial_pose = sapien.Pose()
    return builder.build_kinematic(f"clean_{kind}_socket")


def hide_everything_except(base, keep: set[str]):
    peg_name = "circular_peg" if not base.keyed else "keyed_circular_peg"
    keep_names = set(keep)
    if "peg" in keep_names:
        keep_names.add(peg_name)
    if "socket" in keep_names:
        keep_names.add("keyed_to_circular_socket")
    for name, actor in base.scene.actors.items():
        set_actor_visibility(actor, 1.0 if name in keep_names else 0.0)
    set_articulation_visibility(base.agent.robot, 0.0)


def replace_black_background(frame: np.ndarray) -> np.ndarray:
    out = frame.copy()
    background = np.all(out <= 8, axis=-1)
    out[background] = 255
    return out


def render_one(
    kind: str,
    scene: str,
    out_dir: Path,
    width: int,
    height: int,
    eye: np.ndarray,
    target: np.ndarray,
    shader_pack: str,
    socket_style: str,
):
    spec = PEG_SPECS[kind]
    env = make_env(spec["env_id"], width, height, shader_pack)
    try:
        env.reset(seed=0, options={"causal_delta": [0.0, 0.0, 0.0]})
        base = env.unwrapped
        device = base.device

        show_socket = scene == "socket"
        show_real_socket = show_socket and socket_style == "real"
        hide_everything_except(base, {"peg", "socket"} if show_real_socket else {"peg"})

        if show_socket:
            if show_real_socket:
                socket_p = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
                socket_q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
                base.socket.set_pose(Pose.create_from_pq(socket_p, socket_q))
            else:
                build_clean_socket(base, kind)

        peg_p = torch.tensor(
            [[0.0, 0.0, spec["socket_center_z" if show_socket else "center_z"]]],
            dtype=torch.float32,
            device=device,
        )
        peg_q = torch.tensor(
            np.asarray(euler2quat(0.0, 0.0, np.deg2rad(25.0))).reshape(1, 4),
            dtype=torch.float32,
            device=device,
        )
        base.peg.set_pose(Pose.create_from_pq(peg_p, peg_q))
        aim_camera(env, eye, target)

        frame = env.render()
        if hasattr(frame, "cpu"):
            frame = frame.cpu().numpy()
        frame = np.squeeze(np.asarray(frame))
        frame = replace_black_background(frame)

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (spec["socket_filename"] if show_socket else spec["filename"])
        imageio.imwrite(out_path, frame)
        return out_path, frame.shape
    finally:
        env.close()


def parse_vec3(text: str) -> np.ndarray:
    parts = [float(x) for x in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values")
    return np.asarray(parts, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("phase_switch_symmetry_rollouts_rotated/_isolated_pegs"),
    )
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    parser.add_argument("--eye", type=parse_vec3, default=parse_vec3("0.11,-0.18,0.13"))
    parser.add_argument("--target", type=parse_vec3, default=parse_vec3("0,0,0.04"))
    parser.add_argument("--shader-pack", default="minimal")
    parser.add_argument(
        "--socket-style",
        choices=["clean", "real"],
        default="clean",
        help="clean=display-friendly smooth base, real=task socket geometry",
    )
    parser.add_argument(
        "--kind",
        choices=["all", "circular", "keyed"],
        default="all",
        help="which peg geometry to render",
    )
    parser.add_argument(
        "--scene",
        choices=["all", "peg", "socket"],
        default="all",
        help="peg=peg only, socket=peg inserted into socket",
    )
    args = parser.parse_args()

    kinds = ["circular", "keyed"] if args.kind == "all" else [args.kind]
    scenes = ["peg", "socket"] if args.scene == "all" else [args.scene]
    for kind in kinds:
        for scene in scenes:
            out_path, shape = render_one(
                kind=kind,
                scene=scene,
                out_dir=args.out_dir,
                width=args.width,
                height=args.height,
                eye=args.eye,
                target=args.target,
                shader_pack=args.shader_pack,
                socket_style=args.socket_style,
            )
            print(f"{kind} {scene}: {out_path} {shape}")


if __name__ == "__main__":
    main()
