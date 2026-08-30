from __future__ import annotations

"""Render standalone, clean figures of the two peg geometries for slides.

Two objects, one figure each:
  * circular peg   — the shaft-only cylinder (orange).
  * keyed peg      — the same cylinder plus a rectangular key at its lower tip.

The robot arm, socket, and grasp pedestal are hidden / moved out of frame so each
figure shows *only* the peg geometry standing on the table, lit by the scene's
default lighting, from a fixed 3/4 camera. Output is written to standalone_pegs/.

Run from phase_switch_symmetry/ so `phase_switch_symmetry_env` is importable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sqlite3  # noqa: F401, E402 - load conda sqlite/libstdc++ before torch/sapien

import numpy as np  # noqa: E402
import sapien  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

import mani_skill.envs  # noqa: E402
import phase_switch_symmetry_env  # noqa: E402
from mani_skill.sensors.camera import CameraConfig  # noqa: E402
from mani_skill.utils import sapien_utils  # noqa: E402
from mani_skill.utils.structs.pose import Pose  # noqa: E402

OUT_DIR = Path("standalone_pegs")
RES = 1024


def render_peg(keyed: bool, out_path: Path, resolution: int = RES):
    env_id = "KeyedCircularPhaseSwitch-v1" if keyed else "CircularPhaseSwitch-v1"
    # Human-render camera override: a 3/4 view centred on the peg, high-res.
    # Passed as a constructor kwarg so it applies when the camera is created.
    pose = sapien_utils.look_at([0.07, -0.20, 0.11], [0.0, 0.0, 0.025])
    camera_overrides = {
        "width": resolution,
        "height": resolution,
        "fov": 0.9,
        "near": 0.01,
        "far": 100,
        "pose": pose,
    }
    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_joint_pos",
        reward_mode="sparse",
        sim_backend="physx_cpu",
        render_mode="rgb_array",
        human_render_camera_configs=camera_overrides,
    )
    base = env.unwrapped

    env.reset(seed=20260818, options={"causal_delta": [0.0, 0.0, 0.0]})

    # Move the clutter far away (kinematic actors can't be hidden); keep only the
    # peg on the table.
    far_p = torch.tensor([[5.0, 0.0, 0.0]], dtype=torch.float32, device=base.device)
    far_q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=base.device)
    far_pose = Pose.create_from_pq(far_p, far_q)
    base.socket.set_pose(far_pose)
    base.pedestal.set_pose(far_pose)
    base.agent.robot.set_pose(sapien.Pose([5.0, 0.0, 0.0]))  # out of frustum

    # Stand the peg upright, axis along world +z, centred over the table.
    device = base.device
    peg_p = torch.tensor([[0.0, 0.0, 0.046]], dtype=torch.float32, device=device)
    peg_q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    base.peg.set_pose(Pose.create_from_pq(peg_p, peg_q))

    frame = env.render()
    if hasattr(frame, "cpu"):
        frame = frame.cpu().numpy()
    else:
        frame = np.asarray(frame)
    frame = np.squeeze(frame)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_path, frame)
    env.close()
    return frame


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    circ = render_peg(keyed=False, out_path=OUT_DIR / "peg_circular.png")
    keyed = render_peg(keyed=True, out_path=OUT_DIR / "peg_keyed.png")
    print(f"wrote {OUT_DIR / 'peg_circular.png'}  {circ.shape}")
    print(f"wrote {OUT_DIR / 'peg_keyed.png'}    {keyed.shape}")

    # Convenience side-by-side (matplotlib, RGB PNG).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, img, title in [
        (axes[0], circ, "Circular peg"),
        (axes[1], keyed, "Keyed peg"),
    ]:
        ax.imshow(img)
        ax.set_title(title, fontsize=13)
        ax.axis("off")
    fig.tight_layout(pad=0.5)
    fig.savefig(OUT_DIR / "peg_compare.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'peg_compare.png'}")


if __name__ == "__main__":
    main()
