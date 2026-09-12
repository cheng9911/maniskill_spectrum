from __future__ import annotations

"""Segment 5 (robot-arm variant) -- replay real LIBERO robot demos.

Instead of the state-level counterfactual (object moved directly, arm static),
this re-renders each scene from an actual LIBERO teleoperated demonstration, so
the robot arm visibly executes the task (reach -> grasp -> move -> release).

It replays the demo's saved MuJoCo ``states`` through the scene's real
``OffScreenRenderEnv`` (set_state + forward + render at 512x512), so we get the
honest robot-arm trajectory without needing a learned policy or OSC controller.
Downloading each demo hdf5 on first use (from HuggingFace, ~700 MB each).
"""

import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "phase_switch_symmetry"))

from collect_libero_relation_suite import make_libero_env  # noqa: E402
from libero_relation_suite_specs import specs_by_key  # noqa: E402

from video_common import FPS, write_mp4  # noqa: E402

DEMO_DIR = Path("/home/rocos/.cache/libero/datasets")
REPO = "yifengzhu-hf/LIBERO-datasets"
CAM = 512
N_FRAMES = 48

# (task_key, demo file path inside the HF repo)
TASKS = [
    ("drawer_middle_open", "libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5"),
    ("plate_front_push", "libero_goal/push_the_plate_to_the_front_of_the_stove_demo.hdf5"),
    ("stove_knob_turn", "libero_goal/turn_on_the_stove_demo.hdf5"),
    ("microwave_door_revolute", "libero_10/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo.hdf5"),
    ("bowl_on_stove", "libero_goal/put_the_bowl_on_the_stove_demo.hdf5"),
    ("cream_cheese_in_bowl", "libero_goal/put_the_cream_cheese_in_the_bowl_demo.hdf5"),
    ("bowl_on_plate", "libero_goal/put_the_bowl_on_the_plate_demo.hdf5"),
    ("wine_bottle_on_rack", "libero_goal/put_the_wine_bottle_on_the_rack_demo.hdf5"),
    ("wine_bottle_on_cabinet", "libero_goal/put_the_wine_bottle_on_top_of_the_cabinet_demo.hdf5"),
    ("moka_pot_on_stove", "libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5"),
]


def ensure_demo(rel: str) -> Path:
    local = DEMO_DIR / rel
    if local.exists():
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(repo_id=REPO, filename=rel, repo_type="dataset", local_dir=str(DEMO_DIR))
    return Path(p)


def render_task(task_key: str, rel: str, out_dir: Path) -> Path:
    out = out_dir / f"seg5_robot_{task_key}.mp4"
    if out.exists():
        print(f"skip {out.name}  (already rendered)")
        return out
    demo_path = ensure_demo(rel)
    with h5py.File(demo_path, "r") as f:
        states = np.asarray(f["data"]["demo_0"]["states"], dtype=np.float64)
    n = states.shape[0]
    idx = np.linspace(0, n - 1, N_FRAMES).round().astype(int)

    spec = specs_by_key()[task_key]
    env, _bddl, _name = make_libero_env(spec, CAM)
    env.seed(20260818)
    env.reset()
    sim = env.sim

    frames = []
    for i in idx:
        env.set_state(states[i])
        sim.forward()
        img = np.asarray(sim.render(camera_name="agentview", width=CAM, height=CAM))
        frames.append(img[::-1].copy())
    env.close()

    write_mp4(out, frames, fps=FPS)
    print(f"wrote {out}  frames={len(frames)}  (demo states={n})")
    return out


def main() -> None:
    out_dir = HERE / "segments"
    out_dir.mkdir(parents=True, exist_ok=True)
    for task_key, rel in TASKS:
        render_task(task_key, rel, out_dir)


if __name__ == "__main__":
    main()
