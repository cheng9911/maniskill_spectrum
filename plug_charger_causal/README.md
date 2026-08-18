# PlugCharger-Causal dataset generation

This directory contains a local ManiSkill task wrapper and generator for
controlled interventions on `PlugCharger-v1`.

The wrapper registers:

```text
PlugChargerCausal-v1
```

It keeps the official ManiSkill task geometry and dynamics, then applies a
controlled intervention to the receptacle after reset:

```text
[delta_x_m, delta_y_m, delta_yaw_rad]
```

Only the receptacle pose and derived `goal_pose` are changed.

## Generate a smoke dataset

```bash
cd /home/rocos/sia
MPLCONFIGDIR=/home/rocos/sia/maniskill_spectrum/.matplotlib \
conda run -n maniskill_download \
python maniskill_spectrum/plug_charger_causal/generate_dataset.py \
  --split smoke \
  --num-seeds 1 \
  --only-count-success
```

## Generate isolated causal test grid

```bash
cd /home/rocos/sia
MPLCONFIGDIR=/home/rocos/sia/maniskill_spectrum/.matplotlib \
conda run -n maniskill_download \
python maniskill_spectrum/plug_charger_causal/generate_dataset.py \
  --split isolated_grid \
  --num-seeds 1 \
  --only-count-success
```

Outputs are written to:

```text
maniskill_spectrum/plug_charger_causal/datasets/<split>/
```

Each generated dataset includes:

- `trajectory.h5`
- `trajectory.json`
- `manifest.jsonl`

The HDF5/JSON pair follows ManiSkill `RecordEpisode` format. The manifest stores
the intervention labels for each generated episode.

The default generator disables rendering and records state/action trajectories
only. Add `--save-video --render-mode rgb_array --render-backend sapien_cuda` on a
machine with working SAPIEN rendering if videos are needed.

## Generate reset/state intervention data

If `mplib` motion planning is unavailable on the machine, generate controlled
ground-truth reset states without planner trajectories:

```bash
cd /home/rocos/sia
PYTHONNOUSERSITE=1 \
LD_LIBRARY_PATH=/home/rocos/miniconda3/envs/maniskill_download/lib \
MPLCONFIGDIR=/home/rocos/sia/maniskill_spectrum/.matplotlib \
conda run -n maniskill_download \
python maniskill_spectrum/plug_charger_causal/generate_reset_dataset.py \
  --split isolated_grid \
  --num-seeds 1
```

This writes:

```text
maniskill_spectrum/plug_charger_causal/reset_datasets/<split>/reset_states.h5
maniskill_spectrum/plug_charger_causal/reset_datasets/<split>/reset_states.json
```
