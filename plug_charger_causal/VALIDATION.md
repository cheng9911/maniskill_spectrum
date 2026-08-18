# PlugCharger-Causal Validation

## Reset/State Datasets

```json
{
  "isolated_grid": {
    "charger_pose_max_abs_delta": 0.0,
    "dx_mm": [
      -20.0,
      -10.0,
      0.0,
      10.0,
      20.0
    ],
    "dy_mm": [
      -20.0,
      -10.0,
      10.0,
      20.0
    ],
    "dyaw_deg": [
      -14.999999999999998,
      -7.499999999999999,
      7.499999999999999,
      14.999999999999998
    ],
    "goal_receptacle_position_max_error_m": 0.0,
    "goal_translation_max_error_m": 3.2782554587607038e-09,
    "receptacle_translation_max_error_m": 3.2782554587607038e-09,
    "receptacle_yaw_max_error_rad": 1.8682118518853486e-08,
    "samples": 13,
    "tcp_pose_max_abs_delta": 0.0
  },
  "motionplanning_smoke": {
    "error": "Unable to synchronously open file (truncated file: eof = 96, sblock->base_addr = 0, stored_eof = 2048)",
    "exists": true,
    "valid_h5": false
  },
  "train_mixed": {
    "dx_range_mm": [
      -9.669447289429419,
      9.6167067755246
    ],
    "dy_range_mm": [
      -9.328288493890714,
      9.623900801326887
    ],
    "dyaw_range_deg": [
      -7.458922497447778,
      7.458149036838165
    ],
    "obs_extra_keys": [
      "charger_pose",
      "goal_pose",
      "receptacle_pose",
      "tcp_pose"
    ],
    "samples": 50
  }
}
```

## Interpretation

- The isolated reset dataset validates the intervention mechanism: charger and TCP initial poses stay fixed while receptacle/goal receive the requested dx, dy, and yaw perturbations.
- The mixed reset dataset validates the requested training perturbation range and contains state observations with tcp, charger, receptacle, and goal poses.
- This is not yet a trajectory-level validation of P(s). It validates the controlled do(delta c) data-generation layer only.
- The motion-planning smoke HDF5 is invalid/truncated because mplib segfaulted during planner initialization on this machine.
