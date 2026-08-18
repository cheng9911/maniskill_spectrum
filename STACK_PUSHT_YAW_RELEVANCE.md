# StackCube vs PushT Yaw Relevance

This is an observational task-constraint check on downloaded ManiSkill demonstrations. It is not a controlled intervention test.

```json
{
  "interpretation": {
    "pusht": "Successful PushT tightly constrains Tee yaw to goal_Tee yaw.",
    "stackcube": "Successful stacking tightly constrains cubeA/cubeB xy, but does not require cubeA yaw to match cubeB yaw."
  },
  "pusht": {
    "abs_yaw_error_mean_rad": 0.01944768415030425,
    "abs_yaw_error_p50_deg": 0.9013667411886789,
    "abs_yaw_error_p95_deg": 2.7976156716946523,
    "dataset": "maniskill_spectrum/demos/PushT-v1/rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5",
    "episodes": 719,
    "tee_goal_yaw_circular_corr": NaN,
    "xy_error_mean_m": 0.004115713118018715,
    "xy_error_p95_m": 0.007091913829739278,
    "yaw_match_within_10deg_frac": 1.0
  },
  "stackcube": {
    "abs_yaw_rel_mean_rad": 1.5294066486038747,
    "abs_yaw_rel_p50_deg": 87.67668588118804,
    "abs_yaw_rel_p95_deg": 168.74592273032266,
    "cubeA_cubeB_yaw_circular_corr": 0.01987242718277394,
    "dataset": "maniskill_spectrum/demos/StackCube-v1/motionplanning/trajectory.h5",
    "episodes": 1000,
    "xy_error_mean_m": 0.004773061784136837,
    "xy_error_p95_m": 0.010361864570002346,
    "yaw_match_within_10deg_frac": 0.058
  }
}
```
