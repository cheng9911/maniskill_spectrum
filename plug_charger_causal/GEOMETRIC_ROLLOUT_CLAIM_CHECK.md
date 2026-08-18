# Geometric Rollout Claim Check

Independent reproduction of the key claims on `geometric_rollouts`.

```json
{
  "isolated_test_nrmse": {
    "constant": 0.6362531510623063,
    "dense": 0.09363802309831351,
    "generator": 0.1166905561396736,
    "scalar": 0.11636608331725437
  },
  "isolated_test_nrmse_by_generator": {
    "dx": {
      "constant": 0.6281058699546911,
      "dense": 0.02186502424831824,
      "generator": 0.02893334501088139,
      "scalar": 0.027664481544845874
    },
    "dy": {
      "constant": 0.6292005514647877,
      "dense": 0.05278012718711988,
      "generator": 0.05461224121874361,
      "scalar": 0.05526284651649893
    },
    "dyaw": {
      "constant": 0.6384796519864445,
      "dense": 0.1042833599975665,
      "generator": 0.13066413325773882,
      "scalar": 0.13028897502152872
    }
  },
  "operator_energy_explained": {
    "constant": 0.6012879385829304,
    "dense": 1.0,
    "generator": 0.995871574590327,
    "scalar": 0.995855337471743
  },
  "profile_correlations": {
    "dx_dy": 0.9999999999999998,
    "dx_dyaw": 0.9994568865373501,
    "dy_dyaw": 0.9994568865373507
  }
}
```
