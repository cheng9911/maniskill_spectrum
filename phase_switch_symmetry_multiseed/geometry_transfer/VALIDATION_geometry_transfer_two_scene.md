# Two-Scene Few-Shot Geometry Transfer Check

Date: 2026-08-31

This table extracts two structurally different target scenes from the geometry
transfer supplement:

- `knob_to_microwave_door`: revolute yaw-only relation;
- `bowl_stove_to_cream_cheese_bowl`: support/container placement relation.

The transferred method freezes the source Pdiag N=30 alpha profile and uses only
one target nominal trajectory for geometry adaptation.  Target interventions are
validation probes; they are not used to fit alpha for the transferred method.
Target-scratch baselines fit on N target interventions and are evaluated on
held-out target contexts.

| pair | family | method | N | M_transfer | E_alpha | heldout MSE | count |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `knob_to_microwave_door` | revolute_yaw | Ours transfer | 3 | 1.000 | 1.380e-05 | 1.024e-09 | 3 |
| `knob_to_microwave_door` | revolute_yaw | Pdiag finite target scratch | 3 | 1.000 | 4.012e-05 | 1.402e-08 | 3 |
| `knob_to_microwave_door` | revolute_yaw | Frame-weighted target scratch | 3 | 0.000 | 9.068e-02 | 7.413e-06 | 3 |
| `knob_to_microwave_door` | revolute_yaw | Phase scalar GP target scratch | 3 | 0.000 | 9.068e-02 | 7.413e-06 | 3 |
| `knob_to_microwave_door` | revolute_yaw | Ours transfer | 5 | 1.000 | 1.380e-05 | 1.012e-09 | 3 |
| `knob_to_microwave_door` | revolute_yaw | Pdiag finite target scratch | 5 | 1.000 | 1.455e-05 | 1.219e-08 | 3 |
| `knob_to_microwave_door` | revolute_yaw | Frame-weighted target scratch | 5 | 0.000 | 1.010e-01 | 5.823e-06 | 3 |
| `knob_to_microwave_door` | revolute_yaw | Phase scalar GP target scratch | 5 | 0.000 | 1.010e-01 | 5.823e-06 | 3 |
| `knob_to_microwave_door` | revolute_yaw | Ours transfer | 8 | 1.000 | 1.380e-05 | 1.006e-09 | 3 |
| `knob_to_microwave_door` | revolute_yaw | Pdiag finite target scratch | 8 | 1.000 | 4.601e-05 | 1.434e-08 | 3 |
| `knob_to_microwave_door` | revolute_yaw | TP-GMM SE(3) target scratch | 8 | 1.000 | 3.294e-02 | 8.929e-06 | 3 |
| `knob_to_microwave_door` | revolute_yaw | Frame-weighted target scratch | 8 | 0.000 | 9.915e-02 | 5.343e-06 | 3 |
| `knob_to_microwave_door` | revolute_yaw | Phase scalar GP target scratch | 8 | 0.000 | 9.915e-02 | 5.343e-06 | 3 |
| `bowl_stove_to_cream_cheese_bowl` | support_to_container | Ours transfer | 3 | 1.000 | 3.451e-04 | 1.501e-08 | 3 |
| `bowl_stove_to_cream_cheese_bowl` | support_to_container | Pdiag finite target scratch | 3 | 1.000 | 4.497e-03 | 4.470e-07 | 3 |
| `bowl_stove_to_cream_cheese_bowl` | support_to_container | Frame-weighted target scratch | 3 | 0.000 | 1.489e-01 | 8.611e-06 | 3 |
| `bowl_stove_to_cream_cheese_bowl` | support_to_container | Phase scalar GP target scratch | 3 | 0.000 | 1.489e-01 | 8.611e-06 | 3 |
| `bowl_stove_to_cream_cheese_bowl` | support_to_container | Ours transfer | 5 | 1.000 | 3.451e-04 | 1.494e-08 | 3 |
| `bowl_stove_to_cream_cheese_bowl` | support_to_container | Pdiag finite target scratch | 5 | 1.000 | 1.216e-03 | 3.197e-07 | 3 |
| `bowl_stove_to_cream_cheese_bowl` | support_to_container | Frame-weighted target scratch | 5 | 0.000 | 1.516e-01 | 7.700e-06 | 3 |
| `bowl_stove_to_cream_cheese_bowl` | support_to_container | Phase scalar GP target scratch | 5 | 0.000 | 1.516e-01 | 7.700e-06 | 3 |
| `bowl_stove_to_cream_cheese_bowl` | support_to_container | Ours transfer | 8 | 1.000 | 3.451e-04 | 1.517e-08 | 3 |
| `bowl_stove_to_cream_cheese_bowl` | support_to_container | Pdiag finite target scratch | 8 | 1.000 | 4.611e-04 | 2.844e-07 | 3 |
| `bowl_stove_to_cream_cheese_bowl` | support_to_container | TP-GMM SE(3) target scratch | 8 | 1.000 | 1.928e-02 | 1.334e-05 | 3 |
| `bowl_stove_to_cream_cheese_bowl` | support_to_container | Frame-weighted target scratch | 8 | 0.000 | 1.491e-01 | 6.846e-06 | 3 |
| `bowl_stove_to_cream_cheese_bowl` | support_to_container | Phase scalar GP target scratch | 8 | 0.000 | 1.491e-01 | 6.846e-06 | 3 |

Summary: in both scenes, direct transfer keeps `M_transfer=1.000` for N=3/5/8
and has lower held-out pose-trajectory MSE than target-scratch baselines.  The
N=8 TP-GMM SE(3) check also reaches binary `M_transfer=1.000`, but with much
higher continuous alpha error and held-out trajectory MSE.
