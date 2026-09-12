# Prospective computational protocol for retrospective data

Frozen before computing predictive scores. Data have been explored previously; this is not a fresh confirmatory test.

- Real nominal: all 50 upright episodes; target: 61 tilted episodes. Geometry calibration uses tilted episode_index % 5 == 0 (13 episodes); evaluation uses the other 48. No target evaluation orientation enters calibration or matching.
- Match each tilted trajectory to the upright trajectory with nearest grasp TCP xy. No forced E1/E2/E3 labels; publish distances and reuse counts.
- Use the existing Franka FK caches. Fixed grasp is an assumption. Estimate the minimal rotation direction between mean calibration terminal hand axes and their matched nominal terminal axes. Keep input angle fixed at the user-reported +10°, rather than fit its amplitude.
- Estimate an effective fixed pivot by least squares: (I-R)o = mean(p_tilt - R p_nominal) on calibration terminal TCP windows. The rotation-axis nullspace is anchored to mean nominal terminal position. This is a response-derived effective pivot, not an independently calibrated hole center. Report residuals and axis/position assumptions.
- Frozen pitch law is never refit. Map its four 25-bin source phases (3,4,5,6) through each training baseline trajectory into grip/apex/near/terminal event coordinates. Before source align, response is explicitly extended by zero. Source mapping uses simulation only.
- Event extraction reuses the fixed kinematic detector; primary radius 30 mm; sensitivity 20/40 mm. Target event timing uses target position retrospectively; predictions are offline and not online generation.
- Apply Exp(alpha(s)*10°*[axis]x) on the left to the real nominal TCP rotations and rotate positions about the effective pivot. Save full quaternions but score axis direction (not axial spin) and position separately.
- Baselines: no adaptation; full rigid identity response; fixed linear event ramp (0 at grip, 1 at near) as a heuristic scalar. Under one active generator, scalar w=alpha is algebraically identical to the frozen law and is verified numerically, not advertised as a distinct competitor.
- Score apex-to-terminal and near-to-terminal separately, plus whole interval. RMS Euclidean position in mm and axis-angle in degrees per target episode. No pooling frames as independent replicates. Show episode distributions; no confirmatory p-values.
- Sensitivity: event radius, estimated axis azimuth ±15°, pivot alternatives (nominal terminal TCP and 50/100 mm along the nominal downward hand axis). Never select a sensitivity setting using evaluation error.
- Plot contract: Python quantitative grid, 180 mm width; setup support, trajectory prediction, method errors, and calibration sensitivity. All 48 evaluation episodes contribute; representative trace uses median pickup matching distance, not prediction quality. Export PNG/PDF/SVG/TIFF plus source CSV/NPZ, provenance and QA.
