# Tilt sweep implementation and evidence review

Review is read-only with respect to original training, frozen, and prediction artifacts. Supplementary audit outputs are isolated here.

## Verified numerical work

- 33 formal training episodes, all final-success and phases 3–6 complete, no truncated episodes, attempt_id=0.
- Actual inputs are exactly 0, ±2.5, ±5, ±7.5, ±12.5, ±15 degrees. No ±10-degree examples were used by the fitter.
- Manifest, HDF5, collector/environment, and recorded model-source hashes match.
- Independent in-memory refits for all three seeds reproduce saved model parameters, diagonals and nominal curves exactly (maximum difference 0); optimization costs and nfev match (2/8/2).
- Reconstructed frozen models reproduce stored +10-degree predictions exactly. Mean 10.01829691184701 degrees; sample SD 0.032111788776620555. Real point 10.299727607946036 degrees.
- Separately calculated -10 predictions are saved in reproduced_predictions.json. They are numerically almost equal in magnitude to +10 for these artifacts, but the original plot did not compute them.

## Findings

### P1 — Target nominal trajectory is not used

`real_relation_consistency/predict_10degree.py:177–189` loads the **simulation** nominal frame and nominal curve and predicts a simulated peg-axis response. The real CSV is first read at lines 206–208 solely to calculate a scalar observed response.

The implementation supports a frozen simulated response prediction compared with a real terminal magnitude. It does not implement C_real / X0_real based trajectory prediction, target progress correspondence, or a real response-error evaluation. The phrase “full workflow predicts real trajectories” would overstate the implementation.

### P1 — Prediction-only straight line is not validation; 10 degrees is interpolation

`predict_10degree.py:192–200` generates a grid entirely from model outputs. It does not read held-out simulated trajectories or measured sweep responses for that plot. Near-linearity is largely imposed by scaling a pure pitch twist with fixed alpha(s). It cannot establish empirical strict linearity, predictive accuracy, or extrapolation.

±10 lies inside the training domain [−15,+15]. Correct term: held-out-amplitude **interpolation**. REPORT.md:39–46 and QA.md currently confuse interpolation, extrapolation, and independently evaluated accuracy.

### P1 — Transient profile has only simulation evidence

The recovered rise of alpha during align reflects the response of the executed simulated trajectories and their phase normalization. `collect_phase_switch_rotated.py:243–270` explicitly builds tilted target poses from socket pitch and schedules their execution in align. It is not independently evidence of contact-constraint release or real phase-dependent agreement. No real transient is compared by the new prediction script.

The appropriate test is phase-resolved prediction error on held-out trajectories. This review added a separate numerical check below, without retraining.

### P1 — Simulated hand–peg offset is not a measured hardware error bound

`predict_10degree.py:273–283` refers to an established hand–peg proxy gap when assessing the real comparison. The earlier ~0.1–2 degree gap was measured in simulation, not by independent real gripper-to-peg calibration. It cannot establish that the real difference is within a known hardware error budget. Retain it as motivation to calibrate, not as a validated real tolerance.

### P2 — Negative-amplitude plot and identity baseline use inconsistent definitions

The response function returns a nonnegative angle. `predict_10degree.py:142` draws identity as y=x on negative inputs; it should use |x| for this magnitude definition, or change all outputs to a defined signed response. Line 143 copies the +10 prediction to both x=−10 and x=+10 and labels both “+10”. The original grid only evaluates nonnegative inputs. The real +10 observation band also extends across the entire axis, potentially suggesting observations at other amplitudes.

### P2 — Reproducibility safeguards and diagnostics

- `identify_pitch_law.py:73–89` does not enforce the input amplitude split or manifest hash; summary lines 185–186 hardcode split labels. Current data are clean, but a different --sweep-dir could silently produce a falsely labelled freeze.
- `identify_pitch_law.py:156` console mean_insert includes unlock+insert (`2*bins:`); insert alone is `3*bins:`. Report phase means are correct.
- Unexcited channels are unidentifiable. Their numerical near-zero values must not be interpreted as irrelevance or a full six-dimensional law.
- PDF text audit finds one 4.9 pt text run (alpha math subscript), below the selected figure skill's 5 pt floor. Visual inspection confirms the signed/magnitude plot issue.
- “Law minus real lies within real CI” is dimensionally the wrong comparison: the **prediction** 10.018 lies inside [9.60,11.05]; difference −0.281 needs its own uncertainty interval. Neither proves equivalence.

## Supplementary independent held-out evaluation

Inputs: six existing pitch ±10-degree episodes in `sim_10degree/circular_seed_*.h5`, three seeds. No model refit, real-data tuning, or trajectory exclusions. For each seed, use the saved simulation nominal and saved diagonal. Identity and no-adaptation use the same nominal, with diagonals fixed to one or zero. All original model/source artifacts unchanged.

Actual peg poses use the repository's task_curve_se3 resampling (25 bins each for phases 3,4,5,6). Predicted and actual world peg z axes are compared by angle; position error is Euclidean. Compute RMSE within each trajectory, then average the six trajectory RMSEs. The all-phase metric weights phases equally, not according to raw duration. These are descriptive results on previously seen-in-development evaluation data, not a new blinded confirmation experiment.

| Window | Frozen axis RMSE | Identity axis RMSE | No-adaptation axis RMSE |
|---|---:|---:|---:|
| All four phases | 0.575° | 2.563° | 9.473° |
| Align | 1.120° | 5.121° | 7.857° |
| Insert | 0.138° | 0.125° | 9.930° |

All-phase position RMSE: frozen 4.898 mm; identity 15.201 mm; no adaptation 16.935 mm. Full per-seed, sign, method, and phase values are in independent_heldout_check.csv.

This supports a concrete revised conclusion: the frozen pitch profile improves simulated held-out-amplitude transient prediction over the constant identity baseline, primarily during alignment. It does not outperform identity at terminal insertion, and real transient transfer is still untested.

## Recommended next work

1. Correct the scope and plot; call ±10 an interpolation test and distinguish simulated peg prediction from real hand observation.
2. Integrate actual held-out trajectory error evaluation with saved provenance and baselines into the pipeline.
3. For real trajectory prediction, provide independently justified target frame/axis and hand–peg mapping, extract real nominal assembly curves, and fix phase correspondence. Evaluate on real intervention trajectories without tuning the frozen law to them.

Overall: numerical identification and frozen inference are reproducible; report and QA conclusions currently overstate what the original prediction pipeline evaluates.
