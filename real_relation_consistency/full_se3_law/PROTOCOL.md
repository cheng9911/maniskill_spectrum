# Full six-generator circular-task refit protocol

Use existing CircularPhaseSwitchSE3-v1 circular_honest data, seeds 20260818/20270818/20280818. Not the keyed or multigen task. This is a new retrospective fit on existing data, not newly collected intervention evidence.

Audit every attempt. For each condition choose the earliest usable attempt (success, all required phases, not truncated); retain failures in the audit and disclose retry-conditioned identification. No test-error-based choice.

Train on nominal baseline and 45 of 60 mixed conditions: every fourth condition in sorted mixed IDs is reserved for evaluation (15 mixed). All 14 isolated intervention conditions are additionally reserved (including translation ±15 mm, roll/pitch ±15°, yaw ±15/30°). Thus 46 fitting and 29 evaluation conditions per seed. ±15 mm isolated translations exceed the ±12 mm mixed training range: evaluate as limited extrapolation, not interpolation.

Use the existing SE3SmoothFinitePDiagModel, alpha_max=1.25, n_basis=24, width=.065, smoothness=.1, nominal_iterations=3. No real data used. Check normalized raw-context and SE3-log twist rank=6 before fitting. Freeze all six alpha channels, nominal curve/frame, parameters, training/evaluation IDs and provenance. Compare held-out positions, full orientation and peg axes against no-adaptation, identity, and existing shared frame-scalar baseline.

Model is a six-channel diagonal finite action, not an identified unconstrained 6×6 operator. Coordinates are the nominal socket frame [du,dv,dw,roll,pitch,yaw]; translational twist channels use meters and angular channels radians. Real application requires a nominal task frame and phase correspondence; a base-frame transform must be conjugated into that frame before applying the diagonal law.

Figure contract: Python 180 mm six-channel profile, all 3 source fits with mean±SD; CSV source data; PNG/PDF/SVG/TIFF exports. Four phase labels are simulator control labels, not real contact ground truth.
