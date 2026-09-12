# QA

All 111 episodes/94,752 frames retained. Event ordering checked, calibration references restricted to episode%5==0. After independent review, stable-stage labels now exit on renewed motion and reacquire with dwell; full-frame audit found zero stable labels violating exit hysteresis. Missing events and six video-duration mismatches remain explicit. No contact or insertion-success labels are fabricated.

Causal prototype passes prefix-invariance replay, showing unchanged history when future samples are omitted. Replay coverage is reported separately from offline coverage, not as accuracy. The prototype may miss stages; it requires external calibrated geometry and further validation for deployment.

Evaluator regression tests use clearly synthetic software fixtures in temporary directories: explicit blindness required, exposed/unknown labels stratified, no-label run removes stale scores. Real annotation template is empty, so the actual annotation evaluation status remains NO_INDEPENDENT_OBSERVED_LABELS. Mean event-interval error is conditional on detections; tolerance fractions include missing predictions.

Eight prespecified clips encoded successfully. Annotation page JavaScript syntax passes node --check and all eight referenced videos exist. Automatic proposals hidden by default; exposure recorded in exports. No full browser interaction automation available; browser UX not claimed end-to-end tested. Clip fps15 vs raw30Hz bounds visual time precision. Metadata duration agreement is not exact synchronization validation.

Phase feature and timeline source preflights: 20 pass each. PDF minimum glyphs 5.5 and 6 pt. Visually inspected feature panels and both timeline panels; all episode rows shown with unknown spans gray. Example feature plots use episode0 by fixed rule, not selecting best agreement. No statistical uncertainty band is implied by deterministic timeline colors. Contact accuracy cannot be measured without independent annotations.
