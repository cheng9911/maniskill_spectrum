# QA record

- Verified 15 attempted trajectories (5 conditions × 3 seeds), 15 successes and 15 available terminal phases. All 12 tilt responses retained, no outcome-based exclusions or retries.
- Verified every HDF5 manifest hash against the frozen 10-degree context manifest and checked recorded seeds and intervention magnitudes.
- Checked SAPIEN quaternion conversion against an analytic 10-degree rotation and opposite-axis sign convention. Recomputed comparison means and real-minus-sim arithmetic from per-seed data.
- Sim terminal sensitivity: largest per-seed hand-axis response change between all phase6 frames, last half and last frame is 0.0561 degrees. There are 102 total phase6 frames; these are repeated measurements, not independent samples.
- Source figure preflight: 18 pass, 2 warnings, 0 failures. Warnings concern PNG instead of submission TIFF and random-number use (bootstrap only, not synthetic measurements). Vector PDF/SVG and 600-dpi preview are present.
- PDF text audit: minimum 6 pt, no text below 5 pt. Both rendered panels visually inspected; labels and intervals are unclipped, all simulated seed points shown. All mean-response and gap intervals are conditional exploratory uncertainty, not equivalence acceptance bands.
- No physical tilt-axis calibration, grasp-to-object calibration, real session hierarchy, exact phase correspondence or equivalence margin has been inferred from outcomes. Four simulation directions are reported individually and do not exhaust unknown real directions.
- Existing real raw data and previous simulation outputs unchanged. New outputs kept under this directory.
