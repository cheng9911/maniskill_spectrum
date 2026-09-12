# QA

- Numeric artifact checks pass for all six combined trajectories: sample counts, strictly increasing timestamps, unit/sign-continuous quaternions, exact shared assembly suffixes, gripper-event accounting and common entry.
- Independently reconstructed the finite rotation and pivot action from saved CSVs. Inputs match provenance hashes; no source-law fit performed.
- Figure source preflight passed 20 checks; exported PDF minimum glyph 5.5 pt. Visual review of all four panels: resolved initial 3-D label overlap by increasing panel spacing and reducing tick density. Both 3-D panels use equal metric coordinate scaling. Top panels distinguish connectors and assembly; lower panels display deterministic applied rotation and position displacement, not measured hardware outcomes or confidence intervals.
- Three default pickup translations are design choices, not measured new experimental conditions.
- Scope remains offline Cartesian generation. No IK, joint limits, obstacle/self-collision, attached-object clearance, actuator gripper commands, dynamic limits or physical contact validation. The retained nominal geometry may contain corners despite smooth endpoint timing.
