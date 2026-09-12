# Real phase identification: fixed rules and abstention

Use all 111 real Franka FK trajectories and raw dataset action/state. No task-controller phase ground truth exists. Output is an offline kinematic segmentation with explicit unknown contact labels; do not report phase accuracy without independent video annotation.

Target reference: per-condition terminal position and upward hand axis estimated only from episode_index%5==0 (10 upright,13 tilted); remaining 88 episodes are reference-held-out, but all data have been explored previously. Compare an episode-terminal-reference version as an explicitly retrospective sensitivity diagnostic. Neither reference determines peg-tip/hole contact geometry.

Main event rules, frozen before extraction:
- grasp/release from existing measured-gripper closed interval (threshold .5 and short-gap closure), retaining source window metadata;
- lift clearance: >=20 mm above grip TCP z, sustained observed span >=.15 s;
- assembly region: within 80 mm of reference terminal TCP, sustained >=.20 s;
- axial advance candidate: within 60 mm, hand-axis discrepancy <=5°, velocity along upward reference axis <=−2 mm/s and lateral speed <=20 mm/s, sustained >=.15 s after region entry;
- terminal settle candidate: within 20 mm, axis discrepancy <=5°, linear speed <=10 mm/s, angular speed <=5°/s, sustained >=.30 s; interval must overlap the pre-release terminal window, occur after region entry, and end before release;
- no candidate => missing event/uncertain stage, not an invented transition;
- contact onset is unknown for every episode unless independently annotated.

Centered smoothing: position Savitzky–Golay 9 frames/degree2; axis median7 then unit normalize; angular speed median7. This is OFFLINE and noncausal. Grasp/release and endpoint reference also use future information. Real deployment needs fixed calibrated target frames and a causal history-only FSM with hysteresis and dwell-time delay.

Sensitivity: strict thresholds angle3°/settle15mm and loose angle7°/settle25mm, fixed event durations; alternative own-terminal target frame. Record missingness and timing spread, not 'accuracy'. Raw action[8:15] is retained only as an uncalibrated auxiliary load feature (field semantics unverified), never as contact ground truth or Cartesian force.

Provide all frame features/labels, per-episode event table, confidence flags, timing audit, source provenance, plots, eight fixed video-review clips, browser annotation interface and an annotation-evaluation CLI. Manual event times include uncertainty bounds and visibility status. Evaluate onset error and tolerance coverage only after those labels exist. No fabricated annotations.
