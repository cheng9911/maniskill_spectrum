# Verification record

- Real data: 111/111 episodes and 94,752/94,752 frames exported; 111 terminal windows, no exclusions. Every sensitivity setting retains all 111 episodes (999 rows total).
- FK: joint origins match the local official FER description. Independent modified-DH comparison on 20 configurations has maximum elementwise error 4.996e-16; rotation orthogonality, determinant and exported unit quaternions checked.
- Simulation: all 12 isolated roll/pitch trials included, across three seeds; all report success. Three nominal trials are also exported. This is an existing +/-15-degree reference, not a new 10-degree matched experiment.
- Figure static preflight: 17 pass, 3 warnings, 0 failures. PDF audit: minimum font size 6 pt, zero sub-5-pt text runs.
- Warning resolution: PNG is a 600-dpi preview alongside vector PDF/SVG, not a TIFF submission asset; dropna removes no rows in this run (counts verified); random numbers are for FK checks, bootstrap and horizontal point jitter only, not synthetic observations. Vertical plotted values are measured/FK-derived data. Numerical simulation trials are explicitly labelled as simulation.
- Visual QA: all four panels inspected after final rendering. Axis labels, legends, points and uncertainty bands are legible and unclipped. Panel c's normalized time is explicitly not a matched simulation phase; panel d has a deliberately narrow labelled scale for window sensitivity.
- Video: 13 screenshots inspected; peg near/in assembly location supports a terminal proxy. Two sampled episodes have mismatched video/data spans, so those screenshots do not confirm exact frame alignment. Six mismatched spans in total remain flagged; no success labels inferred from this check.
- Inferential status: exploratory hand-axis contrast. Conditional episode bootstrap intervals do not include unknown session clustering or calibration error. No equivalence threshold, full generator law, yaw invariance or sim-to-real deployment is claimed.
