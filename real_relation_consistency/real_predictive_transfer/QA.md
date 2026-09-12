# Numerical and figure QA

- Calibration 13 and evaluation 48 target episodes are disjoint; all 61 are accounted for. All 50 nominal episodes are candidates; 10 distinct references are used by evaluation episodes.
- 1,728 metric rows = 48 episodes × 4 methods × 3 event radii × 3 segments. No missing or selectively excluded combinations.
- Recomputed primary angle RMSE from saved unit axes using independent atan2(cross,dot), rather than the generating arccos; maximum disagreement below 1e-8 degree. Position RMSE independently recomputed.
- Quaternion norms, unit axes, quaternion-to-axis consistency, no-adaptation identity, homogeneous SE(3) action, effective-pivot nullspace invariance and one-generator scalar equivalence checked.
- All input hashes agree with provenance; frozen law was not modified.
- Figure source preflight: 20 pass, 0 warnings, 0 fail. All three PDF text audits pass; smallest text 5.5/5.5/6 pt.

## Rendered panels

| Figure/panels | Role and data | Spread / sample unit | Inspection |
|---|---|---|---|
| Summary a | All nominal pickup positions and calibration/evaluation support | Raw 50 + 13 + 48 episode positions | Readable; no forced three-context grouping |
| Summary b | Representative response vs four predictions | Single matched episode; no inferential band | Legend moved clear of response curves |
| Summary c | Four-method axis RMSE over apex-to-end | All 48 episode dots; black median and IQR | Error extents contained; no clipping |
| Summary d | Fixed axis-direction sensitivity | Mean over same 48 episodes, conditional deterministic sensitivity, not uncertainty interval | Units and offsets explicit |
| Components a–c | Representative TCP xyz | Single episode and deterministic predictions | Labels and shared method colors readable |
| Components d–f | Representative unit hand-axis xyz | Same single episode | Full components displayed, spin unverified |
| Spatial a–b | Representative top and side projections | Single episode; four event markers | Equal metric aspect per projection; no independent hole-location marker |

No p-values, equivalence claims, fake successful executions or independent hole-calibration claims. Input geometry is estimated from real calibration demonstrations, source-law coefficients remain frozen. Data previously inspected; evaluation is retrospective. Nominal reuse and unknown session dependence prevent treating episode points as independent physical interventions. No uncertainty claim is made for calibration or sensitivity curves. Target event timing uses future target positions, so outputs are offline diagnostics.
