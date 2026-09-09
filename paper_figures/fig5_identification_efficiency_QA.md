# Figure 5 QA record

## Figure contract

- Core conclusion: the structured diagonal prior recovers the complete generator law and reproduces the task with few demonstrations.
- Panel a: matched generator-profile error across the four models.
- Panel b: matched downstream task-reproduction error across the same models.
- Panel c: exact complete-law recovery fraction for frozen Pdiag fits.

## Data and statistics

- Panels a--b retain all successful fits in cells matched across Pdiag finite, TP-GMM SE(2), Full operator, and Generic RBF: 25 matched seed--subset cells at N=5, 10, and 20, plus five shared full-data fits at N=30. Points are arithmetic means and error bars are descriptive s.e.m.
- Panel c applies the preregistered complete-law criterion to every frozen Pdiag fit, including the retained optimizer-limit fit: random N=3 = 44/50; qualified N=3 = 46/50; every random and qualified frozen subset cell at N=5, 8, 10, 15, and 20 = 50/50; shared N=30 full-data fits = 5/5.
- N=30 is shown once in panel c because it is one common full-data condition with five seeded fits, not independent random and qualified protocol evidence.
- The displayed panel-c milestones are N=3, 5, 10, 20, and 30, matching the main-figure comparison axis. Frozen N=8 and N=15 cells are retained in source data and summarized by the explicit “every frozen subset family at N ≥ 5” annotation.
- Panel c is an exact pass fraction over the frozen evaluation set, so it uses numerator/denominator annotations rather than a model-based uncertainty interval.
- `fig5_identification_efficiency_source_data.csv` contains 32 panels-a/b summaries and all 13 panel-c protocol/size fractions.

## Correction from the previous panel c

- The earlier rank-scatter used a transformed, pooled statistic: it reversed within-cell condition-number ranks and then pooled them across protocols before correlating ranks. That statistic was not the frozen primary identifiability analysis and has been removed from the main figure.
- The primary figure no longer makes any conditioning-predictivity claim. The correctly scoped negative-result visualization is supplied separately as `figS_conditioning_predictivity`.

## Export and visual QA

- Python/matplotlib only; no simulated data or row exclusions.
- PNG: 400 dpi; TIFF: 600 dpi; PDF and SVG retain editable text.
- Static preflight: 20 pass, 0 warnings, 0 failures.
- PDF glyph audit: 74 text runs; minimum 5.25 pt; no glyph below 5 pt.
- PDF and SVG contain no embedded raster images.
- Final-size visual inspection: panel labels, legends, annotations, and shared explanatory text are clear without collisions.
