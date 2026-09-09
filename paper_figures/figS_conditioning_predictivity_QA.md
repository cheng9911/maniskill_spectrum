# Supplementary conditioning-predictivity figure QA record

## Statistical scope

- Each of the 12 points is the frozen primary within-(N, protocol) Spearman correlation between context condition number kappa(C) and Pdiag generator-profile RMSE.
- Each correlation uses 10 frozen subsets after averaging five execution seeds within each subset.
- No rank inversion, cross-protocol pooling, pooled regression, or display offsets are used.
- The source-data CSV is copied directly from the frozen primary correlation table after filtering only to this declared metric/recovery pair.

## Interpretation boundary

- This is a narrow negative result: the simple conditioning statistic does not provide a stable, protocol-invariant ranking of few-shot subset quality.
- It does not make a claim about all possible identifiability measures or function-space identifiability of the generator law.

## Export and visual QA

- Python/matplotlib only; no simulated data or exclusions.
- PNG: 400 dpi; TIFF: 600 dpi; PDF and SVG retain editable text.
- Static preflight: 20 pass, 0 warnings, 0 failures.
- PDF glyph audit: 27 text runs; minimum 5.75 pt; no glyph below 5 pt.
- PDF and SVG contain no embedded raster images.
- Final-size visual inspection: title, zero reference lines, and panel labels are clear without overlap or clipping.
