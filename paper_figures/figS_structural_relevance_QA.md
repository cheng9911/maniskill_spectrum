# Figure 4 QA notes

## Figure contract

- Core conclusion: the learned Pdiag-finite generator relevance follows phase,
  symmetry and task-specific relation constraints rather than a fixed coordinate
  pattern.
- Archetype: strict 2×2 quantitative grid with two full-structure panels and
  two targeted counterfactual controls.
- Panels a--b: full six-generator heatmaps for full-SE(3) insertion and
  multi-generator release. Panels c--d: yaw-only phase strips contrasting
  keyed versus circular symmetry and heading-constrained versus free-yaw pushes.
- Center statistic: mean relevance within each phase over frozen seed--subset
  fits at N=30.
- Replicate unit: one frozen seed--subset fit; n=18 in every displayed heatmap.

## Data and uncertainty

- All 144 phase/generator cells are exported in the accompanying source-data
  table with their mean, across-fit standard deviation and an explicit
  `displayed_in_figure` flag. The rendered figure uses 64 cells: all 48 cells
  in a--b, plus the decisive eight yaw cells in c and eight in d.
- The heatmap color encodes only the mean. Error bars are not compatible with
  the cell-grid encoding; uncertainty is retained in the source-data table so
  every displayed cell remains auditable.
- No fits, tasks, phases or generators are excluded from the source data. The
  c--d reduction to the yaw generator is a declared evidence-design choice that
  prevents the keyed control from redundantly repeating panel a.
- No simulated, reconstructed or hand-entered relevance values are used.

## Export and inspection

- Heatmap cells are rendered as vector rectangles, not raster images.
- PDF/SVG use editable text; TIFF is exported at 600 dpi.
- The colorbar is limited to 0--1.05 because every displayed phase mean is at
  most 1.013; no displayed mean is clipped. The raw profile range remains
  0--1.25 and is not re-scaled in the source data.
- Visual inspection checked the strict outer 2×2 layout, phase labels, shared
  color mapping, panel/title hierarchy and colorbar clearance at final size.
