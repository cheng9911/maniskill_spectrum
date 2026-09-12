# Empirical matrices for Fig. 3c

## Data and selection

Frozen keyed physics-simulation trajectories, seeds 20260818, 20270818 and
20280818. All six existing N=30 subsets (three random, three qualified) are
used for every seed. Each subset has 30 mixed interventions; the union is
59 unique conditions per seed. A zero-intervention baseline is also used.

There are 90, 83 and 83 recorded attempts, respectively. Following the
existing benchmark, usability requires terminal success, all four task
phases and no truncation. The last usable attempt in HDF5 key order
represents each condition. Each file has 75 usable conditions. Only the
frozen mixed-condition subsets and baseline enter this analysis; no
requested subset or unsuccessful fit is silently removed. JSON provenance
lists source hashes, selected episodes, condition IDs, seeds and subsets.

## Estimator

Generator order: [du, dv, dw, roll, pitch, yaw]. Rotation convention:
R = Rx(roll) Ry(pitch) Rz(yaw). Translation and unwrapped Euler angles are
resampled to 25 progress points per phase, following the benchmark's phase
interpolation convention; rotations are then converted back to matrices.

Let U_i be the recorded intervention, C0 the nominal socket frame, X_i(s)
the peg pose, and X0(s) the baseline peg pose. Define metric-scaled twists:

    x_i = D Log(U_i)
    y_i(s) = D Log(C0^-1 X_i(s) X0(s)^-1 C0)
    D = diag(1, 1, 1, 0.03, 0.03, 0.03).

Translations are in metres and rotations are scaled by 0.03 m/rad,
matching the model layer. Each intervened socket pose times U_i^-1 must
agree with baseline C0 within 2e-5 in matrix entries, or rebuilding stops.

At each progress point fit the unconstrained affine map by least squares:

    y_i(s) = A(s) x_i + b(s) + residual_i(s).

All 36 entries are estimated without diagonal constraints, symmetry,
regularization, thresholding or post-fit scaling. All 18 designs have rank
7; the column-standardized input condition numbers range from 1.82 to 2.42.
Average A(s) within each phase, then equally over fits. These are
finite-intervention spatial-response estimates, not the stored world-pose
Full operator coefficients or the Pdiag learned relevance coefficients.

## Paired change of basis

Use the frozen rotated-1 angles (30°, −20°, 40°), Rx Ry Rz order, with
Q = diag(R,R). Both inputs and outputs change coordinates:

    x_rot = Q^T x_local
    y_rot = Q^T y_local
    A_rot = Q^T A_local Q.

D commutes with Q. The rotated matrix is independently refit using the
transformed samples. Every fit and progress point verifies equality to
Q^T A_local Q and invariance of the Frobenius norm (atol 2e-10).
Changing only the input would produce A_local Q: a different map that
must not be labelled as a simultaneous change of generator basis.

## Display and interpretation

Display the insert phase to examine anisotropy after yaw release. Rows
denote output and columns input, in each panel's stated basis. The signed
coefficient scale is −1.05 to 1.05; entries are not converted to absolute
values. The annotation is the ratio of norms of the *mean matrix*:

    r_off = ||mean A − Diag(diag(mean A))||_F / ||mean A||_F.

The values are 0.050262 (local) and 0.312619 (rotated). The local yaw
diagonal is approximately 0.005; rotation mixes the anisotropic rotational
response and creates off-diagonal coupling. Enter is approximately
isotropic: both ratios are 0.021. Align ratios are 0.406 and 0.402; local
diagonality is not a claim about every phase. All four phases are exported.

The *per-fit* insert ratios have mean ± SD 0.07249 ± 0.01003 (local) and
0.31655 ± 0.00352 (rotated). These are different statistics from the
ratios of mean matrices shown under the heatmaps. Fits comprise three
execution seeds and six overlapping subsets, not 18 independent seeds.
Coefficient SDs are exported; the heatmaps show descriptive means, not
significance tests. The source is simulation, not physical-robot testing.

## Reproducibility

- `rebuild_fig3_basis.py`: estimator from frozen HDF5 files and subset manifest.
- `fig3_basis_empirical.npz`: per-fit [18,2,4,6,6] matrices, means, SDs,
  rotation, metric scales, seeds and subset IDs.
- `fig3_basis_empirical.csv`: all 288 mean coefficients and their SDs.
- `fig3_basis_empirical.json`: provenance, fit diagnostics, ratio summaries.
- `fig3_se3_generators_publication_basis.csv`: the 72 displayed coefficients.

Both Fig. 3 scripts load this package and fail if it is missing. Historical
0.15/2.77 annotations are not estimation targets and are not reproduced by
rescaling. Their exact earlier probe and coordinate convention would need
to be recovered before making a like-for-like numerical comparison.

Numerical QA includes an independent scipy matrix-exponential check of the
SE(3) logarithm (zero, small and finite rotations), rotated refit
equivalence, singular-value invariance, equality of CSV/NPZ means and
source hash verification. Both exported Fig. 3 PDFs pass the 5 pt glyph
floor. The publication layout was visually checked after label placement.
