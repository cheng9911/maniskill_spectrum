# Frozen Results: Full-Phase Generator Selectivity (Step 1 re-analysis)

This documents the full-phase-resolution re-analysis the user requested before
any long-horizon task is added: *"verify whether yaw is really 0 → 1 → 0."* The
existing fitted model (`benchmark_phase_switch_baselines.py`) only sees the four
insertion phases `PHASE_CODES=(3,4,5,6)` = align_keyed, enter_key, unlock_yaw,
circular_insert. This re-analysis reads the empirical response diagonal off the
held-out isolated interventions over **all seven** solver phases
(reach 0, grasp 1, lift 2, align_keyed 3, enter_key 4, unlock_yaw 5,
circular_insert 6), so the "per-phase generator selectivity" claim is grounded in
existing data, not a new task.

## Frozen protocol

```text
data      : keyed arm, rotated_Q1_seed_{20260818,20270818,20280818}.h5 (39/39)
interventions : held-out isolated, +-15/30 deg yaw, +-0.015 x/y translation,
            plus the zero baseline (identical to the Step-3 held-out set)
response  : phase-ENDPOINT task-local response ratio, evaluated at the last
            state the solver labels with each phase code:
              alpha_x   = (x_obs - x_ref)/d_x        (x-translation episodes)
              alpha_y   = (y_obs - y_ref)/d_y        (y-translation episodes)
              alpha_yaw = wrap_pi(yaw_obs - yaw_ref)/d_axial   (yaw episodes)
pooling   : mean over 12 held-out interventions (4 yaw + 2 x + 2 y per seed
            x 3 seeds)
script    : phase_switch_symmetry/analyze_full_phase_profile.py
outputs   : phase_switch_symmetry_multiseed/full_phase_reanalysis/
            {full_phase_endpoint_response.csv, full_phase_endpoint_summary.csv,
             full_phase_empirical_profile.csv, full_phase_profile.pdf,
             full_phase_verdict.json}
```

## Result — phase-endpoint response diagonal (mean, 12 held-out)

| phase | label          | α_x   | α_y   | α_yaw |
|-------|----------------|-------|-------|-------|
| 0     | Reach          | 0.000 | 0.000 | 0.000 |
| 1     | Grasp          | +0.001| -0.001| -0.002|
| 2     | Lift           | +0.001| +0.006| -0.001|
| 3     | Align keyed    | +1.002| +0.997| +1.000|
| 4     | Enter key      | +1.001| +1.013| +1.000|
| 5     | Unlock yaw     | +1.003| +1.014| +0.069|
| 6     | Circular insert| +1.005| +1.045| +0.041|

`yaw_is_0_1_0 = true`, `translation_is_0_then_1 = true`,
`translation_is_constant_1 = false`.

## Interpretation (frozen)

1. **The yaw generator is exactly 0 → 1 → 0.** `α_yaw` is ~0 in reach/grasp/lift
   (the peg is at a fixed pedestal pose independent of the socket intervention),
   ~1 in align_keyed/enter_key (the peg is aligned to the socket's keyed yaw),
   and collapses to ~0.07/0.04 in unlock_yaw/circular_insert (the peg is rotated
   to a fixed yaw 0). The transition is a step up at align_keyed and a physical
   *release* down through unlock_yaw — matching the frozen M0 finding that the
   keyed switch is detected consistently at progress ~0.61, not the 0.5 phase
   boundary, because the smooth profile keeps yaw relevance through the first
   part of unlock before dropping it.

2. **The translation generators are 0 → 1, NOT constant 1.** `α_x`, `α_y` are
   ~0 in reach/grasp/lift and ~1 from align_keyed onward. This is the correct
   and stronger statement of "per-phase generator selectivity": the whole
   response turns *on* at align_keyed (the peg is moved to the shifted socket
   centre), and only yaw turns back *off* at unlock_yaw (the peg is de-rotated
   to the circular bore). Reach/grasp/lift are pre-contact phases in which no
   generator responds.

3. **Why the model's 4-phase representation is not a loss.** The model fits
   `PHASE_CODES=(3,4,5,6)` and recovers `α_yaw = [1,1,0,0]` there. This
   re-analysis shows the excluded phases 0-2 are trivially `[0,0,0]` for every
   generator, by geometry (peg far from socket, fixed pickup pose). So the
   "selectivity" the model learns is not an artifact of truncating the curve; the
   pre-contact phases carry no response signal and adding them would only add
   three flat zero segments.

## Verdict (frozen)

The hypothesis behind the paper's "per-phase generator selectivity" is supported
by existing data at full phase resolution: **yaw is the single selective
generator (0 → 1 → 0), translation is gated on at align_keyed and stays on
(0 → 1), and both are 0 during pre-contact approach.** No long-horizon task is
needed to establish this claim.
