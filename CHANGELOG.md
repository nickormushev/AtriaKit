# Changelog

All notable changes to AtriaKit are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
While the version is 0.x, minor releases may contain breaking changes.

## [Unreleased]

## [0.3.0] - 2026-11-16

### Added
- Optional dependency set `manuscript` in `pyproject.toml` with the exact versions used for the
  manuscript results (Python 3.12.11): `pip install "atriakit[manuscript]"`.

### Changed
- **Default preprocessing changed (breaking for users of the defaults):** the feature signal is now
  band-pass filtered at 0.5-150 Hz (was a 120 Hz low-pass), and no baseline correction is applied by
  default (the P-onset spline, `global_baseline_mode: p_onset_spline`, is now opt-in). Choose the
  baseline correction that suits your data. Configurations that set these values explicitly, such
  as `examples/atriakit_manuscript_config.yaml`, are unaffected.
- **Output values changed (supervised PTF only):** the supervised inflexion point
  (`inflection_point`, used by the supervised `ptf`) now uses the same criterion and code as
  `ptf_auto` (`morphology.steepest_inflexion`, on the full annotated window). Previously it used a
  separate forward-difference implementation that counted an exactly zero second difference as
  positive, so the inflexion point of an inverted biphasic wave (aVR) could land one sample late.
  On beats classified as biphasic the supervised `ptf` now equals `ptf_auto`; `ptf_auto`,
  `p_wave_morphology` and all other descriptors are unchanged.
- `calculate_inflexion` now takes the P-wave segment instead of its first derivative.

### Fixed
- `morphology_classifier`: a step of exactly zero now continues the current phase instead of
  counting as rising, so a perfectly flat stretch (e.g. the baseline that lead-combined windows add
  after a lead's own offset on a noise-free signal) no longer creates an extra phase. Signals
  without exactly-zero steps are unaffected.

### Tests
- The aVR inflexion-point test is no longer an expected failure, and the supervised PTF of aVR is
  tested at the same tolerance as V1. New test: supervised and default PTF agree on biphasic waves.
- Removed the expected-failure test that compared the morphology label between per-lead and
  lead-combined windows. It tested the effect of a different window, not whether the label is
  computed correctly; the flat-baseline case it exposed is fixed (see Fixed) and covered by a unit
  test of the classifier. The suite has no expected failures.
- The artifact-suppression test now reads `examples/atriakit_manuscript_config.yaml` (0.5-150 Hz
  band-pass, onset alignment) instead of its own 1-150 Hz configuration. `offset_amplitude` is
  checked on all beats except the first and last, which lie in the filters' edge transients.

### Documentation
- `examples/atriakit_manuscript_config.yaml`: the header comment now gives the 0.5-150 Hz band-pass
  actually used (it said 1-150 Hz), and every setting that overrides a package default is marked.
  Shannon entropy now uses 128 bins over -0.5 to 0.5 mV (was -2 to 2 mV, which left a typical
  P-wave in only 4-6 bins). The demo configuration `examples/pipeline_config.yaml` uses the same range.
- README: a Known Limitations section (threshold dependence of the morphology class and of the
  supervised PTF) and the `manuscript` install option.

## [0.2.1] - 2026-10-22

### Fixed
- **Feature values changed:** fragment detection now uses hysteresis. A turning
  point only splits the P-wave into fragments once the signal has reversed from
  it by the fragment amplitude threshold (`fragment_noise_multiplier` × noise),
  as in Murthy et al.'s "adjacent extrema more than 3 noise SDs apart". On noisy
  single beats, ripple previously cut every deflection into runs shorter than
  `min_fragment_length_ms`, so some beats had no fragments at all. Affects
  `fragment_count`, `fragment_width`, `fragment_height` and the VCG fragment
  features. `fragments_finder(..., hysteresis=False)` gives the previous behavior.

## [0.2.0] - 2026-10-16

### Changed
- `compute_all` now skips `axis` and the VCG features (NaN plus a warning) when
  the signal preprocessor uses normalization, since per-lead scaling distorts
  the amplitude ratios they depend on.
- **Feature values changed:** the `dispersion` column now defaults to per-beat
  (max - min across leads within each beat, assigned to that beat's rows)
  instead of one recording-wide max - min. Set `dispersion_per_beat: false` to
  restore the previous behavior.
- **Feature values changed:** `ptf` and `ptf_auto` now compute duration × depth
  of the negative trough (previously duration × max absolute amplitude).
  Segments with no negative part return `0.0` (previously a positive value).
  Empty segments still return `NaN`. Recompute any stored PTF features.
- **Renamed:** `get_onset_offset_angle` / `onset_offset_angle` are now
  `get_onset_offset_slope` / `onset_offset_slope`, and no longer pass the value through
  `arctan` — it is a slope in mV/ms, not a geometric angle. Numerically unchanged for
  typical P-wave slopes.
- **Feature values changed:** `estimate_noise`'s pre-onset window is now converted from
  ms to samples correctly. It previously rounded to 0 samples below 1000 Hz, so the noise
  estimate was always `0` there, silently disabling the fragmentation and morphology noise
  thresholds. Recompute fragment and morphology features (and supervised `ptf`) for
  recordings below 1000 Hz.
- **Feature values changed:** fragments now span inclusive boundaries. Interior fragments
  reach their closing extremum (previously one sample short), the last fragment is one
  sample narrower, and a monotonic wave counts as 1 fragment (previously 0). Affects
  `fragment_count` / `fragment_width` / `fragment_height` and the `vcg_*_fragment_*`
  columns; recompute stored values.
- **Feature values changed:** `min_fragment_length_ms` now defaults to `10.0` (was `0.0`),
  matching `fragments_finder`'s own default — now that fragment boundaries are correct,
  this is what filters out single-sample noise steps. Set to `0` to count every run.
- **Feature values changed:** supervised `ptf` now inverts aVR like `ptf_auto` already
  did; it previously returned `0.0` for a Negative-Positive aVR wave.
- **Feature values changed:** `dispersion` now uses each lead's own onset/offset
  (`onset_original` / `offset_original`) instead of the active ones; under the default
  `cross_lead` boundary mode it was always `0`.

### Added
- `per_beat` argument to `FeatureCalculators.dispersion` and
  `dispersion_per_beat` option in `FeatureComputationConfig` (default `True`).
  When enabled, dispersion is max - min of P-wave duration across leads within
  each beat, and each annotation row gets its beat's value.
- `ptf_zero_crossing` option in `FeatureComputationConfig` (default `False`)
  and `zero_crossing` argument to `FeatureCalculators.ptf` / `ptf_auto`.
  When `True`, unsupervised PTF uses the negative run starting at the
  positive-to-negative zero crossing (conventional PTF, Morris et al., 1964).
- `ptf_auto` raises `ValueError` if `seg_morph` length differs from `segment`.
