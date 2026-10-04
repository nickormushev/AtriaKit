# Changelog

All notable changes to AtriaKit are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
While the version is 0.x, minor releases may contain breaking changes.

## [Unreleased]

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
