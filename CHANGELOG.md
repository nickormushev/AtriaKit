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
