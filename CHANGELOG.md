# Changelog

All notable changes to AtriaKit are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
While the version is 0.x, minor releases may contain breaking changes.

## [Unreleased]

### Changed
- **Feature values changed:** `ptf` and `ptf_auto` now compute duration × depth
  of the negative trough (previously duration × max absolute amplitude).
  Segments with no negative part return `0.0` (previously a positive value).
  Empty segments still return `NaN`. Recompute any stored PTF features.

### Added
- `ptf_zero_crossing` option in `FeatureComputationConfig` (default `False`)
  and `zero_crossing` argument to `FeatureCalculators.ptf` / `ptf_auto`.
  When `True`, unsupervised PTF uses the negative run starting at the
  positive-to-negative zero crossing (conventional PTF, Morris et al., 1964).
- `ptf_auto` raises `ValueError` if `seg_morph` length differs from `segment`.
