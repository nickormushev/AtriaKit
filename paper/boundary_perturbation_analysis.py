"""Boundary-perturbation robustness table for the manuscript.

How much do the descriptors change when the P-wave boundaries are off by a few ms?

For each shift d, every onset and offset (per lead, per beat) moves by +d or -d with a
seeded random sign. This happens before AnnotationsLoader, so the cross-lead window is
rebuilt from the shifted boundaries.

Per descriptor (leads pooled, PTF in V1 only), with f_0 the original and f_d the shifted value:
    R_f(d) = median |f_d - f_0| / IQR(f_0)
    R_f95  = 95th percentile instead of the median, at WORST_CASE_SHIFT_MS
    ICC    = ICC(2,1), absolute agreement of f_0 and f_d, at WORST_CASE_SHIFT_MS
Angle differences are wrapped to [-180, 180).

Sections: 1 settings, 2 perturbation, 3 pipeline runs (cached), 4 table, 5 main.

Run:
    bash paper/run_perturbation.sh

Feature tables are cached in <out-dir>/conditions; a run with different annotations, config
or seed is refused. Only aggregate numbers are written.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark_pipeline import _load_config
from atriakit.configs.annotations_loader_config import AnnotationsLoaderConfig
from atriakit.configs.pipeline_config import PipelineConfig
from atriakit.io import AnnotationsLoader, ECGLoader
from atriakit.models.annotation_schema import AnnotationSchema
from atriakit.processing.pipeline import Pipeline

# ---------------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------------

DEFAULT_SHIFTS_MS = (2, 5, 10, 20)  # 20 ms: stress test
WORST_CASE_SHIFT_MS = 10  # shift for R_f95 and ICC: the largest plausible error

# Manuscript descriptors, in table order. heart_rate is left out: it doesn't use the boundaries.
FEATURES = [
    "duration",
    "area",
    "area_to_duration_ratio",
    "max_absolute_amplitude",
    "max_amplitude",
    "min_amplitude",
    "ptp_amplitude",
    "ptf",
    "dispersion",
    "atrial_rate",
    "complexity",
    "shannon_entropy",
    "sample_entropy",
    "fragment_count",
    "fragment_width",
    "fragment_height",
    "axis",
    "vcg_area",
    "vcg_axis_azimuth",
    "vcg_axis_elevation",
    "vcg_roundness",
    "vcg_flatness",
]
FEATURE_LEADS = {"ptf": "V1"}  # PTF is reported for V1 only
CIRCULAR_FEATURES_DEGREES = {"axis", "vcg_axis_azimuth"}

# ---------------------------------------------------------------------------
# 2. Perturbation
# ---------------------------------------------------------------------------


def ms_to_samples(milliseconds: float, sampling_rate: float) -> int:
    """Convert a duration to the nearest whole number of samples."""
    return int(round(milliseconds * sampling_rate / 1000))


def jitter_boundaries(raw: pd.DataFrame, shift_samples: int, seed: int) -> pd.DataFrame:
    """Copy of ``raw`` with each onset and offset moved by +-shift_samples.

    Signs are drawn independently per boundary, seeded by (seed, shift_samples).
    """
    rng = np.random.default_rng([seed, shift_samples])
    onset_shift = rng.choice([-shift_samples, shift_samples], size=len(raw))
    offset_shift = rng.choice([-shift_samples, shift_samples], size=len(raw))

    perturbed = raw.copy()
    perturbed[AnnotationSchema.ONSET] = raw[AnnotationSchema.ONSET] + onset_shift
    perturbed[AnnotationSchema.OFFSET] = raw[AnnotationSchema.OFFSET] + offset_shift

    if (perturbed[AnnotationSchema.OFFSET] <= perturbed[AnnotationSchema.ONSET]).any():
        raise ValueError(f"A {shift_samples}-sample shift put an offset at or before its onset.")
    return perturbed


# ---------------------------------------------------------------------------
# 3. Running the pipeline
# ---------------------------------------------------------------------------


def probe_sampling_rate(ecg_root: Path, raw: pd.DataFrame, loader: ECGLoader) -> float:
    """Sampling rate of the first referenced file, used to convert ms to samples."""
    first_file = raw[AnnotationSchema.FILE_PATH].iloc[0]
    return loader.load(str(ecg_root / first_file)).get_sampling_frequency()


def make_pipeline(
    ecg_root: Path, config_path: Path | None, loader: ECGLoader, sampling_rate: float
) -> tuple[Pipeline, AnnotationsLoader]:
    """Pipeline and annotations loader from the YAML config.

    Stops on any file whose sampling rate differs from ``sampling_rate``, because the
    ms-to-samples conversion assumes a single rate.
    """
    if config_path:
        loader_config, pipeline_config = _load_config(config_path)
    else:
        loader_config, pipeline_config = AnnotationsLoaderConfig(), PipelineConfig()
    pipeline = Pipeline(ecg_base_path=ecg_root, pipeline_config=pipeline_config, loader=loader)

    load_ecg = pipeline._load_ecg

    def load_and_check_rate(file_path):
        ecg = load_ecg(file_path)
        if ecg.get_sampling_frequency() != sampling_rate:
            raise ValueError(
                f"{ecg.get_sampling_frequency()} Hz file found, expected {sampling_rate} Hz: "
                "mixed sampling rates are not supported by this analysis."
            )
        return ecg

    pipeline._load_ecg = load_and_check_rate
    return pipeline, AnnotationsLoader(config=loader_config)


def prepare_cache(cache_dir: Path, signature: str) -> None:
    """Create the cache folder, or stop if it holds results for other settings."""
    signature_path = cache_dir / "signature.txt"
    if signature_path.exists() and signature_path.read_text() != signature:
        raise ValueError(
            f"{cache_dir} holds results for different settings; delete it or pick another --out-dir."
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    signature_path.write_text(signature)


def features_for(
    name: str, raw: pd.DataFrame, pipeline: Pipeline, annotations_loader: AnnotationsLoader, cache_dir: Path
) -> pd.DataFrame:
    """Feature table for one annotation set, read from the cache if present."""
    cache_path = cache_dir / f"{name}.csv"
    if cache_path.exists():
        print(f"  {name}: using cached {cache_path.name}")
        return pd.read_csv(cache_path, index_col=0)

    print(f"  {name}: computing...")
    annotations = annotations_loader.from_dataframe(raw)
    table = pipeline.run(annotations, cache_dir=None).get_df()
    table.to_csv(cache_path)
    return table


# ---------------------------------------------------------------------------
# 4. The table
# ---------------------------------------------------------------------------


def signed_change(original: pd.DataFrame, perturbed: pd.DataFrame, feature: str) -> pd.Series:
    """Per-wave (perturbed - original), wrapped for angles. NaN where either side is NaN."""
    change = perturbed[feature] - original[feature]
    if feature in CIRCULAR_FEATURES_DEGREES:
        change = (change + 180) % 360 - 180
    return change


def icc_absolute_agreement(first: np.ndarray, second: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single measurement. Drops NaN rows."""
    pairs = np.column_stack([first, second])
    pairs = pairs[~np.isnan(pairs).any(axis=1)]
    n_waves, n_measurements = pairs.shape
    grand_mean = pairs.mean()
    between_waves = n_measurements * ((pairs.mean(axis=1) - grand_mean) ** 2).sum() / (n_waves - 1)
    between_measurements = n_waves * ((pairs.mean(axis=0) - grand_mean) ** 2).sum() / (n_measurements - 1)
    residual = pairs - pairs.mean(axis=1, keepdims=True) - pairs.mean(axis=0, keepdims=True) + grand_mean
    error = (residual**2).sum() / ((n_waves - 1) * (n_measurements - 1))
    return float(
        (between_waves - error)
        / (between_waves + (n_measurements - 1) * error + n_measurements * (between_measurements - error) / n_waves)
    )


def table_row(baseline: pd.DataFrame, perturbed_by_shift: dict[float, pd.DataFrame], feature: str) -> dict:
    """One row: R_f at every shift, then R_f95 and ICC at WORST_CASE_SHIFT_MS."""
    lead = FEATURE_LEADS.get(feature)
    rows = baseline.index if lead is None else baseline.index[baseline[AnnotationSchema.LEAD] == lead]
    original = baseline.loc[rows]
    iqr = original[feature].quantile(0.75) - original[feature].quantile(0.25)
    row = {"feature": feature if lead is None else f"{feature} ({lead})", "n_waves": int(original[feature].notna().sum())}

    for shift_ms, perturbed in sorted(perturbed_by_shift.items()):
        change = signed_change(original, perturbed.loc[rows], feature).abs()
        row[f"R_f_{shift_ms:g}ms"] = change.median() / iqr if iqr else np.nan

    change = signed_change(original, perturbed_by_shift[WORST_CASE_SHIFT_MS].loc[rows], feature)
    row[f"R_f95_{WORST_CASE_SHIFT_MS:g}ms"] = change.abs().quantile(0.95) / iqr if iqr else np.nan
    # add the wrapped change so angles stay on the same branch as the original
    row[f"ICC_{WORST_CASE_SHIFT_MS:g}ms"] = icc_absolute_agreement(
        original[feature].to_numpy(float), (original[feature] + change).to_numpy(float)
    )
    return row


# ---------------------------------------------------------------------------
# 5. main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", required=True, help="Annotation CSV (raw per-lead table)")
    parser.add_argument("--ecg-root", required=True, help="Directory containing the ECG files")
    parser.add_argument("--config", default=None, help="Pipeline YAML config (defaults used otherwise)")
    parser.add_argument("--shifts-ms", type=float, nargs="+", default=list(DEFAULT_SHIFTS_MS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", required=True, help="Where the cache and the table are written")
    args = parser.parse_args()
    if WORST_CASE_SHIFT_MS not in args.shifts_ms:
        parser.error(f"--shifts-ms must include the worst-case shift ({WORST_CASE_SHIFT_MS:g} ms)")

    out_dir = Path(args.out_dir)
    ecg_root = Path(args.ecg_root)
    raw = pd.read_csv(args.annotations)

    config_text = Path(args.config).read_text() if args.config else ""
    annotations_hash = hashlib.sha256(Path(args.annotations).read_bytes()).hexdigest()
    cache_dir = out_dir / "conditions"
    prepare_cache(cache_dir, f"annotations sha256={annotations_hash} seed={args.seed}\n{config_text}")

    loader = ECGLoader()
    sampling_rate = probe_sampling_rate(ecg_root, raw, loader)
    pipeline, annotations_loader = make_pipeline(ecg_root, Path(args.config) if args.config else None, loader, sampling_rate)
    print(f"{raw[AnnotationSchema.PATIENT_ID].nunique()} patients, {len(raw)} annotation rows, {sampling_rate:g} Hz")

    baseline = features_for("baseline", raw, pipeline, annotations_loader, cache_dir)
    perturbed_by_shift = {}
    for shift_ms in args.shifts_ms:
        shift_samples = ms_to_samples(shift_ms, sampling_rate)
        perturbed = features_for(
            f"jitter_{shift_ms:g}ms", jitter_boundaries(raw, shift_samples, args.seed), pipeline, annotations_loader, cache_dir
        )
        if not perturbed.index.equals(baseline.index):
            raise ValueError(f"{shift_ms:g} ms: rows differ from the baseline, cannot pair them.")
        perturbed_by_shift[shift_ms] = perturbed

    table = pd.DataFrame(
        [table_row(baseline, perturbed_by_shift, feature) for feature in FEATURES if feature in baseline.columns]
    )
    table_path = out_dir / "boundary_perturbation_robustness.csv"
    table.to_csv(table_path, index=False)

    print(f"\nR_f = median|f_d - f_0| / IQR(f_0); R_f95 and ICC at {WORST_CASE_SHIFT_MS:g} ms")
    print(table.set_index("feature").to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nTable saved to {table_path}")


if __name__ == "__main__":
    main()
