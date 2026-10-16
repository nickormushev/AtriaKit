"""Per-patient wall-clock benchmark of the AtriaKit ``Pipeline``.

Patients run one at a time in a seeded random order. Loading and the rest of the
pipeline are timed separately; optional resampling is not counted. Z-score statistics, if the config
uses them, are computed once before timing starts.

Only counts and timings are printed or saved, no patient identifiers.

Usage:
    python paper/benchmark_pipeline.py \\
        --annotations /path/to/annotations \\
        --ecg-root /path/to/ecg/files \\
        --config examples/atriakit_manuscript_config.yaml \\
        [--n-patients 10] [--seed 42] [--resample-hz 1000] [--out results.csv]

``--annotations`` is a CSV file or a directory readable by ``AnnotationsLoader``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from dataclasses import replace
from fractions import Fraction
from importlib import metadata
from pathlib import Path

import pandas as pd
import yaml
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atriakit.configs.annotations_loader_config import AnnotationsLoaderConfig
from atriakit.configs.feature_computation_config import FeatureComputationConfig
from atriakit.configs.pipeline_config import PipelineConfig
from atriakit.configs.segment_config import SegmentConfig
from atriakit.configs.signal_preprocessor_config import SignalPreprocessorConfig
from atriakit.io import AnnotationsLoader, ECGLoader
from atriakit.models.annotation_schema import AnnotationSchema
from atriakit.models.annotations import Annotations
from atriakit.models.ecg_data import ECGData
from atriakit.preprocessing.signals import SignalPreprocessor
from atriakit.processing.dataset import ECGDataset
from atriakit.processing.pipeline import Pipeline

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "examples" / "atriakit_manuscript_config.yaml"
)

_SAMPLE_INDEX_COLUMNS = (
    AnnotationSchema.ONSET,
    AnnotationSchema.OFFSET,
    AnnotationSchema.ONSET_ORIGINAL,
    AnnotationSchema.OFFSET_ORIGINAL,
    AnnotationSchema.ONSET_MULTILEAD,
    AnnotationSchema.OFFSET_MULTILEAD,
    AnnotationSchema.QRS_ONSET,
    AnnotationSchema.QRS_ONSET_ORIGINAL,
)


def _resample_ecg_data(ecg_data: ECGData, target_fs: int) -> ECGData:
    """Resample the signal to ``target_fs`` Hz with a polyphase filter."""
    original_fs = ecg_data.get_sampling_frequency()
    ratio = Fraction(target_fs, int(original_fs)).limit_denominator(10_000)
    resampled = resample_poly(ecg_data.ecg, ratio.numerator, ratio.denominator, axis=1)
    return ECGData(
        ecg=resampled,
        fs=target_fs,
        lead_to_index=ecg_data.get_lead_to_index(),
        content_date=ecg_data.get_content_date(),
        content_time=ecg_data.get_content_start_time(),
    )


def _rescale_annotations(annotations: Annotations, ratio: float) -> Annotations:
    """Return a copy of ``annotations`` with sample indices scaled by ``ratio``."""
    df = annotations.get_df(copy=True)
    for col in _SAMPLE_INDEX_COLUMNS:
        if col in df.columns:
            df[col] = (df[col] * ratio).round().astype(int)
    return Annotations(df)


def _load_config(config_path: Path) -> tuple[AnnotationsLoaderConfig, PipelineConfig]:
    """Read the "annotations:" and "pipeline:" sections of a YAML config."""
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    loader_config = AnnotationsLoaderConfig(**(raw.get("annotations") or {}))
    section = dict(raw.get("pipeline") or {})
    for key, cls in (
        ("feature_computation", FeatureComputationConfig),
        ("segment_config", SegmentConfig),
        ("signal_preprocessor_config", SignalPreprocessorConfig),
        ("morphology_preprocessor_config", SignalPreprocessorConfig),
    ):
        if key in section and isinstance(section[key], dict):
            section[key] = cls(**section[key])
    return loader_config, PipelineConfig(**section)


def _prepare_normalization(
    pipeline_config: PipelineConfig,
    ecg_root: Path,
    annotations,
    loader: ECGLoader,
) -> PipelineConfig:
    """Fill in dataset-level mean/std for any preprocessor that normalizes."""
    dataset = ECGDataset(dataset_dir=str(ecg_root), loader=loader)

    signal_preprocessor = SignalPreprocessor(pipeline_config.signal_preprocessor_config)
    if signal_preprocessor.uses_normalization():
        mean, std = dataset.calculate_mean_std_p_waves(
            annotations, preprocessor=signal_preprocessor
        )
        pipeline_config = replace(
            pipeline_config,
            signal_preprocessor_config=replace(
                pipeline_config.signal_preprocessor_config, mean=mean, std=std
            ),
        )

    morphology_preprocessor = SignalPreprocessor(
        pipeline_config.morphology_preprocessor_config
    )
    if morphology_preprocessor.uses_normalization():
        mean, std = dataset.calculate_mean_std_p_waves(
            annotations, preprocessor=morphology_preprocessor
        )
        pipeline_config = replace(
            pipeline_config,
            morphology_preprocessor_config=replace(
                pipeline_config.morphology_preprocessor_config, mean=mean, std=std
            ),
        )

    return pipeline_config


def benchmark(
    annotations_path: str,
    ecg_root: str,
    config_path: str,
    n_patients: int | None,
    seed: int,
    resample_hz: int | None = None,
) -> pd.DataFrame:
    """Run the pipeline per patient and return one row of timings per patient."""
    loader_config, pipeline_config = _load_config(Path(config_path))
    all_annotations = AnnotationsLoader(config=loader_config).load(annotations_path)
    patient_ids = sorted(all_annotations[AnnotationSchema.PATIENT_ID].unique())

    rng = random.Random(seed)
    rng.shuffle(patient_ids)
    if n_patients is not None and n_patients < len(patient_ids):
        patient_ids = patient_ids[:n_patients]

    loader = ECGLoader()
    pipeline_config = _prepare_normalization(
        pipeline_config, Path(ecg_root), all_annotations, loader
    )

    pipeline = Pipeline(
        ecg_base_path=ecg_root, pipeline_config=pipeline_config, loader=loader
    )

    # Annotations must be rescaled before pipeline.run(), so read the native rate
    # from the first file and check that every later file has the same one.
    native_fs: float | None = None
    resample_ratio: float | None = None
    if resample_hz is not None:
        probe_file = all_annotations[AnnotationSchema.FILE_PATH].iloc[0]
        probe_ecg = loader.load(str(Path(ecg_root) / probe_file))
        native_fs = probe_ecg.get_sampling_frequency()
        resample_ratio = resample_hz / native_fs

    rows = []
    print(f"Benchmarking {len(patient_ids)} patient(s), processed sequentially...\n")
    for position, pid in enumerate(patient_ids, start=1):
        patient_annotations = all_annotations[
            all_annotations[AnnotationSchema.PATIENT_ID] == pid
        ]
        n_files = patient_annotations[AnnotationSchema.FILE_PATH].nunique()
        n_p_waves = len(patient_annotations)

        load_times: list[float] = []
        resample_times: list[float] = []
        orig_load_ecg = pipeline._load_ecg

        def _timed_load(
            file_path,
            _orig=orig_load_ecg,
            _load_acc=load_times,
            _resample_acc=resample_times,
        ):
            t0 = time.perf_counter()
            ecg = _orig(file_path)
            _load_acc.append(time.perf_counter() - t0)

            if resample_hz is None:
                return ecg

            observed_fs = ecg.get_sampling_frequency()
            if observed_fs != native_fs:
                raise ValueError(
                    f"File has sampling rate {observed_fs}Hz, expected {native_fs}Hz "
                    "(mixed sampling rates in one run are not supported by --resample-hz)."
                )

            t1 = time.perf_counter()
            resampled = _resample_ecg_data(ecg, resample_hz)
            _resample_acc.append(time.perf_counter() - t1)
            return resampled

        pipeline._load_ecg = _timed_load
        try:
            if resample_ratio is not None:
                patient_annotations = _rescale_annotations(
                    patient_annotations, resample_ratio
                )

            t_start = time.perf_counter()
            pipeline.run(patient_annotations, cache_dir=None)
        except Exception as exc:
            print(
                f"  patient {position}/{len(patient_ids)}: ERROR ({type(exc).__name__})"
            )
            continue
        finally:
            pipeline._load_ecg = orig_load_ecg

        # Resampling is a benchmark-only step, so it is not part of the timings.
        # We do it to have a more common 1kHz signal for the evaluation
        wall_s = time.perf_counter() - t_start - sum(resample_times)
        load_s = sum(load_times)
        features_s = wall_s - load_s

        rows.append(
            {
                "patient": position,
                "n_files": n_files,
                "n_p_waves": n_p_waves,
                "wall_s": round(wall_s, 2),
                "load_s": round(load_s, 2),
                "features_s": round(features_s, 2),
            }
        )
        print(
            f"  patient {position}/{len(patient_ids)}: wall={wall_s:.2f}s "
            f"load={load_s:.2f}s features={features_s:.2f}s "
            f"(files={n_files}, p_waves={n_p_waves})"
        )

    return pd.DataFrame(rows)


def hardware_info() -> dict:
    """Collect machine and library versions to report next to the timings."""
    model = platform.processor()
    physical_cores = set()
    try:
        physical_id = None
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                key, _, value = line.partition(":")
                key, value = key.strip(), value.strip()
                if key == "model name":
                    model = value
                elif key == "physical id":
                    physical_id = value
                elif key == "core id":
                    physical_cores.add((physical_id, value))
    except OSError:
        pass  # non-Linux: fall back to platform.processor()

    try:
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (ValueError, OSError, AttributeError):
        ram_gb = None

    def _version(package: str) -> str | None:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return None

    return {
        "platform": platform.platform(),
        "cpu_model": model or None,
        "physical_cores": len(physical_cores) or None,
        "logical_cpus": os.cpu_count(),
        "usable_cpus": (
            len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        ),
        "ram_gb": None if ram_gb is None else round(ram_gb, 1),
        "python": sys.version.split()[0],
        "atriakit": _version("atriakit"),
        "numpy": _version("numpy"),
        "scipy": _version("scipy"),
        "pandas": _version("pandas"),
        "thread_env": {
            name: os.environ[name]
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            if name in os.environ
        },
    }


def print_hardware(info: dict) -> None:
    """Print the hardware/software info, one field per line."""
    print("=== Hardware / software ===")
    for key, value in info.items():
        print(f"{key:<15}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to an annotation CSV file or directory",
    )
    parser.add_argument(
        "--ecg-root", required=True, help="Base directory containing the ECG files"
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Pipeline YAML config (default: the manuscript config)",
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=None,
        help="Number of patients to sample (default: all)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for patient sampling/order"
    )
    parser.add_argument(
        "--out", default=None, help="Where to save the aggregate results CSV (optional)"
    )
    parser.add_argument(
        "--resample-hz",
        type=int,
        default=None,
        help="Resample to this rate (Hz) before the pipeline; not included in the timings",
    )
    args = parser.parse_args()

    hardware = hardware_info()
    print_hardware(hardware)
    print()

    results = benchmark(
        annotations_path=args.annotations,
        ecg_root=args.ecg_root,
        config_path=args.config,
        n_patients=args.n_patients,
        seed=args.seed,
        resample_hz=args.resample_hz,
    )

    if results.empty:
        print("\nNo patients were successfully processed.")
        return

    print()
    print("=== Summary (aggregate only; no patient identifiers) ===")
    summary = (
        results[["wall_s", "load_s", "features_s"]]
        .describe()
        .loc[["count", "mean", "std", "min", "50%", "max"]]
    )
    print(summary.to_string(float_format=lambda x: f"{x:.2f}"))
    print()
    print(f"Median per-patient wall time : {results['wall_s'].median():.2f}s")
    print(f"Mean per-patient wall time   : {results['wall_s'].mean():.2f}s")
    print(f"Total wall time ({len(results)} patients) : {results['wall_s'].sum():.2f}s")
    print(
        f"Total features time ({len(results)} patients) : {results['features_s'].sum():.2f}s"
    )

    if len(results) > 1:
        print()
        for label, col in (("Wall", "wall_s"), ("Features", "features_s")):
            per_file = results[col] / results["n_files"]
            per_p_wave = results[col] / results["n_p_waves"]
            print(
                f"{label} time per file    : mean={per_file.mean():.2f}s  std={per_file.std():.2f}s  "
                f"(r={results[col].corr(results['n_files']):.2f})"
            )
            print(
                f"{label} time per P-wave  : mean={per_p_wave.mean():.2f}s  std={per_p_wave.std():.2f}s  "
                f"(r={results[col].corr(results['n_p_waves']):.2f})"
            )

    print()
    print_hardware(hardware)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_path, index=False)
        hardware_path = out_path.with_suffix(".hardware.json")
        hardware_path.write_text(json.dumps(hardware, indent=2), encoding="utf-8")
        print(f"\nAggregate results saved to {out_path}")
        print(f"Hardware info saved to {hardware_path}")


if __name__ == "__main__":
    main()
