import contextlib
import functools
import inspect
import warnings
from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd

import atriakit.features.amplitude as amplitude_features
from atriakit.models.annotations import Annotations
from atriakit.configs.feature_calculator_config import FeatureCalculatorConfig
from atriakit.configs.signal_preprocessor_config import (
    default_morphology_preprocessor_config,
    default_signal_preprocessor_config,
)
from atriakit.constants import AXIS_LEADS, VCG_FEATURE_COLUMNS, VCG_LEADS
from atriakit.features.area import area, ptf, ptf_auto, ptf_segment_selector
from atriakit.features.complexity import extrema_counter
from atriakit.features.entropy import sample_entropy_nolds, shannon_entropy
from atriakit.features.fragments import (
    fragment_amplitude,
    fragment_count,
    fragment_width,
    fragments_finder,
)
from atriakit.features.morphology import morphology_classifier
from atriakit.features.vcg import (
    vcg_area_calculator,
    vcg_axis_angles,
    vcg_eigenfeatures_calculator,
)
from atriakit.models.annotation_schema import AnnotationSchema
from atriakit.models.ecg_data import ECGData
from atriakit.preprocessing.signals import SignalPreprocessor
from atriakit.processing.segment_processor import SegmentProcessor

FRAGMENT_NAN_VALUE = (np.nan, np.nan, np.nan, np.nan)


def _no_baseline(method):
    """Opt-out marker: prevents _auto_wrap_baseline from adding a global-baseline context."""
    method._skip_baseline = True
    return method


def _auto_wrap_baseline(cls):
    """Class decorator that adds type checking and automatic spline baseline correction to public annotation methods."""
    for attr_name, obj in list(vars(cls).items()):
        if attr_name.startswith("_") or not callable(obj):
            continue
        skip_baseline = getattr(obj, "_skip_baseline", False)
        if skip_baseline:

            @functools.wraps(obj)
            def wrapper(self, annotations, *args, _m=obj, **kwargs):
                if not isinstance(annotations, Annotations):
                    raise TypeError(
                        f"Expected Annotations, got {type(annotations).__name__}. "
                        "Use AnnotationsLoader to create Annotations from a DataFrame."
                    )
                return _m(self, annotations, *args, **kwargs)

        else:

            @functools.wraps(obj)
            def wrapper(self, annotations, *args, _m=obj, **kwargs):
                if not isinstance(annotations, Annotations):
                    raise TypeError(
                        f"Expected Annotations, got {type(annotations).__name__}. "
                        "Use AnnotationsLoader to create Annotations from a DataFrame."
                    )
                with self._global_baseline(annotations):
                    return _m(self, annotations, *args, **kwargs)

        setattr(cls, attr_name, wrapper)
    return cls


@_auto_wrap_baseline
class FeatureCalculators:
    """Computes P-wave features from ECG signals and beat annotations.

    Baseline correction is applied automatically on each call if enabled
    in the config (on by default).

    All feature methods return NaN for annotations where the segment is empty.

    Attributes:
        signal_preprocessor: Preprocessor for amplitude and duration features.
        morphology_preprocessor: Preprocessor for shape-based features; uses a
            narrower bandpass to capture the general P-wave shape.
        noise_estimation_window_ms: Pre-onset window (ms) used to estimate per-lead baseline noise.
        segment_processor: Extracts annotation-bounded, baseline-corrected
            segments and computes metrics over them; see ``SegmentConfig``.
    """

    def __init__(
        self,
        config: FeatureCalculatorConfig = None,
        signal_preprocessor: SignalPreprocessor | None = None,
        morphology_preprocessor: SignalPreprocessor | None = None,
    ):
        if config is None:
            config = FeatureCalculatorConfig()

        self.signal_preprocessor = signal_preprocessor or SignalPreprocessor(
            default_signal_preprocessor_config()
        )
        self.morphology_preprocessor = morphology_preprocessor or SignalPreprocessor(
            default_morphology_preprocessor_config()
        )
        self.noise_estimation_window_ms = config.noise_estimation_window_ms
        self.segment_processor = SegmentProcessor(config.segment_config)

    @staticmethod
    def _identity_segment(segment: np.ndarray, _row) -> np.ndarray:
        return segment

    @staticmethod
    def _file_label(annotations: Annotations) -> str:
        if not annotations.empty:
            return str(annotations.iloc[0].file_path)
        return "<unknown>"

    def _get_signal_for_lead(self, ecg_data: ECGData, lead: str) -> np.ndarray:
        return ecg_data.get_lead_signal(lead, preprocessor=self.signal_preprocessor)

    @contextlib.contextmanager
    def _global_baseline(self, annotations: Annotations):
        """Context manager that sets spline baseline onsets on both preprocessors. Reentrant-safe: nested calls are no-ops."""
        if self.signal_preprocessor.has_baseline_onsets():
            yield
            return
        self.signal_preprocessor.set_baseline_onsets(annotations)
        self.morphology_preprocessor.set_baseline_onsets(annotations)
        try:
            yield
        finally:
            self.signal_preprocessor.clear_baseline_onsets()
            self.morphology_preprocessor.clear_baseline_onsets()

    def _get_morphology_signal_for_lead(
        self, ecg_data: ECGData, lead: str
    ) -> np.ndarray:
        return ecg_data.get_lead_signal(lead, preprocessor=self.morphology_preprocessor)

    def _compute_segment_metric(
        self,
        annotations: Annotations,
        ecg_data: ECGData,
        metric_func: Callable[[np.ndarray, object], object],
        get_signal: Callable | None = None,
        segment_selector: Callable[[np.ndarray, object], np.ndarray] | None = None,
        nan_value=np.nan,
    ) -> list:
        """Iterate annotations and apply a metric function to each extracted signal segment.

        Args:
            annotations: Beat annotations with onset/offset columns.
            ecg_data: ECG signal source.
            metric_func: Callable(segment, row) -> scalar applied to each segment.
            get_signal: Optional callable(row) -> signal array; defaults to the signal preprocessor for the row's lead.
            segment_selector: Optional callable(segment, row) -> sub-segment; defaults to the full segment.
            nan_value: Value to use when a segment cannot be extracted.
        """
        if get_signal is None:
            get_signal = lambda row: self._get_signal_for_lead(ecg_data, row.lead)

        return self.segment_processor.compute_segment_metric(
            annotations,
            ecg_data,
            metric_func,
            get_signal=get_signal,
            segment_selector=segment_selector,
            nan_value=nan_value,
        )

    @_no_baseline
    def duration(self, annotations: Annotations, fs: int) -> np.ndarray:
        """P-wave duration in seconds for each annotation: (offset - onset) / fs."""
        if annotations.empty:
            return np.array([])

        durations = (
            annotations[AnnotationSchema.OFFSET] - annotations[AnnotationSchema.ONSET]
        ) / fs
        return np.array(durations)

    def ptf(self, annotations: Annotations, ecg_data: ECGData) -> list:
        """Compute P-wave terminal force (PTF) for each annotation.

        If ``p_wave_morphology`` and ``inflection_point`` columns are present,
        uses the annotated inflection point and skips monophasic/complex waves
        (supervised mode). Otherwise detects the inflection automatically from the heavily
        low-pass filtered signal and computes PTF for all waves and leads
        regardless of morphology (unsupervised mode).

        Returns:
            PTF value per annotation (NaN for skipped or invalid segments).

        References:
            Sclarovsky, S., et al. (2017). P-wave terminal force analysis in ECG.
            https://www.sciencedirect.com/science/article/pii/S2405500X17311866
        """
        fs = ecg_data.get_sampling_frequency()
        supervised = (
            "p_wave_morphology" in annotations.columns
            and "inflection_point" in annotations.columns
        )

        if supervised:

            def ptf_wrapper(segment, row):
                return ptf(segment, fs)

            return self._compute_segment_metric(
                annotations,
                ecg_data,
                ptf_wrapper,
                segment_selector=lambda segment, row: ptf_segment_selector(
                    segment, row, row.onset
                ),
            )

        def ptf_auto_metric(segment, row):
            signal_morph = self._get_morphology_signal_for_lead(ecg_data, row.lead)
            seg_morph = self.segment_processor.extract_segment(
                signal_morph, self._identity_segment, fs, row
            )
            return ptf_auto(segment, fs, lead=row.lead, seg_morph=seg_morph)

        return self._compute_segment_metric(annotations, ecg_data, ptf_auto_metric)

    def area_to_duration_ratio(
        self, annotations: Annotations, ecg_data: ECGData
    ) -> list:
        """Compute P-wave area-to-duration ratio for each annotation.

        Returns:
            Area-to-duration ratio per annotation.

        Raises:
            ValueError: If any P-wave duration is zero.

        References:
            Rasmussen et al. (2020). Ann. Noninvasive Electrocardiol., 25(5):e12751.
        """
        areas = self.area(annotations, ecg_data)
        durations = self.duration(annotations, ecg_data.get_sampling_frequency())

        if np.any(durations == 0):
            raise ValueError(
                "Duration contains zero values, cannot compute area to duration ratio."
            )

        ratios = areas / durations
        return ratios

    def area(self, annotations: Annotations, ecg_data: ECGData) -> list:
        """Compute the absolute integral of each annotated P-wave segment."""
        fs = ecg_data.get_sampling_frequency()
        return self._compute_segment_metric(
            annotations,
            ecg_data,
            lambda segment, _: area(segment, fs),
        )

    def max_amplitude(self, annotations: Annotations, ecg_data: ECGData) -> list:
        """Compute the maximum amplitude of each P-wave segment."""
        return self._compute_segment_metric(
            annotations, ecg_data, amplitude_features.max_amplitude
        )

    def min_amplitude(self, annotations: Annotations, ecg_data: ECGData) -> list:
        """Compute the minimum amplitude of each P-wave segment."""
        return self._compute_segment_metric(
            annotations, ecg_data, amplitude_features.min_amplitude
        )

    def max_absolute_amplitude(
        self, annotations: Annotations, ecg_data: ECGData
    ) -> list:
        """Compute the maximum absolute amplitude of each P-wave segment."""
        return self._compute_segment_metric(
            annotations, ecg_data, amplitude_features.max_absolute_amplitude
        )

    def peak_to_peak_amplitude(
        self, annotations: Annotations, ecg_data: ECGData
    ) -> list:
        """Compute the peak-to-peak amplitude of each P-wave segment."""
        return self._compute_segment_metric(
            annotations, ecg_data, amplitude_features.peak_to_peak_amplitude
        )

    @_no_baseline
    def dispersion(self, annotations: Annotations, fs: int) -> float:
        """Compute the dispersion of P-wave durations (max - min) in seconds."""
        if annotations.empty:
            return np.nan

        durations = self.duration(annotations, fs)
        return np.max(durations) - np.min(durations)

    def axis(self, annotations: Annotations, ecg_data: ECGData) -> list:
        """Compute the frontal P-wave axis in degrees for each annotated P-wave.

        The axis is derived from leads I and aVF:
            axis = arctan2(2 * sum(segment_aVF), sqrt(3) * sum(segment_I))

        Returns:
            Axis value in degrees per P-wave (NaN for P-waves with missing or invalid segments).

        Raises:
            ValueError: If annotations contain no entries for lead I or aVF.
        """
        if annotations.empty:
            return []

        i_lead, avf_lead = AXIS_LEADS
        aVF_lead_anns = annotations.filter_by_lead(avf_lead)
        I_lead_anns = annotations.filter_by_lead(i_lead)

        if len(aVF_lead_anns) == 0 or len(I_lead_anns) == 0:
            raise ValueError(
                "Axis calculation failed due to missing annotations in one of the "
                f"leads. Annotation file {self._file_label(annotations)}"
            )

        lead_aVF = self._get_signal_for_lead(ecg_data, avf_lead)
        lead_I = self._get_signal_for_lead(ecg_data, i_lead)
        axes = []
        fs = ecg_data.get_sampling_frequency()
        for p_wave_id in sorted(pd.unique(annotations[AnnotationSchema.P_WAVE_ID])):
            ann_aVF = aVF_lead_anns[
                aVF_lead_anns[AnnotationSchema.P_WAVE_ID] == p_wave_id
            ]
            ann_I = I_lead_anns[I_lead_anns[AnnotationSchema.P_WAVE_ID] == p_wave_id]

            if ann_aVF.empty or ann_I.empty:
                warnings.warn(
                    f"Skipping P-wave ID {p_wave_id} in file '{self._file_label(annotations)}': "
                    f"missing annotations in one or both leads (I lead: {len(ann_I)} entries, "
                    f"aVF lead: {len(ann_aVF)} entries). Result will be set to -1.",
                    stacklevel=2,
                )
                axes.append(np.nan)
                continue

            if len(ann_aVF) > 1 or len(ann_I) > 1:
                warnings.warn(
                    f"Multiple annotations found for p_wave_id={p_wave_id}. Using the first entry."
                )

            try:
                segment_aVF = self.segment_processor.extract_segment(
                    lead_aVF, lambda seg, row: seg, fs, ann_aVF.iloc[0]
                )
                segment_I = self.segment_processor.extract_segment(
                    lead_I, lambda seg, row: seg, fs, ann_I.iloc[0]
                )
            except ValueError:
                warnings.warn(
                    f"Skipping P-wave {p_wave_id} in file '{self._file_label(annotations)}': "
                    "invalid onset/offset values detected. Result will be set to np.nan.",
                    stacklevel=2,
                )
                axes.append(np.nan)
                continue

            peak_aVF = np.sum(segment_aVF)
            peak_I = np.sum(segment_I)

            angle = np.degrees(np.arctan2(2 * peak_aVF, np.sqrt(3) * peak_I))
            axes.append(angle)

        return axes

    def complexity(
        self,
        annotations: Annotations,
        ecg_data: ECGData,
        threshold_multiplier: float = 0.1,
    ) -> list:
        """Compute the number of significant peaks in each annotated P-wave.

        A peak is significant if it differs from both neighboring extrema by at
        least ``threshold_multiplier`` times the per-lead maximum P-wave amplitude.

        Args:
            annotations: P-wave annotations.
            ecg_data: ECG signal source.
            threshold_multiplier: Fraction of per-lead maximum amplitude used as
                the significance threshold (default 0.1).

        Returns:
            Number of significant peaks per annotation.

        References:
            https://www.sciencedirect.com/science/article/pii/S0022073616300309
        """
        max_abs_amplitude = self.max_absolute_amplitude(annotations, ecg_data)
        if len(max_abs_amplitude) == 0:
            return np.array([])

        max_abs_amplitude_per_lead = (
            pd.Series(max_abs_amplitude, index=annotations[AnnotationSchema.LEAD])
            .groupby(level=0)
            .quantile(0.95)
        )

        def extrema_counter_wrapper(segment, row):
            lead = row.lead
            return extrema_counter(
                segment,
                lead,
                max_abs_amplitude_per_lead,
                threshold_multiplier=threshold_multiplier,
            )

        return self._compute_segment_metric(
            annotations, ecg_data, extrema_counter_wrapper
        )

    def _fragments_wrapper(
        self,
        ecg_data: ECGData,
        noise,
        min_fragment_length_ms,
        fragment_noise_multiplier,
    ):
        lead_to_index = ecg_data.get_lead_to_index()

        def wrapper(segment, row):
            # VCG is a synthetic single-channel signal — use index 0
            noise_idx = 0 if row.lead == "VCG" else lead_to_index[row.lead]
            return fragments_finder(
                segment,
                noise[noise_idx],
                ecg_data.get_sampling_frequency(),
                min_fragment_length_ms,
                fragment_noise_multiplier,
            )

        return wrapper

    def fragments(
        self,
        annotations: Annotations,
        ecg_data: ECGData,
        min_fragment_length_ms: float = 0,
        noise_sd_multiplier: float = 3.0,
        fragment_noise_multiplier: float = 3.0,
    ) -> list:
        """Compute P-wave fragmentation metrics for each annotation.

        Fragmentation quantifies discontinuous electrical conduction in the P-wave,
        associated with atrial fat infiltration and AF risk.

        A fragment is a sub-segment between two local extrema of the P-wave whose
        amplitude change exceeds the noise threshold.

        Args:
            annotations: P-wave annotations.
            ecg_data: ECG signal source.
            min_fragment_length_ms: Minimum fragment length in ms to consider (default 0).
            noise_sd_multiplier: Multiplier for filtering out pre-onset noise estimation
                windows that are too large relative to P-wave amplitude (default 3.0).
            fragment_noise_multiplier: A fragment's amplitude change must exceed this
                multiple of the noise level to count as a valid fragment (default 3.0).

        Returns:
            Fragment metrics per annotation.

        References:
            Murthy et al. (2014). Number of P-wave fragmentations as a marker of atrial fat.
        """
        noise = self.estimate_noise(
            annotations, ecg_data, sd_threshold=noise_sd_multiplier
        )
        wrapper = self._fragments_wrapper(
            ecg_data, noise, min_fragment_length_ms, fragment_noise_multiplier
        )
        return self._compute_segment_metric(
            annotations, ecg_data, wrapper, nan_value=[FRAGMENT_NAN_VALUE]
        )

    @_no_baseline
    def _vcg_fragments_from_signal(
        self,
        annotations: Annotations,
        ecg_data: ECGData,
        vcg_sum: np.ndarray,
        min_fragment_length_ms: float = 0,
        noise_sd_multiplier: float = 3.0,
        fragment_noise_multiplier: float = 3.0,
    ) -> list:
        """Compute raw fragment metrics from a pre-built VCG signal array."""
        noise = self.estimate_noise(
            annotations, ecg_data, vcg_sum, sd_threshold=noise_sd_multiplier
        )
        wrapper = self._fragments_wrapper(
            ecg_data, noise, min_fragment_length_ms, fragment_noise_multiplier
        )
        return self._compute_segment_metric(
            annotations,
            ecg_data,
            wrapper,
            get_signal=lambda row: vcg_sum,
            nan_value=[FRAGMENT_NAN_VALUE],
        )

    @_no_baseline
    def atrial_rate(self, annotations: Annotations, ecg_data: ECGData) -> int:
        """Compute the median atrial rate in bpm from P-wave annotations.

        PP intervals are derived from the minimum onset across leads per beat.

        Returns:
            Median atrial rate in bpm.
        """
        fs = ecg_data.get_sampling_frequency()

        grouped = annotations.groupby(AnnotationSchema.P_WAVE_ID)[
            AnnotationSchema.ONSET
        ]

        consensus_onsets = grouped.min().sort_index().values

        pp_intervals = np.diff(consensus_onsets) / fs * 1000  # in ms
        pp_intervals = pp_intervals[
            pp_intervals > 0
        ]  # remove zeros. Shouldn't happen but just in case

        if len(pp_intervals) == 0:
            return 0  # or np.nan if you prefer

        atrial_rate_values = 60000 / pp_intervals  # bpm
        return int(np.median(atrial_rate_values))

    def _r_peaks_for_lead(
        self,
        lead_signal: np.ndarray,
        lead_annotations: Annotations,
        window_samples: int,
    ) -> list[int]:
        r_peaks = []
        for idx, ann in enumerate(lead_annotations.itertuples()):
            offset_orig = ann.offset_original
            start_idx = int(offset_orig if not np.isnan(offset_orig) else ann.offset)
            end_idx = min(start_idx + window_samples, len(lead_signal))
            if idx != len(lead_annotations) - 1:
                next_orig = lead_annotations.iloc[idx + 1].onset_original
                next_onset = int(
                    next_orig
                    if not np.isnan(next_orig)
                    else lead_annotations.iloc[idx + 1].onset
                )
                end_idx = min(end_idx, next_onset)

            segment = lead_signal[start_idx:end_idx]
            if segment.size == 0:
                raise ValueError(
                    f"Segment with range [{start_idx}:{end_idx}] and lead signal length {len(lead_signal)} is empty"
                )

            # Update to reduce baseline drift effect
            segment = segment - segment[0]

            r_peaks.append(np.argmax(np.abs(segment)) + start_idx)
        return r_peaks

    def find_r_peaks_median(
        self,
        annotations: Annotations,
        ecg_data: ECGData,
        leads: list = None,
        max_r_peak_window_ms: float = 350,
    ) -> np.ndarray:
        """Detect R-peaks using a window after each P-wave, returning the median index across leads per beat."""
        if annotations.empty:
            return np.array([])

        if leads is None:
            leads = annotations[AnnotationSchema.LEAD].unique()

        fs = ecg_data.get_sampling_frequency()
        max_window_samples = int(max_r_peak_window_ms * fs / 1000)

        per_lead_r_peaks = {}
        for lead in leads:
            try:
                lead_signal = self._get_signal_for_lead(ecg_data, lead)
                lead_annotations = annotations.filter_by_lead(lead)
                r_peaks = self._r_peaks_for_lead(
                    lead_signal, lead_annotations, max_window_samples
                )
                if r_peaks:
                    per_lead_r_peaks[lead] = r_peaks
            except Exception as e:
                warnings.warn(f"Failed to compute R-peaks for lead {lead}: {e}")

        if not per_lead_r_peaks:
            return np.array([])

        # Truncate all leads to the shortest detected peak list so they can be stacked for median.
        # Leads may detect different counts if a beat is missed on one lead.
        n_beats = min(len(peaks) for peaks in per_lead_r_peaks.values())
        if n_beats == 0:
            return np.array([])

        r_peaks_array = np.array(
            [peaks[:n_beats] for peaks in per_lead_r_peaks.values()]
        )
        return np.median(r_peaks_array, axis=0).astype(int)

    def heart_rate(
        self,
        annotations: Annotations,
        ecg_data: ECGData,
        leads: list[str] | None = None,
    ) -> int:
        """Compute median heart rate in bpm from P-wave annotations.

        R-peaks are detected using a consensus across the specified leads
        (median position), then RR intervals are converted to bpm.

        Args:
            annotations: P-wave annotations used to locate beats.
            ecg_data: ECG signal source.
            leads: Leads used for R-peak detection (default: ["II", "V5", "V6"]).

        Returns:
            Median heart rate in bpm, -1 if fewer than two beats are available,
            or an empty array if ``annotations`` is empty.
        """
        if annotations.empty:
            return np.array([])

        if leads is None:
            leads = ["II", "V5", "V6"]

        # Detect a single global R-peak per beat (median across leads)
        r_peaks = self.find_r_peaks_median(annotations, ecg_data, leads=leads)

        if len(r_peaks) < 2:
            # Not enough beats to compute RR intervals
            return -1

        # Compute RR intervals in milliseconds
        rr_intervals_ms = np.diff(r_peaks) / ecg_data.get_sampling_frequency() * 1000

        # Compute heart rate per interval and return median across all intervals
        heart_rate_bpm = 60000 / rr_intervals_ms
        return int(np.nanmedian(heart_rate_bpm))

    def classify_p_wave_morphology(
        self,
        annotations: Annotations,
        ecg_data: ECGData,
        *,
        morphology_min_phase_fraction: float = 0.1,
        morphology_noise_sd_multiplier: float = 3.0,
        noise_sd_multiplier: float = 3.0,
    ) -> tuple[list, list]:
        """Classify P-wave morphology and detect inflection points for each annotation.

        Categories include monophasic positive/negative, biphasic positive–negative/
        negative–positive, flat, and complex. Classification is rule-based (derivative
        sign patterns) and sensitive to noise — results may benefit from manual review
        or parameter tuning on new datasets.

        Noise is estimated as the average standard deviation of pre-onset windows;
        windows where the standard deviation is too large relative to the P-wave
        amplitude (e.g. due to T-wave overlap) are excluded from the average.

        Args:
            annotations: P-wave annotations.
            ecg_data: ECG signal source.
            morphology_min_phase_fraction: Derivative sign-change regions shorter than
                this fraction of the P-wave are merged into the dominant region; increase
                to reduce sensitivity to small deflections (default 0.1).
            morphology_noise_sd_multiplier: A sign-change region must produce an amplitude
                change larger than this multiple of the noise level to count as a distinct
                morphology phase (default 3.0).
            noise_sd_multiplier: Multiplier used when filtering out pre-onset windows
                where the standard deviation is too large relative to P-wave amplitude,
                e.g. due to T-wave overlap (default 3.0).

        Returns:
            Tuple of (morphology labels, inflection point indices) per annotation.
            Failed classifications return ("N/A", -1).
        """
        if annotations.empty:
            return np.array([]), np.array([])

        noise = self.estimate_noise(
            annotations, ecg_data, sd_threshold=noise_sd_multiplier
        )

        lead_to_index = ecg_data.get_lead_to_index()

        def morphology_wrapper(segment, row):
            lead_noise = noise[lead_to_index[row.lead]]
            return morphology_classifier(
                segment,
                row.onset,
                lead_noise,
                morphology_min_phase_fraction,
                morphology_noise_sd_multiplier,
            )

        tuples = self._compute_segment_metric(
            annotations,
            ecg_data,
            morphology_wrapper,
            get_signal=lambda row: self._get_morphology_signal_for_lead(
                ecg_data, row.lead
            ),
            nan_value=("N/A", -1),
        )
        phase_types, inflection_points = zip(*tuples)
        inflection_points = [int(ip) for ip in inflection_points]
        return phase_types, inflection_points

    def fragment_metrics(
        self,
        annotations: Annotations,
        ecg_data: ECGData,
        normalize_by_duration: bool = False,
        min_fragment_length_ms: float = 0.0,
        noise_sd_multiplier: float = 3.0,
        fragment_noise_multiplier: float = 3.0,
    ) -> tuple[list, list, list]:
        """Compute fragment count, width, and amplitude for each annotated P-wave.

        A fragment is a sub-segment between two local extrema whose amplitude change
        exceeds the noise threshold. Three metrics are returned: count, average width,
        and average amplitude of fragments per P-wave.

        Args:
            annotations: P-wave annotations.
            ecg_data: ECG signal source.
            normalize_by_duration: If True, normalize each metric per 100 ms of
                P-wave duration, as in the original fragmentation paper (default False).
            min_fragment_length_ms: Fragments shorter than this duration in ms are
                discarded (default 0).
            noise_sd_multiplier: Multiplier for filtering out pre-onset noise estimation
                windows that are too large relative to P-wave amplitude (default 3.0).
            fragment_noise_multiplier: A fragment's amplitude change must exceed this
                multiple of the noise level to count as a valid fragment (default 3.0).

        Returns:
            Tuple of (fragment count, fragment width, fragment amplitude) per annotation.

        References:
            Murthy et al. (2014). Number of P-wave fragmentations as a marker of atrial fat.
        """
        per_annotation_fragments = self.fragments(
            annotations,
            ecg_data,
            min_fragment_length_ms=min_fragment_length_ms,
            noise_sd_multiplier=noise_sd_multiplier,
            fragment_noise_multiplier=fragment_noise_multiplier,
        )

        raw_counts = np.asarray(fragment_count(per_annotation_fragments), dtype=float)
        raw_widths = np.asarray(
            fragment_width(per_annotation_fragments, ecg_data.get_sampling_frequency()),
            dtype=float,
        )
        raw_amplitudes = np.asarray(
            fragment_amplitude(per_annotation_fragments),
            dtype=float,
        )

        if not normalize_by_duration:
            return (
                list(raw_counts),
                list(raw_widths),
                list(raw_amplitudes),
            )

        durations = self.duration(annotations, ecg_data.get_sampling_frequency())
        normalization_factors = np.where(durations > 0, 0.1 / durations, np.nan)

        return (
            list(raw_counts * normalization_factors),
            list(raw_widths * normalization_factors),
            list(raw_amplitudes * normalization_factors),
        )

    def _compute_lead_noise(
        self, lead_annotations, lead_signal, window_samples, sd_threshold
    ):
        if lead_annotations.empty:
            return 0

        lead_noise_estimates = []
        for annotation in lead_annotations.itertuples():
            pwave_onset = int(annotation.onset)
            noise_segment_start = int(max(pwave_onset - window_samples, 0))
            noise_segment_end = int(pwave_onset)

            noise_segment = lead_signal[noise_segment_start:noise_segment_end]

            if len(noise_segment) == 0:
                continue

            noise_std = np.std(noise_segment)
            # only consider noise if it's not too small compared to the signal
            if sd_threshold * noise_std < annotation.p_wave_max:
                lead_noise_estimates.append(noise_std)

        return np.mean(lead_noise_estimates) if lead_noise_estimates else 0

    def estimate_noise(
        self,
        annotations: Annotations,
        ecg_data: ECGData,
        lead_signal: np.ndarray | None = None,
        window_in_ms: int | None = None,
        sd_threshold: int = 3,
    ) -> list:
        """Estimate baseline noise per lead as the average pre-onset standard deviation.

        For each beat, the standard deviation is computed in a window before the P-wave
        onset. Windows where the noise is too large relative to the P-wave amplitude
        (e.g. due to T-wave overlap) are excluded from the average.

        Args:
            annotations: P-wave annotations.
            ecg_data: ECG signal source.
            lead_signal: Precomputed signal array; pass for synthetic signals like VCG.
            window_in_ms: Pre-onset window length in ms; defaults to
                ``noise_estimation_window_ms`` from config.
            sd_threshold: Windows where ``sd_threshold * noise_std >= p_wave_max``
                are excluded (default 3).

        Returns:
            Noise estimate per lead, in the same order as the leads in annotations.

        Raises:
            ValueError: If a VCG lead is present in ``annotations`` but
                ``lead_signal`` is not provided.
        """
        if annotations.empty:
            return []

        annotations_with_p_wave_max = annotations.copy()
        if lead_signal is None:
            annotations_with_p_wave_max["p_wave_max"] = self.max_absolute_amplitude(
                annotations, ecg_data
            )
        else:
            annotations_with_p_wave_max["p_wave_max"] = self._compute_segment_metric(
                annotations,
                ecg_data,
                lambda x, _: np.max(np.abs(x)),
                get_signal=lambda row: lead_signal,
            )

        if window_in_ms is None:
            window_in_ms = self.noise_estimation_window_ms

        per_lead_noise_estimates = []
        sampling_rate = ecg_data.get_sampling_frequency()
        offset_window = window_in_ms * int(sampling_rate / 1000)
        leads_in_annotations = list(
            pd.unique(annotations_with_p_wave_max[AnnotationSchema.LEAD])
        )
        if "VCG" in leads_in_annotations and lead_signal is None:
            raise ValueError("Lead_signal must be provided for VCG leads.")

        if lead_signal is not None and leads_in_annotations == ["VCG"]:
            leads = ["VCG"]
        else:
            leads = ecg_data.get_leads()

        for lead in leads:
            lead_annotations = annotations_with_p_wave_max.filter_by_lead(lead)

            if lead != "VCG":
                lead_signal = self._get_signal_for_lead(ecg_data, lead)

            if lead_annotations.empty:
                per_lead_noise_estimates.append(0)
                continue

            lead_noise = self._compute_lead_noise(
                lead_annotations, lead_signal, offset_window, sd_threshold
            )
            per_lead_noise_estimates.append(lead_noise)

        return per_lead_noise_estimates

    def _get_vcg_signal(self, row=None, ecg_data: ECGData = None) -> np.ndarray:
        return ecg_data.get_vcg(preprocessor=self.signal_preprocessor)

    def vcg_area(self, annotations: Annotations, ecg_data: ECGData) -> list:
        """Compute the VCG area for each P-wave."""
        vcg_ann = annotations.vcg_annotations()

        return self._compute_segment_metric(
            vcg_ann,
            ecg_data,
            lambda vcg_segment, _: vcg_area_calculator(
                vcg_segment, ecg_data.get_sampling_frequency()
            ),
            get_signal=lambda row: self._get_vcg_signal(row, ecg_data),
        )

    def vcg_axis(
        self, annotations: Annotations, ecg_data: ECGData
    ) -> list[tuple[float, float]]:
        """Compute 3-D P-wave VCG axis (elevation, azimuth) in degrees for each P-wave."""
        vcg_ann = annotations.vcg_annotations()
        return self._compute_segment_metric(
            vcg_ann,
            ecg_data,
            lambda vcg_segment, _: vcg_axis_angles(vcg_segment),
            get_signal=lambda row: self._get_vcg_signal(row, ecg_data),
            nan_value=(np.nan, np.nan),
        )

    def vcg_eigenfeatures(
        self, annotations: Annotations, ecg_data: ECGData, return_eigenvectors=False
    ) -> tuple[list, list, list, list]:
        """Compute SVD-based shape features from the 3-D VCG loop of each P-wave.

        Uses SVD to extract principal components and derives:
        eigenvalues (variance along each axis), roundness (ratio of 2nd to 1st
        eigenvalue), and flatness (ratio of 3rd to sum of 1st and 2nd eigenvalues).

        Args:
            annotations: P-wave annotations.
            ecg_data: ECG signal source.
            return_eigenvectors: If True, also returns the principal directions
                as eigenvectors (default False).

        Returns:
            Tuple of (eigenvalues, roundness, flatness[, eigenvectors], p_wave_ids).

        References:
            Ortigosa et al. (2021). Characterization of Changes in P-Wave VCG Loops
            Following Pulmonary-Vein Isolation. Sensors, 21(5), 1923.
            https://doi.org/10.3390/s21051923
        """

        def vcg_eigenfeatures_wrapper(vcg_segment, row):
            eigenvalues, roundness, flatness, eigenevectors = (
                vcg_eigenfeatures_calculator(vcg_segment)
            )
            return (
                eigenvalues,
                roundness,
                flatness,
                eigenevectors,
                row.p_wave_id,
            )

        vcg_ann = annotations.vcg_annotations()
        metrics_list = self._compute_segment_metric(
            vcg_ann,
            ecg_data,
            vcg_eigenfeatures_wrapper,
            get_signal=lambda row: self._get_vcg_signal(row, ecg_data),
            nan_value=(
                np.array([np.nan, np.nan, np.nan]),
                np.nan,
                np.nan,
                None,
                np.nan,
            ),
        )

        (
            eigenvalues,
            roundness,
            flatness,
            eigenvectors,
            p_wave_ids,
        ) = zip(*metrics_list)

        if return_eigenvectors:
            return eigenvalues, roundness, flatness, eigenvectors, p_wave_ids

        return eigenvalues, roundness, flatness, p_wave_ids

    def vcg_fragments(
        self,
        annotations: Annotations,
        ecg_data: ECGData,
        mode: Literal["sum", "x", "y", "z"] = "sum",
        normalize_by_duration: bool = False,
        min_fragment_length_ms: float = 0.0,
        noise_sd_multiplier: float = 3.0,
        fragment_noise_multiplier: float = 3.0,
    ) -> tuple[list, list, list]:
        """Compute VCG fragment count, width, and amplitude for each annotated P-wave.

        A fragment is a sub-segment between two local extrema of the VCG signal whose
        amplitude change exceeds the noise threshold.

        Args:
            annotations: P-wave annotations.
            ecg_data: ECG signal source.
            mode: VCG combination mode — ``"sum"`` uses the Euclidean norm of X/Y/Z;
                ``"x"``, ``"y"``, ``"z"`` use a single VCG lead (default ``"sum"``).
            normalize_by_duration: If True, normalize each metric per 100 ms of
                P-wave duration, as in the original fragmentation paper (default False).
            min_fragment_length_ms: Fragments shorter than this duration in ms are
                discarded (default 0).
            noise_sd_multiplier: Multiplier for filtering out pre-onset noise estimation
                windows that are too large relative to P-wave amplitude (default 3.0).
            fragment_noise_multiplier: A fragment's amplitude change must exceed this
                multiple of the noise level to count as a valid fragment (default 3.0).

        Raises:
            ValueError: If ``mode`` is not one of the supported values.

        Returns:
            Tuple of (fragment count, fragment width, fragment amplitude) per annotation.

        References:
            Murthy et al. (2014). Number of P-wave fragmentations as a marker of atrial fat.
        """
        vcg = self._get_vcg_signal(ecg_data=ecg_data)

        vcg_map = {
            "SUM": np.linalg.norm(vcg, axis=0),
            "X": vcg[0],
            "Y": vcg[1],
            "Z": vcg[2],
        }

        mode = mode.upper()

        if mode not in vcg_map:
            raise ValueError(
                f"Invalid mode '{mode}'. Expected one of {list(vcg_map.keys())}."
            )

        vcg_signal = vcg_map[mode]

        vcg_ann = annotations.vcg_annotations()
        fragments = self._vcg_fragments_from_signal(
            vcg_ann,
            ecg_data,
            vcg_signal,
            min_fragment_length_ms=min_fragment_length_ms,
            noise_sd_multiplier=noise_sd_multiplier,
            fragment_noise_multiplier=fragment_noise_multiplier,
        )

        raw_counts = np.asarray(fragment_count(fragments), dtype=float)
        raw_widths = np.asarray(
            fragment_width(fragments, ecg_data.get_sampling_frequency()),
            dtype=float,
        )
        raw_amplitudes = np.asarray(fragment_amplitude(fragments), dtype=float)

        if not normalize_by_duration:
            return (
                list(raw_counts),
                list(raw_widths),
                list(raw_amplitudes),
            )

        durations = self.duration(vcg_ann, ecg_data.get_sampling_frequency())
        normalization_factors = np.where(durations > 0, 0.1 / durations, np.nan)

        return (
            list(raw_counts * normalization_factors),
            list(raw_widths * normalization_factors),
            list(raw_amplitudes * normalization_factors),
        )

    def get_offset_amplitude(self, annotations: Annotations, ecg_data: ECGData) -> list:
        """Return the amplitude at the offset of each annotated segment."""
        return self._compute_segment_metric(annotations, ecg_data, lambda x, _: x[-1])

    def get_onset_offset_angle(
        self, annotations: Annotations, ecg_data: ECGData
    ) -> list:
        """Compute the angle between the onset and offset of each annotated segment."""
        fs = ecg_data.get_sampling_frequency()
        scale = 1000 / fs
        return self._compute_segment_metric(
            annotations, ecg_data, lambda x, _: np.arctan(x[-1] / (len(x) * scale))
        )

    def get_shannon_entropy(
        self,
        annotations: Annotations,
        ecg_data: ECGData,
        n_bins: int = 32,
        bin_range: tuple[float, float] = None,
    ) -> list:
        """Compute Shannon entropy of the amplitude distribution for each annotated P-wave.

        Entropy is computed over a histogram of signal amplitudes; higher values indicate
        a more uniform amplitude distribution (complex or noisy shape), lower values
        indicate a more concentrated distribution (simple shape).

        Args:
            annotations: P-wave annotations.
            ecg_data: ECG signal source.
            n_bins: Number of histogram bins (default 32).
            bin_range: Amplitude range (min, max) for the histogram;
                defaults to the segment's own min/max.

        Returns:
            Shannon entropy per annotation.
        """

        def shannon_entropy_wrapper(segment, _):
            return shannon_entropy(segment, n_bins, bin_range)

        return self._compute_segment_metric(
            annotations, ecg_data, shannon_entropy_wrapper
        )

    def get_sample_entropy(
        self,
        annotations: Annotations,
        ecg_data: ECGData,
        m: int = 2,
        r_factor: float = 0.25,
    ) -> list:
        """Compute sample entropy of each annotated P-wave segment.

        Sample entropy measures signal complexity: lower values indicate more
        self-similar (regular) waveforms; higher values indicate less predictable
        (complex or noisy) shapes.

        Args:
            annotations: P-wave annotations.
            ecg_data: ECG signal source.
            m: Template length (embedding dimension). Typical value: 2.
            r_factor: Tolerance as a fraction of the segment standard deviation.
                Typical range: 0.2–0.25.

        Returns:
            Sample entropy per annotation.

        References:
            Richman & Moorman (2000). https://doi.org/10.1152/ajpheart.2000.278.6.H2039
        """

        def _wrapper(segment: np.ndarray, _) -> float:
            return sample_entropy_nolds(segment, m, r_factor)

        return self._compute_segment_metric(annotations, ecg_data, _wrapper)

    def _build_group_features(
        self,
        annotations: Annotations,
        ecg_data: ECGData,
        *,
        extrema_threshold_multiplier: float,
        shannon_entropy_n_bins: int,
        shannon_entropy_bin_range,
        sample_entropy_m: int,
        sample_entropy_r_factor: float,
        noise_sd_multiplier: float,
        fragment_noise_multiplier: float,
        min_fragment_length_ms: float,
        normalize_by_duration: bool,
    ) -> dict:
        fs = ecg_data.get_sampling_frequency()

        group_features = {
            "duration": self.duration(annotations, fs),
            "area": self.area(annotations, ecg_data),
            "area_to_duration_ratio": self.area_to_duration_ratio(
                annotations, ecg_data
            ),
            "max_absolute_amplitude": self.max_absolute_amplitude(
                annotations, ecg_data
            ),
            "max_amplitude": self.max_amplitude(annotations, ecg_data),
            "min_amplitude": self.min_amplitude(annotations, ecg_data),
            "ptp_amplitude": self.peak_to_peak_amplitude(annotations, ecg_data),
            "ptf": self.ptf(annotations, ecg_data),
            "ptf_auto": self.ptf(
                annotations.drop(
                    columns=[
                        c
                        for c in ("p_wave_morphology", "inflection_point")
                        if c in annotations.columns
                    ]
                ),
                ecg_data,
            ),
            "dispersion": self.dispersion(annotations, fs),
            "atrial_rate": self.atrial_rate(annotations, ecg_data),
            "heart_rate": self.heart_rate(annotations, ecg_data),
            "offset_amplitude": self.get_offset_amplitude(annotations, ecg_data),
            "onset_offset_angle": self.get_onset_offset_angle(annotations, ecg_data),
            "complexity": self.complexity(
                annotations,
                ecg_data,
                threshold_multiplier=extrema_threshold_multiplier,
            ),
            "shannon_entropy": self.get_shannon_entropy(
                annotations,
                ecg_data,
                n_bins=shannon_entropy_n_bins,
                bin_range=shannon_entropy_bin_range,
            ),
            "sample_entropy": self.get_sample_entropy(
                annotations,
                ecg_data,
                m=sample_entropy_m,
                r_factor=sample_entropy_r_factor,
            ),
        }

        (
            group_features["fragment_count"],
            group_features["fragment_width"],
            group_features["fragment_height"],
        ) = self.fragment_metrics(
            annotations,
            ecg_data,
            normalize_by_duration=normalize_by_duration,
            min_fragment_length_ms=min_fragment_length_ms,
            noise_sd_multiplier=noise_sd_multiplier,
            fragment_noise_multiplier=fragment_noise_multiplier,
        )

        return group_features

    def _map_axis_by_p_wave_id(
        self, annotations: Annotations, axis_values: list[float]
    ) -> pd.Series:
        axis_by_p_wave_id = dict(
            zip(sorted(pd.unique(annotations[AnnotationSchema.P_WAVE_ID])), axis_values)
        )
        return annotations[AnnotationSchema.P_WAVE_ID].map(axis_by_p_wave_id)

    def _build_vcg_feature_frame(
        self,
        annotations: Annotations,
        ecg_data: ECGData,
        *,
        noise_sd_multiplier: float,
        fragment_noise_multiplier: float,
        min_fragment_length_ms: float,
        normalize_by_duration: bool,
    ) -> pd.DataFrame:
        missing = set(VCG_LEADS) - set(ecg_data.get_lead_to_index())
        if missing:
            warnings.warn(
                f"Skipping VCG features: missing leads {sorted(missing)} in "
                f"{self._file_label(annotations)}"
            )
            p_wave_ids = annotations.vcg_annotations()[
                AnnotationSchema.P_WAVE_ID
            ].tolist()
            nan_cols = {col: [np.nan] * len(p_wave_ids) for col in VCG_FEATURE_COLUMNS}
            return pd.DataFrame({AnnotationSchema.P_WAVE_ID: p_wave_ids, **nan_cols})

        vcg_eigenvalues, roundness, flatness, p_wave_ids = self.vcg_eigenfeatures(
            annotations, ecg_data
        )
        vcg_area = self.vcg_area(annotations, ecg_data)
        vcg_axis_vals = self.vcg_axis(annotations, ecg_data)
        vcg_axis_elevation, vcg_axis_azimuth = (
            zip(*vcg_axis_vals) if vcg_axis_vals else ([], [])
        )

        def _vcg_frag(mode: Literal["sum", "x", "y", "z"]) -> tuple[list, list, list]:
            return self.vcg_fragments(
                annotations,
                ecg_data,
                mode=mode,
                normalize_by_duration=normalize_by_duration,
                min_fragment_length_ms=min_fragment_length_ms,
                noise_sd_multiplier=noise_sd_multiplier,
                fragment_noise_multiplier=fragment_noise_multiplier,
            )

        vcg_frag = {
            "sum": _vcg_frag("sum"),
            "x": _vcg_frag("x"),
            "y": _vcg_frag("y"),
            "z": _vcg_frag("z"),
        }

        vcg_eigenvalues = np.array(vcg_eigenvalues)
        return pd.DataFrame(
            {
                AnnotationSchema.P_WAVE_ID: p_wave_ids,
                "vcg_eigenvalues_1": vcg_eigenvalues[:, 0],
                "vcg_eigenvalues_2": vcg_eigenvalues[:, 1],
                "vcg_eigenvalues_3": vcg_eigenvalues[:, 2],
                "vcg_roundness": roundness,
                "vcg_flatness": flatness,
                "vcg_area": vcg_area,
                "vcg_axis_elevation": list(vcg_axis_elevation),
                "vcg_axis_azimuth": list(vcg_axis_azimuth),
                "vcg_sum_fragment_count": vcg_frag["sum"][0],
                "vcg_sum_fragment_width": vcg_frag["sum"][1],
                "vcg_sum_fragment_height": vcg_frag["sum"][2],
                "vcg_x_fragment_count": vcg_frag["x"][0],
                "vcg_x_fragment_width": vcg_frag["x"][1],
                "vcg_x_fragment_height": vcg_frag["x"][2],
                "vcg_y_fragment_count": vcg_frag["y"][0],
                "vcg_y_fragment_width": vcg_frag["y"][1],
                "vcg_y_fragment_height": vcg_frag["y"][2],
                "vcg_z_fragment_count": vcg_frag["z"][0],
                "vcg_z_fragment_width": vcg_frag["z"][1],
                "vcg_z_fragment_height": vcg_frag["z"][2],
            }
        )

    def compute_all(
        self,
        annotations: Annotations,
        ecg_data: ECGData,
        extrema_threshold_multiplier: float = 0.1,
        shannon_entropy_n_bins: int = 32,
        shannon_entropy_bin_range: tuple | None = None,
        sample_entropy_m: int = 2,
        sample_entropy_r_factor: float = 0.25,
        noise_sd_multiplier: float = 3.0,
        fragment_noise_multiplier: float = 3.0,
        morphology_min_phase_fraction: float = 0.1,
        morphology_noise_sd_multiplier: float = 3.0,
        min_fragment_length_ms: float = 0.0,
        normalize_by_duration: bool = False,
    ) -> pd.DataFrame:
        """Compute all P-wave features for a group of annotations.

        For batch use, prefer ``Pipeline``, which calls this method with
        parameters drawn from ``PipelineConfig``.

        Args:
            annotations: P-wave annotations for a single recording.
            ecg_data: ECG signal source.

            # complexity():
            extrema_threshold_multiplier: Peak must exceed this fraction of max
                amplitude to count as an extremum.

            # get_shannon_entropy():
            shannon_entropy_n_bins: Number of histogram bins.
            shannon_entropy_bin_range: Amplitude range (min, max) for the histogram;
                ``None`` uses the segment's own min/max.

            # get_sample_entropy():
            sample_entropy_m: Template length (embedding dimension).
            sample_entropy_r_factor: Tolerance as a fraction of segment standard deviation.

            # estimate_noise() / fragment_metrics() / vcg_fragments():
            noise_sd_multiplier: Pre-onset windows whose std exceeds this many multiples
                of the estimated noise are excluded from the noise estimate.
            fragment_noise_multiplier: A fragment boundary must produce an amplitude
                change exceeding this many noise SDs to count as a valid fragment.

            # classify_p_wave_morphology():
            morphology_min_phase_fraction: Minimum fraction of segment length a sign
                group must span to count as a distinct morphology phase.
            morphology_noise_sd_multiplier: Noise SD multiplier used inside the
                morphology classifier.

            # fragment_metrics() / vcg_fragments():
            min_fragment_length_ms: Fragments shorter than this (ms) are discarded.
            normalize_by_duration: If ``True``, normalize fragment metrics per 100 ms
                of P-wave duration.

        Returns:
            DataFrame indexed like ``annotations``, with one column per feature.
        """
        working_annotations = annotations.copy()

        phase_types, inflection_points = self.classify_p_wave_morphology(
            working_annotations,
            ecg_data,
            morphology_min_phase_fraction=morphology_min_phase_fraction,
            morphology_noise_sd_multiplier=morphology_noise_sd_multiplier,
            noise_sd_multiplier=noise_sd_multiplier,
        )
        working_annotations["p_wave_morphology"] = phase_types
        working_annotations["inflection_point"] = inflection_points

        group_features = {
            "p_wave_morphology": phase_types,
            "inflection_point": inflection_points,
            **self._build_group_features(
                working_annotations,
                ecg_data,
                extrema_threshold_multiplier=extrema_threshold_multiplier,
                shannon_entropy_n_bins=shannon_entropy_n_bins,
                shannon_entropy_bin_range=shannon_entropy_bin_range,
                sample_entropy_m=sample_entropy_m,
                sample_entropy_r_factor=sample_entropy_r_factor,
                noise_sd_multiplier=noise_sd_multiplier,
                fragment_noise_multiplier=fragment_noise_multiplier,
                min_fragment_length_ms=min_fragment_length_ms,
                normalize_by_duration=normalize_by_duration,
            ),
        }
        features_df = pd.DataFrame(group_features, index=working_annotations.index)

        # --- Axis feature ---
        features_df["axis"] = np.nan
        if set(AXIS_LEADS).issubset(ecg_data.get_lead_to_index()):
            try:
                axis = self.axis(working_annotations, ecg_data)
                features_df["axis"] = self._map_axis_by_p_wave_id(
                    working_annotations, axis
                )
            except Exception:
                warnings.warn(
                    f"Failed to annotate axis for {self._file_label(working_annotations)}"
                )
        else:
            warnings.warn(
                f"Skipping axis: leads I and aVF not present in "
                f"{self._file_label(working_annotations)}"
            )

        # --- VCG features ---
        vcg_df = self._build_vcg_feature_frame(
            working_annotations,
            ecg_data,
            noise_sd_multiplier=noise_sd_multiplier,
            fragment_noise_multiplier=fragment_noise_multiplier,
            min_fragment_length_ms=min_fragment_length_ms,
            normalize_by_duration=normalize_by_duration,
        )

        # Merge VCG features into features_df
        merged = working_annotations._df.merge(
            vcg_df, on=AnnotationSchema.P_WAVE_ID, how="left"
        )
        vcg_cols = vcg_df.columns.difference([AnnotationSchema.P_WAVE_ID])
        features_df[vcg_cols] = merged[vcg_cols].values

        return features_df
