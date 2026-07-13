import logging

import numpy as np

from atriakit.annotations import Annotations
from atriakit.io import ECGLoader
from atriakit.models.annotations import AnnotationSchema
from atriakit.preprocessing.signals import SignalPreprocessor

LOGGER = logging.getLogger(__name__)


class ECGDataset:
    """Utility for computing normalization statistics across a set of ECG files.

    Used internally by ``Pipeline`` to compute per-lead mean and standard
    deviation before z-score normalization.

    Args:
        dataset_dir: Base directory containing the ECG files.
        loader: ECG file loader. Defaults to ``ECGLoader()``.

    Attributes:
        dataset_dir: Base directory for ECG files.
        loader: Active ECG file loader.
    """

    def __init__(
        self,
        dataset_dir,
        loader: ECGLoader | None = None,
    ):
        self.dataset_dir = dataset_dir
        self.loader = loader or ECGLoader()

    def calculate_mean_std_p_waves(
        self,
        annotations: Annotations,
        preprocessor: SignalPreprocessor,
    ):
        """Compute per-lead mean and std over P-wave segments across all files.

        Args:
            annotations: Annotation table referencing files in ``dataset_dir``.
            preprocessor: Preprocessor to apply; its normalization is bypassed.

        Returns:
            Tuple of ``(mean, std)`` arrays of shape ``(n_leads,)``, where
            ``n_leads`` is inferred from the first file loaded.

        Raises:
            ValueError: If an annotation's window is invalid (end index not
                after start index), or if no samples were accumulated for at
                least one lead.

        Note:
            All recordings must have the same number of leads in the same order.
            Mixed-lead datasets (e.g. combining 8-lead and 12-lead files) are
            not supported and will produce incorrect statistics.
        """
        total_samples = sum_x = sum_x2 = None
        preprocessor = preprocessor.with_overrides(normalization_type="none")

        n_files = annotations[AnnotationSchema.FILE_PATH].nunique()
        LOGGER.info(
            "Starting P-wave normalization statistics over %d file(s) with lowcut=%s, highcut=%s.",
            n_files,
            preprocessor.config.lowcut,
            preprocessor.config.highcut,
        )

        for file_idx, (file_path, file_annotations) in enumerate(
            annotations.groupby(AnnotationSchema.FILE_PATH), start=1
        ):
            LOGGER.info(
                "Normalization progress: file %d/%d (%s), %d annotation(s).",
                file_idx,
                n_files,
                file_path,
                len(file_annotations),
            )
            full_path = f"{self.dataset_dir}/{file_path}"
            try:
                ecg_data = self.loader.load(full_path)
            except FileNotFoundError:
                LOGGER.warning(
                    "File not found during normalization: %s. Skipping.",
                    full_path,
                )
                continue

            filtered_ecg = ecg_data.get_ecg(preprocessor=preprocessor)
            if total_samples is None:
                n_leads = filtered_ecg.shape[0]
                total_samples = np.zeros(n_leads)
                sum_x = np.zeros(n_leads)
                sum_x2 = np.zeros(n_leads)

            for row in file_annotations.itertuples():
                start = int(max(row.onset - 30, 0))
                end = int(min(row.offset + 20, filtered_ecg.shape[1]))

                if end <= start:
                    raise ValueError(
                        f"Invalid normalisation window: start index ({start}) "
                        f"is greater than end index ({end}). "
                        f"Onset feature: 'onset', "
                        f"offset feature: 'offset'."
                    )

                ecg_segment = filtered_ecg[:, start:end]
                sum_x += np.sum(ecg_segment, axis=1)
                sum_x2 += np.sum(ecg_segment**2, axis=1)
                total_samples += ecg_segment.shape[1]

        if np.any(total_samples == 0):
            raise ValueError(
                "No samples accumulated for at least one lead during normalisation."
            )

        mean = sum_x / total_samples
        std = np.sqrt(sum_x2 / total_samples - mean**2)
        LOGGER.info("Finished P-wave normalization statistics.")
        return mean, std

    def calculate_mean_std(
        self,
        annotations: Annotations,
        preprocessor: SignalPreprocessor,
    ):
        """Compute per-lead mean and std over full recording segments across all files.

        Args:
            annotations: Annotation table referencing files in ``dataset_dir``.
            preprocessor: Preprocessor to apply; its normalization is bypassed.

        Returns:
            Tuple of ``(mean, std)`` arrays of shape ``(n_leads,)``, where
            ``n_leads`` is inferred from the first file loaded.

        Raises:
            ValueError: If no samples were accumulated for at least one lead.

        Note:
            All recordings must have the same number of leads in the same order.
            Mixed-lead datasets (e.g. combining 8-lead and 12-lead files) are
            not supported and will produce incorrect statistics.
        """
        total_samples = sum_x = sum_x2 = None
        preprocessor = preprocessor.with_overrides(normalization_type="none")

        n_files = annotations[AnnotationSchema.FILE_PATH].nunique()
        LOGGER.info(
            "Starting global normalization statistics over %d file(s) with lowcut=%s, highcut=%s.",
            n_files,
            preprocessor.config.lowcut,
            preprocessor.config.highcut,
        )

        for file_idx, (file_path, file_annotations) in enumerate(
            annotations.groupby(AnnotationSchema.FILE_PATH), start=1
        ):
            LOGGER.info(
                "Global normalization progress: file %d/%d (%s), %d annotation(s).",
                file_idx,
                n_files,
                file_path,
                len(file_annotations),
            )
            full_path = f"{self.dataset_dir}/{file_path}"
            try:
                ecg_data = self.loader.load(full_path)
            except FileNotFoundError:
                LOGGER.warning(
                    "File not found during normalization: %s. Skipping.",
                    full_path,
                )
                continue

            filtered_ecg = ecg_data.get_ecg(preprocessor=preprocessor)
            if total_samples is None:
                n_leads = filtered_ecg.shape[0]
                total_samples = np.zeros(n_leads)
                sum_x = np.zeros(n_leads)
                sum_x2 = np.zeros(n_leads)

            groups = (
                [g for _, g in file_annotations.groupby(AnnotationSchema.TYPE)]
                if AnnotationSchema.TYPE in file_annotations.columns
                else [file_annotations]
            )
            for annotations_for_group in groups:
                start = int(
                    max(annotations_for_group[AnnotationSchema.ONSET].min() - 100, 0)
                )
                end = annotations_for_group[AnnotationSchema.OFFSET].max() + 100
                if AnnotationSchema.QRS_ONSET in annotations_for_group.columns:
                    end = max(
                        end, annotations_for_group[AnnotationSchema.QRS_ONSET].max()
                    )
                end = int(min(end, filtered_ecg.shape[1]))
                ecg_segment = filtered_ecg[:, start:end]
                sum_x += np.sum(ecg_segment, axis=1)
                sum_x2 += np.sum(ecg_segment**2, axis=1)
                total_samples += ecg_segment.shape[1]

        if total_samples is None or np.any(total_samples == 0):
            raise ValueError(
                "No samples accumulated for at least one lead during normalisation."
            )

        mean = sum_x / total_samples
        std = np.sqrt(sum_x2 / total_samples - mean**2)
        LOGGER.info("Finished global normalization statistics.")
        return mean, std
