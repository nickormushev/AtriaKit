import pandas as pd

from pathlib import Path


ANNOTATION_COLUMNS = [
    "patient_id",
    "lead",
    "p_wave_id",
    "onset",
    "offset",
    "qrs_onset",
    "ignore",
    "comment",
    "confidence",
    "type",
    "file_path",
]


class AnnotationsLoader:
    def __init__(self, annotations_path: Path):
        self.annotations_path = annotations_path
        if annotations_path.exists():
            self.df = pd.read_csv(annotations_path)
        else:
            self.df = pd.DataFrame(columns=ANNOTATION_COLUMNS)

    def get_file_annotations(self, file_path):
        return self.df.loc[self.df["file_path"] == file_path]

    def replace_file_annotations(self, patient_file, rows_to_add: list[dict]):
        # Remove existing annotations for this file
        self.df = self.df[self.df["file_path"] != patient_file]
        if rows_to_add:
            self.df = pd.concat([self.df, pd.DataFrame(rows_to_add)], ignore_index=True)

    def save(self, output_file: Path | None = None):
        target = output_file or self.annotations_path
        self.df.to_csv(target, index=False)

    def get_annotations_for_ecg(self, patient_ecg_file):
        """As we regenerated some IMA files their names change so we build and id with which we can make sure it's the same file and load the annotations"""

        def make_id_from_path(path):
            # safe extraction of basename and parts, robust to filenames with few dots
            fname = Path(path).name
            parts = fname.split(".")
            first = parts[0] if len(parts) >= 1 else ""
            second_last = parts[-2] if len(parts) >= 2 else ""  # '' if not present
            return first + second_last

        file_path_id = self.df["file_path"].apply(make_id_from_path)
        patient_file_id = make_id_from_path(patient_ecg_file)
        mask = file_path_id.str.endswith(patient_file_id)
        return self.df[mask]
