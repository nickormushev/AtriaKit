# ECG P-Wave Annotation Tool

A desktop application for manually annotating P-wave onsets, offsets, and QRS onsets in 12-lead ECG recordings stored as DICOM files.

## Features

- Loads DICOM ECG files recursively from a selected directory
- Displays all 12 leads with an ECG-standard grid (dynamically hides minor grid lines when zoomed out)
- Annotate P-wave onset (red), offset (green), and QRS onset (blue) per lead
- Per-lead ignore and confidence controls
- Overview window with region selection for navigating long recordings
- Saves annotations to CSV; re-opening a patient restores previous annotations
- Ships as a standalone executable (no Python installation required for end users)

## Output

Annotations are saved to `p-wave-annotations.csv` in the selected DICOM directory. Each row represents one annotated P-wave:

| Column | Description |
|---|---|
| `patient_id` | DICOM patient ID |
| `lead` | Lead name (e.g. `I`, `II`, `V1`) |
| `p_wave_id` | Index of the P-wave within the recording |
| `onset` | P-wave onset, in samples |
| `offset` | P-wave offset, in samples |
| `qrs_onset` | QRS onset, in samples (`-1` if not annotated) |
| `ignore` | Whether the lead was marked as ignored |
| `type` | Annotation type: `Before`, `After`, or `Both` |
| `confidence` | Annotator confidence: `Low`, `Med`, or `High` |
| `comment` | Free-text comment for the recording |
| `file_path` | Path to the source DICOM file |

## Running from source

**Requirements:** Python 3.11+

```bash
pip install -r requirements.txt
python annotate.py
```

A directory picker will open on launch. Select the folder containing your DICOM (`.IMA`) files — the tool searches recursively.

## Building the executable

Requires [PyInstaller](https://pyinstaller.org):

```bash
pip install pyinstaller
bash make_binary.sh
```

The standalone executable will be placed in `dist/annotate/`.

## Running tests

```bash
pytest annotate_test.py annotation_service_test.py
```

## License

MIT — see [LICENSE](LICENSE).
