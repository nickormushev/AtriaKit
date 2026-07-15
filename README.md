<p align="center">
  <img src="https://raw.githubusercontent.com/nickormushev/AtriaKit/main/docs/logo.png" alt="AtriaKit" height="90">
</p>

<p align="center">
An Open-Source Framework for Reproducible P-Wave Phenotyping
</p>

## Quick Example

The fastest way to try AtriaKit hands-on is `examples/demo.ipynb`, which generates synthetic ECG data and runs it through the full pipeline end to end.

```python
from atriakit import AnnotationsLoader, Pipeline

annotations = AnnotationsLoader().load("annotations.csv")
pipeline = Pipeline(ecg_base_path="path/to/dicom/root")

features = pipeline.run(annotations)
print(features.get_df().head())
```

## Highlights

- Temporal, morphological, nonlinear, and spatial (VCG) P-wave descriptors, each with an explicit mathematical definition; see the [manuscript](#citation)
- Paired with [`ecg_annotator`](https://github.com/nickormushev/AtriaKit/tree/main/ecg_annotator), an interactive tool for expert P-wave boundary annotation
- Configuration-driven: filtering, boundary mode, and baseline correction are explicit and reproducible, not hard-coded; tune them via config classes directly in Python or a single YAML file that maps to them
- Batch feature computation across recordings through a standardized `Pipeline` interface
- Unit-tested reusable components

Atrial fibrillation (AF) is a major cause of cardiovascular morbidity, and the P-wave of the surface electrocardiogram is an important non-invasive marker of atrial remodeling and conduction heterogeneity. In practice, however, P-wave analysis is often implemented through custom or ad hoc signal-processing pipelines, which makes results harder to reproduce across studies. General ECG processing libraries exist, but they often do not provide the more specialized atrial and P-wave metrics needed for high-resolution AF research.

AtriaKit addresses that gap with a modular feature-extraction library (see [Highlights](#highlights) above), built to scale from exploratory analysis to larger clinical datasets.

## Implemented Feature Families

<p align="center">
  <img src="https://raw.githubusercontent.com/nickormushev/AtriaKit/main/docs/Features.png" alt="Overview of AtriaKit's P-wave feature families" width="720">
</p>

The library currently includes:

- Duration and Dispersion
- Amplitude-Based P-Wave Features
- Area and Area-To-Duration Ratio
- P-Wave Terminal Force
- Morphology and Inflection-Point Features
- Fragmentation-Based Features
- Complexity
- Shannon Entropy and Sample Entropy
- Atrial Rate and Heart Rate
- Frontal and VCG Axis
- VCG Area and Eigenfeatures
- Signal-Averaged Vector-Magnitude P-Wave Amplitudes

The set is modular and can be extended with new metrics. For implementation details and mathematical definitions of each feature, see the AtriaKit manuscript ([Citation](#citation)).

## Installation

```bash
pip install atriakit
```

To work from a clone of this repository instead (e.g. for development), install in editable mode:

```bash
pip install -e .
```

For development dependencies (testing, etc.):

```bash
pip install -e .[dev]
```

The package requires Python 3.10+.

## Core API

The public package interface is exposed through `atriakit`:

```python
from atriakit import (
    Annotations,
    AnnotationsLoader,
    ECGData,
    ECGDataset,
    ECGLoader,
    FeatureCalculators,
    FeatureComputationConfig,
    Pipeline,
    PipelineConfig,
    SegmentConfig,
    SignalPreprocessor,
    SignalPreprocessorConfig,
)
```

Important components:

- `ECGLoader`: loads ECG waveforms from DICOM/IMA files and returns `ECGData`; register additional file types with `ECGLoader().register(extension, loader)`
- `ECGData`: holds the raw signal array plus lead layout, and returns preprocessed lead signals via a `SignalPreprocessor`
- `AnnotationsLoader`: loads a CSV or directory of annotation files into an `Annotations` object; also extensible via `.register(extension, loader)` for other annotation file formats
- `Annotations`: wrapper around the annotation table
- `FeatureCalculators`: computes ECG and VCG features for annotated segments, constructed with a `SegmentConfig`
  - `SegmentConfig`: baseline correction mode and leading-sample skip applied to extracted segments
- `Pipeline`: runs batch feature extraction over a set of files, constructed with a `PipelineConfig`
- `PipelineConfig`: bundles everything needed to tune a `Pipeline` run: segment-boundary and baseline-correction settings, preprocessing filters for both feature and morphology signals, feature-computation parameters (via `FeatureComputationConfig`), and beat-grouping tolerance
  - `FeatureComputationConfig`: per-call parameters for `FeatureCalculators.compute_all` (entropy, complexity, noise, morphology, and fragmentation settings)
- `SignalPreprocessor`: applies notch filtering, bandpass filtering, baseline correction, and normalization as a reusable preprocessing object; each step is optional and toggled independently via `SignalPreprocessorConfig`
- `ECGDataset`: computes per-lead mean/std across a set of files, for `SignalPreprocessorConfig`'s z-score normalization

## Usage

`Pipeline` is the main entry point: it batches feature computation across many recordings at once, matching each file to its rows in a combined annotations table (e.g. one `annotations.csv` covering a whole directory of DICOM/IMA files). Every config object ships with reasonable defaults, as in the Quick Example above: `Pipeline(ecg_base_path=...)` works with no further configuration, but you can tune it to your needs via a YAML file or by building the config objects directly in Python:

### 1. Configure via YAML (recommended)

Rather than building `PipelineConfig`/`SignalPreprocessorConfig` objects by hand, point `Pipeline.from_yaml` at a config file. See `examples/pipeline_config.yaml` for the full reference, covering paths, annotation boundary mode, preprocessing filters, and pipeline parameters (abbreviated below):

```yaml
ecg_base_path: ./data
annotations_dir: ./data
pipeline:
  group_tolerance_ms: 200
  signal_preprocessor_config:
    lowcut: 1
    highcut: 40
    notch_freq: 50.0
  feature_computation:
    shannon_entropy_n_bins: 128
```

```python
pipeline, annotations = Pipeline.from_yaml("pipeline_config.yaml")
result = pipeline.run(annotations)
```

Required annotation columns: `file_path`, `lead`, `onset`, `offset`, `p_wave_id`. Optional columns such as `type`, `patient_id`, and `ignore` are used when available.

### 2. Or configure directly in Python

For programmatic control (e.g. sweeping parameters in a script), pass a `PipelineConfig` straight to `Pipeline`:

```python
from atriakit import FeatureComputationConfig, Pipeline, PipelineConfig, SignalPreprocessorConfig

pipeline = Pipeline(
    ecg_base_path="path/to/dicom/root",
    pipeline_config=PipelineConfig(
        signal_preprocessor_config=SignalPreprocessorConfig(
            lowcut=1.0, highcut=40.0, notch_freq=50.0
        ),
        morphology_preprocessor_config=SignalPreprocessorConfig(
            lowcut=1.0, highcut=30.0, notch_freq=50.0
        ),
        feature_computation=FeatureComputationConfig(shannon_entropy_n_bins=128),
    ),
)
```

### 3. Compute features for one recording directly

Unlike `Pipeline`, `FeatureCalculators` works on a single already-loaded recording, so its annotations must be pre-filtered to that one file, since onset/offset are sample indices into that specific signal:

```python
from atriakit import AnnotationsLoader, ECGLoader, FeatureCalculators

ecg_data = ECGLoader().load("path/to/file.IMA")
annotations = AnnotationsLoader().load("annotations.csv")
file_annotations = annotations[annotations["file_path"] == "path/to/file.IMA"]

fc = FeatureCalculators()

area = fc.area(file_annotations, ecg_data)
max_amplitude = fc.max_amplitude(file_annotations, ecg_data)
all_features = fc.compute_all(file_annotations, ecg_data)
```

## Preprocessing

Signal preprocessing is intentionally separated from feature extraction.

- filter settings live in `SignalPreprocessorConfig`
- runtime normalization state (`mean` / `std`) lives on `SignalPreprocessor`
- `ECGData` delegates preprocessing to a `SignalPreprocessor`
- `FeatureCalculators` can use separate preprocessors for standard signal features and morphology-specific features

Example:

```python
from atriakit import ECGLoader, FeatureCalculators, SignalPreprocessor, SignalPreprocessorConfig

preprocessor = SignalPreprocessor(
    SignalPreprocessorConfig(lowcut=1.0, highcut=40.0, notch_freq=50.0)
)

ecg_data = ECGLoader().load("path/to/file.IMA")
lead_ii = ecg_data.get_lead_signal("II", preprocessor=preprocessor)

# FeatureCalculators takes the same kind of preprocessor, separately for
# standard signal features and morphology-specific features (both optional;
# see Core API above for the defaults used when omitted)
fc = FeatureCalculators(
    signal_preprocessor=preprocessor,
    morphology_preprocessor=SignalPreprocessor(
        SignalPreprocessorConfig(lowcut=1.0, highcut=30.0, notch_freq=50.0)
    ),
)
```

## Repository Layout

```text
AtriaKit/
├── atriakit/
│   ├── __init__.py
│   ├── datasets.py
│   ├── feature_calculator.py
│   ├── utils.py
│   ├── configs/
│   ├── features/
│   ├── io/
│   ├── models/
│   ├── preprocessing/
│   └── processing/
├── examples/
├── tests/
├── ecg_annotator/        # companion desktop annotation tool (separate package)
└── pyproject.toml
```

Main directories:

- `atriakit/`: reusable ECG feature extraction package
- `atriakit/io/`: DICOM/ECG loading, annotation-file loading, and waveform access
- `atriakit/models/`: `ECGData`, `Annotations`, and the annotation column schema
- `atriakit/preprocessing/`: signal and annotation preprocessing
- `atriakit/processing/`: dataset utilities and the batch pipeline
- `atriakit/features/`: individual feature implementations
- `atriakit/configs/`: processing and feature configuration
- `examples/`: demo material
- `tests/`: unit tests
- `ecg_annotator/`: standalone PyQt6 desktop app for manually annotating P-waves; a separate, independently packaged project, see its own [README](https://github.com/nickormushev/AtriaKit/blob/main/ecg_annotator/README.md)

## Testing

Run the test suite with:

```bash
pytest
```

Or run a specific test module:

```bash
pytest tests/test_features.py -q
```

## Notes

The repository favors a pragmatic research-code style, but the core package is structured to support gradual cleanup, extension, and more standardized use over time.

## Citation

<!-- TODO: fill in once the paper is public -->

If you use AtriaKit in your research, please cite:

```bibtex
@software{atriakit,
  author  = {Kormushev, Nikolay},
  title   = {AtriaKit: An Open-Source Framework for Reproducible P-Wave Phenotyping},
  year    = {2026},
  url     = {https://github.com/nickormushev/AtriaKit}
}
```

## License

MIT, see [LICENSE](https://github.com/nickormushev/AtriaKit/blob/main/LICENSE).

## Development Ideas

Areas that would likely improve maintainability further:

- split `feature_calculator.py` into smaller focused modules over time
- expand examples and API documentation
- add parallelisation
