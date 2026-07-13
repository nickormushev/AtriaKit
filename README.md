# AtriaKit

AtriaKit is a framework for atrial ECG analysis, designed to support
standardized and reproducible extraction of P-wave features from annotated
12-lead ECG recordings.

Atrial fibrillation (AF) is a major cause of cardiovascular morbidity, and the
P-wave of the surface electrocardiogram is an important non-invasive marker of
atrial remodeling and conduction heterogeneity. In practice, however, P-wave
analysis is often implemented through custom or ad hoc signal-processing
pipelines, which makes results harder to reproduce across studies. General ECG
processing libraries exist, but they often do not provide the more specialized
atrial and P-wave metrics needed for high-resolution AF research.

This repository addresses that gap through a modular Python library for
feature extraction and a companion annotation workflow for research use. The
core reusable package is centered on `ecg/` and provides:

- DICOM ECG loading and preprocessing
- annotation handling utilities
- beat- and group-level P-wave feature extraction
- batch feature computation through a standardized pipeline interface
- unit tests for the main reusable components

Paper-specific notebooks and one-off research utilities are kept separate from
the core package.

## Why AtriaKit

The project is intended to reduce implementation effort and improve
reproducibility in atrial ECG research by packaging established temporal,
morphological, and spatial P-wave metrics into a single consistent framework.
The emphasis is on modularity, explicit preprocessing choices, and a workflow
that can scale from exploratory analysis to larger clinical datasets.

## Repository Layout

```text
feature-library/
├── ecg/
│   ├── __init__.py
│   ├── annotations.py
│   ├── feature_calculator.py
│   ├── preprocessing/
│   ├── utils.py
│   ├── configs/
│   ├── features/
│   ├── io/
│   └── processing/
├── examples/
├── paper/
├── tests/
└── pyproject.toml
```

Main directories:

- `ecg/`: reusable ECG feature extraction package
- `ecg/io/`: waveform loading and preprocessed signal access
- `ecg/preprocessing/`: reusable signal and annotation preprocessing helpers
- `ecg/processing/`: dataset utilities and batch pipeline
- `ecg/features/`: individual feature implementations
- `ecg/configs/`: processing and feature configuration
- `examples/`: demo material
- `paper/`: manuscript-specific code and notebooks
- `tests/`: unit tests

## Installation

Create and activate a Python environment, then install the package in editable
mode:

```bash
pip install -e .
```

Alternatively, install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

For development dependencies:

```bash
pip install -e .[dev]
```

The package requires Python 3.10+.

`requirements.txt` contains a cleaned minimal dependency list for the packages
currently imported by the repository, including the small set of testing
dependencies used in `tests/`.

## Core API

The public package interface is exposed through `ecg`:

```python
from ecg import (
    Annotations,
    ECGData,
    FeatureCalculatorConfig,
    FeatureCalculators,
    Pipeline,
    PipelineConfig,
    SignalPreprocessorConfig,
)
```

Signal preprocessing itself is provided through `ecg.preprocessing`:

```python
from ecg import SignalPreprocessorConfig
from ecg.preprocessing import SignalPreprocessor
```

Important components:

- `ECGData`: loads ECG waveforms from DICOM and exposes preprocessed lead signals
- `Annotations`: wrapper around annotation tables
- `FeatureCalculators`: computes ECG and VCG features for annotated segments
- `Pipeline`: runs batch feature extraction over a set of files
- `FeatureCalculatorConfig`: controls baseline correction and annotation
  boundary settings
- `SignalPreprocessorConfig`: defines filter settings for normal and
  morphology-specific preprocessing
- `SignalPreprocessor`: applies notch filtering, bandpass filtering, and
  optional normalization as a reusable preprocessing object

## Preprocessing

Signal preprocessing is intentionally separated from feature extraction.

- filter settings live in `SignalPreprocessorConfig`
- runtime normalization state (`mean` / `std`) lives on `SignalPreprocessor`
- `ECGData` delegates preprocessing to a `SignalPreprocessor`
- `FeatureCalculators` can use separate preprocessors for standard signal
  features and morphology-specific features

This keeps filtering logic out of `FeatureCalculatorConfig` and makes it easy
to swap preprocessing strategies without changing the rest of the API.

Example:

```python
from ecg import ECGData, SignalPreprocessorConfig
from ecg.preprocessing import SignalPreprocessor

preprocessor = SignalPreprocessor(
    SignalPreprocessorConfig(lowcut=1.0, highcut=40.0, notch_freq=50.0)
)

ecg_data = ECGData("path/to/file.IMA", preprocessor=preprocessor)
lead_ii = ecg_data.get_lead_signal("II")
```

You can also override the preprocessor per call:

```python
morphology_preprocessor = SignalPreprocessor(
    SignalPreprocessorConfig(lowcut=1.0, highcut=30.0, notch_freq=50.0)
)

lead_ii_morphology = ecg_data.get_lead_signal(
    "II",
    preprocessor=morphology_preprocessor,
)
```

## Quick Start

### 1. Run the pipeline

```python
import logging
import pandas as pd

from ecg import Pipeline, PipelineConfig

logging.basicConfig(level=logging.INFO, force=True)

annotations = pd.read_csv("annotations.csv")

pipeline = Pipeline(
    dicom_base_path="path/to/dicom/root",
    annotations=annotations,
    pipeline_config=PipelineConfig(
        shannon_entropy_n_bins=64,
        sample_entropy_m=2,
        sample_entropy_r_factor=0.25,
        extrema_threshold_multiplier=0.1,
    ),
    show_progress=True,
)

features = pipeline.run()
print(features.head())
```

If you want to control the preprocessing explicitly, pass configured
preprocessors into the pipeline:

```python
from ecg import Pipeline, PipelineConfig, SignalPreprocessorConfig
from ecg.preprocessing import SignalPreprocessor

pipeline = Pipeline(
    dicom_base_path="path/to/dicom/root",
    annotations=annotations,
    signal_preprocessor=SignalPreprocessor(
        SignalPreprocessorConfig(lowcut=1.0, highcut=40.0, notch_freq=50.0)
    ),
    morphology_preprocessor=SignalPreprocessor(
        SignalPreprocessorConfig(lowcut=1.0, highcut=30.0, notch_freq=50.0)
    ),
    pipeline_config=PipelineConfig(),
)
```

Expected annotation columns:

- `file_path`
- `lead`
- `onset`
- `offset`

Optional columns such as `type`, `patient_id`, and `ignore` are used when
available.

### 2. Compute features for one recording directly

```python
import pandas as pd

from ecg import ECGData, FeatureCalculators

ecg_data = ECGData("path/to/file.IMA")
annotations = pd.read_csv("annotations.csv")

fc = FeatureCalculators()
file_annotations = annotations[annotations["file_path"] == "relative/path/to/file.IMA"]

area = fc.area(file_annotations, ecg_data)
max_amplitude = fc.max_amplitude(file_annotations, ecg_data)
all_features = fc.compute_all(file_annotations, ecg_data)
```

## Implemented Feature Families

The library currently includes:

- duration and dispersion
- amplitude-based P-wave features
- area and area-to-duration ratio
- P-wave terminal force
- morphology and inflection-point features
- fragmentation-based features
- complexity
- Shannon entropy and sample entropy
- atrial rate and heart rate
- frontal axis
- VCG area and eigenfeatures
- signal-averaged vector-magnitude P-wave amplitudes

These features are designed to support atrial phenotyping and AF-oriented
research workflows while remaining modular enough to extend with new metrics.

## Examples

Demo material is available in `examples/`. These files are intended as simple
entry points for exploring the package in notebooks.

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

- Some environments may require installing optional dependencies used by certain
  feature modules, such as `nolds`.
- The `paper/` directory is not part of the core reusable package API.
- The repository currently favors a pragmatic research-code style, but the core
  package is structured to support gradual cleanup, extension, and more
  standardized use over time.

## Development Ideas

Areas that would likely improve maintainability further:

- continue migrating all annotation logic toward the `Annotations` wrapper
- split `feature_calculator.py` into smaller focused modules over time
- keep paper-specific processing isolated from the reusable `ecg` package
- expand examples and API documentation
- add parallelisation
