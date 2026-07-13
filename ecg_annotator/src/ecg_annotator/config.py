import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_FILE = Path("ecg-annotator.toml")


@dataclass
class AppConfig:
    output_path:        str   = "./p-wave-annotations.csv"
    distance_threshold: float = 0.4
    default_confidence: str   = "High"
    amplitude_scale:    float = 10.0


DEFAULTS = AppConfig()

_SAMPLE_CONFIG = """\
# ECG Annotator configuration
# Edit this file to change application defaults.

# Where annotation CSVs are saved (when no path is given on the command line)
output_path = "./p-wave-annotations.csv"

# Maximum allowed gap (seconds) between a red and its paired green line
distance_threshold = 0.4

# Default confidence level for new annotations ("Low", "Med", "High")
default_confidence = "High"

# Amplitude scaling factor — increase to make waves taller, decrease to compress them
# This also controls the ECG paper grid calibration (1 large box = 0.5 mV)
amplitude_scale = 10.0
"""


def load_config() -> AppConfig:
    if not CONFIG_FILE.exists():
        _write_sample_config()
        return AppConfig()

    with CONFIG_FILE.open("rb") as f:
        raw = tomllib.load(f)

    return AppConfig(**{
        k: v for k, v in raw.items() if k in AppConfig.__dataclass_fields__
    })


def _write_sample_config() -> None:
    CONFIG_FILE.write_text(_SAMPLE_CONFIG)
