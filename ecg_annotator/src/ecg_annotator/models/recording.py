from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


@dataclass
class PatientInfo:
    id: str = "anonymous"
    gender: str = ""
    age_year: float = float("nan")
    height_cm: float = float("nan")
    weight_kg: float = float("nan")
    pacemaker: np.ndarray = field(default_factory=lambda: np.array([]))
    current_condition: np.ndarray = field(default_factory=lambda: np.array([]))
    medical_history: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class FilterInfo:
    band_pass_Hz: tuple[float, float] = (float("nan"), float("nan"))
    notch_Hz: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class ECGRecording:
    """Generic ECG recording, corresponding to the 'rec' struct from the MATLAB data format."""

    patient: PatientInfo = field(default_factory=PatientInfo)
    file_name: str = ""
    start: datetime | None = None
    acquisition_condition: str = ""
    lead_names: np.ndarray = field(default_factory=lambda: np.array([]))
    dimensionality: str = ""
    ecg_mV: np.ndarray = field(default_factory=lambda: np.array([]))
    fs_Hz: float = 1000.0
    filter: FilterInfo = field(default_factory=FilterInfo)
    pfa_signals: np.ndarray | None = None
    pfa_lead_names: list[str] = field(default_factory=list)

    @property
    def start_hms(self) -> str:
        """Start time as HHMMSS string, for display and annotation alignment."""
        return self.start.strftime("%H%M%S") if self.start else "000000"
