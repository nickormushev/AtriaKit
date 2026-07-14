from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class PatientInfo:
    id: str = "anonymous"


@dataclass
class ECGRecording:
    """Generic ECG recording."""

    patient: PatientInfo = field(default_factory=PatientInfo)
    file_name: str = ""
    start: datetime | None = None
    lead_names: np.ndarray = field(default_factory=lambda: np.array([]))
    ecg_mV: np.ndarray = field(default_factory=lambda: np.array([]))
    fs_Hz: float = 1000.0
    pfa_signals: np.ndarray | None = None
    pfa_lead_names: list[str] = field(default_factory=list)

    @property
    def start_hms(self) -> str:
        """Start time as HHMMSS string, for display and annotation alignment."""
        return self.start.strftime("%H%M%S") if self.start else "000000"
