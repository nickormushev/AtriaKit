import logging
import math
from pathlib import Path

import numpy as np

from ecg_annotator.annotations.serialization import build_annotation_rows
from ecg_annotator.annotations.state import AnnotationSession
from ecg_annotator.annotations.validation import validate_annotation_session
from ecg_annotator.config import AppConfig
from ecg_annotator.const import LEAD_NAMES, MAIN_ECG_LEAD_NAMES
from ecg_annotator.loaders.annotation_loader import AnnotationsLoader
from ecg_annotator.loaders.data_loader import DataLoader
from ecg_annotator.models.recording import ECGRecording
from ecg_annotator.utils.time import time_to_sample

log = logging.getLogger(__name__)


class ECGAnnotator:
    PIXEL_TOLERANCE = 100  # max pixel distance from ECG line for a click to register

    def _reset_ecg_view(self):
        self.ecg = None
        self.window_duration = 30.0
        self.time_sec = np.arange(0, 300, 1.0)
        self.view_start = 0.0

    def _reset_annotation_state(self):
        self.annotation_session = AnnotationSession.for_leads(
            self.main_lead_names, default_confidence=self.default_confidence
        )

    def _configure_leads(self, ecg_lead_names: list[str]) -> None:
        """Update lead metadata from the loaded file's lead lists."""
        self.main_lead_names = ecg_lead_names
        self.lead_name_to_index = {n: i for i, n in enumerate(ecg_lead_names)}
        if hasattr(self, "plotter"):
            self.plotter.update_lead_controls(ecg_lead_names)

    def _wire_lead_controls(self):
        """Connect per-lead checkboxes and radio buttons to the annotation session.
        Called at init and again after lead rows are rebuilt for a new file."""
        for lead_idx, (checkbox, radio_group) in enumerate(
            zip(self.plotter.ignore_checks, self.plotter.radio_buttons)
        ):
            checkbox.toggled.connect(
                lambda checked, idx=lead_idx: self._on_ignore_changed(idx, checked)
            )
            radio_group.buttonToggled.connect(
                lambda btn, checked, idx=lead_idx: self._on_confidence_changed(
                    idx, btn.text(), checked
                )
            )

    def _wire_controls_to_session(self):
        self.plotter.comment_box.textChanged.connect(self._on_comment_changed)
        self.plotter.type_group.buttonToggled.connect(self._on_type_changed)
        self._wire_lead_controls()

    def _on_comment_changed(self, text):
        self.annotation_session.comment = text

    def _on_type_changed(self, button, checked):
        if checked:
            self.annotation_session.annotation_type = button.text()

    def _on_ignore_changed(self, lead_idx, checked):
        self.annotation_session.lead_states[lead_idx].ignored = checked

    def _on_confidence_changed(self, lead_idx, text, checked):
        if checked:
            self.annotation_session.lead_states[lead_idx].confidence = text

    def _apply_session_to_controls(self):
        if not hasattr(self, "plotter"):
            return

        self.plotter.set_comment(self.annotation_session.comment)
        self.plotter.set_selected_annotation_type(
            self.annotation_session.annotation_type
        )
        self.plotter.set_ignored_leads(
            [lead_state.ignored for lead_state in self.annotation_session.lead_states]
        )
        self.plotter.set_confidence_levels(
            [
                lead_state.confidence
                for lead_state in self.annotation_session.lead_states
            ]
        )

    def _red_points(self):
        return [
            lead_state.red_points for lead_state in self.annotation_session.lead_states
        ]

    def _green_points(self):
        return [
            lead_state.green_points
            for lead_state in self.annotation_session.lead_states
        ]

    def _blue_points(self):
        return [
            lead_state.blue_points for lead_state in self.annotation_session.lead_states
        ]

    def _update_segments(self):
        self.plotter.update_segments(
            self._red_points(),
            self._green_points(),
            self._blue_points(),
            self.cached_scaled_leads,
        )

    def _load_ecg_data(self, idx):
        ecg_data = self.ecg_data_loader.load_patient_by_idx(idx)
        if not ecg_data:
            return None

        self.ecg = ecg_data
        self.time_sec = np.arange(ecg_data.ecg_mV.shape[1]) / ecg_data.fs_Hz

        return ecg_data

    def get_visible_leads(self):
        if self.ecg is not None:
            return self.ecg.ecg_mV.shape[0]
        return 0

    def _rebuild_scaled_leads(self):
        self.cached_scaled_leads = [
            self.plotter.scale_lead(self.ecg.ecg_mV[i], i)
            for i in range(self.ecg.ecg_mV.shape[0])
        ]

    def _rebuild_pfa_leads(self):
        pfa = self.ecg.pfa_signals
        if pfa is None or pfa.size == 0:
            self.cached_pfa_leads = []
            return
        n_main = self.ecg.ecg_mV.shape[0]
        self.cached_pfa_leads = [
            self.plotter.scale_pfa_lead(pfa[i], i, n_main) for i in range(pfa.shape[0])
        ]

    def _plot_pfa_if_available(self):
        if not self.cached_pfa_leads:
            return
        n_main = len(self.cached_scaled_leads)
        self.plotter.plot_pfa_leads(
            self.cached_pfa_leads,
            self.ecg.pfa_lead_names,
            lead_offset=n_main + self.plotter.PFA_EXTRA_GAP_LEADS,
        )

    def on_spacing_change(self, value):
        if self.patient_file_idx == -1:
            return
        self.plotter.vertical_spacing = -round(value, 1)
        self.plotter.clear_lines()
        self._rebuild_scaled_leads()
        self._rebuild_pfa_leads()
        self.plotter.plot_ecg_leads(self.cached_scaled_leads, self.main_lead_names)
        self._plot_pfa_if_available()
        self._update_segments()

    def _update_view_start(self, earliest_annotation):
        if earliest_annotation == math.inf:
            self.view_start = 0.0
            return

        self.view_start = max(0.0, (earliest_annotation / self.ecg.fs_Hz) - 1.0)
        if (
            self.view_start + self.window_duration
            > self.ecg.ecg_mV.shape[1] / self.ecg.fs_Hz
        ):
            self.view_start = max(
                0.0, (self.ecg.ecg_mV.shape[1] / self.ecg.fs_Hz) - self.window_duration
            )

    def _switch_to_patient_by_idx(self, patient_file_idx: int = -1):
        self._reset_ecg_view()

        if patient_file_idx == -1:
            log.info("All patients annotated.")
            self._reset_annotation_state()
            return

        try:
            ecg_data = self._load_ecg_data(patient_file_idx)
            if ecg_data is None:
                return
            ecg_lead_names = list(ecg_data.lead_names)
            self._configure_leads(ecg_lead_names)
            anns = self.annotations_loader.get_annotations_for_ecg(ecg_data.file_name)
            self.annotation_session, earliest = AnnotationSession.from_annotations(
                anns,
                ecg_lead_names,
                self.lead_name_to_index,
                ecg_data,
                self.default_confidence,
            )
            self._update_view_start(earliest)

        except FileNotFoundError:
            log.error(
                "File not found: %s", self.ecg_data_loader.get_path(patient_file_idx)
            )
        except Exception as e:
            log.error("Error reading file: %s", e)
            if "ECG" in str(e):
                raise e  # re-raise for handling elsewhere

    def __init__(
        self, dicom_dir, output_file, config: AppConfig | None = None, plotter=None
    ):
        self.output_file = output_file
        self.annotations_loader = AnnotationsLoader(output_file)

        cfg = config or AppConfig()
        self.ecg: ECGRecording | None = None
        self.main_lead_names = MAIN_ECG_LEAD_NAMES
        self.lead_name_to_index = {
            lead_name: idx for idx, lead_name in enumerate(self.main_lead_names)
        }

        self.pixel_tolerance = self.PIXEL_TOLERANCE
        self.distance_threshold = cfg.distance_threshold
        self.default_confidence = cfg.default_confidence

        self.annotation_session = AnnotationSession.for_leads(
            self.main_lead_names, default_confidence=self.default_confidence
        )
        self.cached_scaled_leads = []
        self.cached_pfa_leads: list = []
        self.window_duration = 30.0
        self.view_start = 0.0

        self.ecg_data_loader = DataLoader(None)
        try:
            self.ecg_data_loader = DataLoader(dicom_dir)
            self.patient_file_idx = self.get_next_unnanotated_patient_file_idx(-1)
            self.load_patient_signals(
                self.patient_file_idx, self.get_next_unnanotated_patient_file_idx
            )
        except Exception as e:
            log.error("Error loading patient files: %s", e)
            self.patient_file_idx = -1

        self.plotter = plotter
        if plotter is not None:
            self._initialize_ui()

    def _initialize_ui(self) -> None:
        self.plotter.window_duration = self.window_duration
        self.plotter.add_ignore_checkbuttons(
            self._on_show_only_main_toggle,
            self.next_signal,
            self.prev_signal,
            list(self.main_lead_names),
            browse_callback=self.browse_for_patient,
            get_browse_dir=self._current_browse_dir,
        )
        self.plotter._add_spacing_slider(self.on_spacing_change)
        self._init_plot()
        self.plotter.add_comment_box()
        self._wire_controls_to_session()
        self._apply_session_to_controls()
        self._connect_events()

    def load_patient_signals(self, patient_start_idx, update_idx):
        self.ecg = None

        # Attempt to load the first available patient file, skipping files that fail to load.
        if not self.ecg_data_loader.patient_file_paths:
            self.patient_file_idx = -1
            return

        max_attempts = len(self.ecg_data_loader.patient_file_paths)
        attempts = 0
        idx = patient_start_idx
        while attempts < max_attempts:
            path = self.ecg_data_loader.get_path(idx)
            self._switch_to_patient_by_idx(idx)
            if self.ecg is not None:
                self.patient_file_idx = idx
                return

            log.warning("Could not load file at index %d (%s), skipping.", idx, path)
            next_idx = update_idx(idx)
            if next_idx == -1 or next_idx == self.ecg_data_loader.get_file_count():
                self.patient_file_idx = -1
                break
            idx = next_idx
            attempts += 1

        # If nothing could be loaded, mark as done
        self.patient_file_idx = -1

    def _on_show_only_main_toggle(self, state):
        if self.patient_file_idx == -1:
            return
        self.plotter.on_show_only_main_toggle(state)

    def _init_plot(self):
        """Initialises the main ECG plot."""
        self.plotter._clear_plot()

        if self.patient_file_idx == -1 or self.ecg is None:
            self.plotter.plot_no_patient_plot()
            return

        self.plotter.fs = self.ecg.fs_Hz
        self.plotter.window_duration = self.window_duration
        self._rebuild_scaled_leads()
        self._rebuild_pfa_leads()
        self.plotter.view_start = self.view_start
        self.plotter.plot_ecg_leads(self.cached_scaled_leads, self.main_lead_names)
        self._plot_pfa_if_available()
        self.plotter._init_segments(len(self.cached_scaled_leads))
        self.plotter.configure_axes(
            self.ecg.patient.id,
            self.patient_file_idx,
            self.ecg_data_loader.get_file_count(),
            len(self.cached_scaled_leads),
            pfa_lead_count=len(self.cached_pfa_leads),
        )
        self.plotter.set_pfa_available(bool(self.cached_pfa_leads))
        effective_lead_count = self.plotter.effective_lead_count(
            len(self.cached_scaled_leads), len(self.cached_pfa_leads)
        )
        max_spacing = (
            self.plotter.spacing_slider.maximum()
            if self.plotter.spacing_slider
            else 100
        )
        self.plotter.draw_ecg_grid(extend_y=effective_lead_count * max_spacing)
        self.plotter.add_hover_annotation()
        self.plotter.create_overview_axis_pg(
            self.cached_scaled_leads[0],
            self.ecg.start_hms,
            self._on_overview_region_changed,
            self._on_overview_clicked,
        )
        self.plotter.set_time_window(
            self.ecg.start_hms,
            self.cached_scaled_leads,
            start_time=self.view_start,
        )
        self._update_segments()

    def _on_overview_region_changed(self):
        self.plotter.on_overview_region_changed(
            self.ecg.start_hms, self.cached_scaled_leads
        )
        self._update_segments()

    def _on_overview_clicked(self, event):
        self.plotter.on_overview_clicked(
            event, self.ecg.start_hms, self.cached_scaled_leads
        )
        self._update_segments()

    def on_hover(self, pos) -> None:
        if self.ecg is None:
            return
        leads = self.cached_scaled_leads
        names = list(self.main_lead_names)
        if self.cached_pfa_leads and self.plotter._pfa_visible:
            leads += self.cached_pfa_leads
            names += list(self.ecg.pfa_lead_names)

        self.plotter.on_hover(pos, leads, self.ecg.start_hms, names)

    def on_key(self, direction: str) -> None:
        if self.ecg is None:
            return
        self.plotter.pan(direction, self.ecg.start_hms, self.cached_scaled_leads)
        self._update_segments()

    def _connect_events(self):
        self.plotter.wire_hover(self.on_hover)
        self.plotter.add_click_listener(self.on_click)
        self.plotter.add_keyboard_listener(self.on_key)

    def get_next_unnanotated_patient_file_idx(self, patient_file_idx=-1):
        next_idx = patient_file_idx + 1
        while next_idx < self.ecg_data_loader.get_file_count():
            patient = self.annotations_loader.get_file_annotations(
                self.ecg_data_loader.get_path(next_idx)
            )
            if patient.empty:
                return next_idx
            next_idx += 1
        return -1

    def _time_to_sample(self, time_sec):
        """Convert seconds -> nearest sample index (clipped)."""
        return time_to_sample(time_sec, self.ecg.fs_Hz, self.ecg.ecg_mV.shape[1])

    def is_close_to_line(self, x_click, y_click, lead_idx):
        """
        Check if (x_click, y_click) is within pixel tolerance of the line.
        Returns: (close: bool, x_snap: float, y_snap: float)
        """
        x_snap, y_idx = self.plotter.snap_to_line(x_click, self.ecg.ecg_mV.shape[1])
        y_snap = self.cached_scaled_leads[lead_idx][y_idx]
        dist_pixels = self.plotter.distance_to_line(x_click, y_click, x_snap, y_snap)
        return dist_pixels <= self.pixel_tolerance, x_snap, y_snap

    def find_near_point(self, x_snap, points) -> int | None:
        """Find index of a point near the click in PyQtGraph."""
        if len(points) == 0:
            return None

        pts = []
        view_start, view_end = self.plotter.get_view()
        for point_id, (x_sec, _) in enumerate(points):
            if not (view_start <= x_sec <= view_end):
                continue  # skip points outside current window
            x_idx = self._time_to_sample(x_sec)
            pts.append((x_idx, point_id))
        if len(pts) == 0:
            return None

        pts = np.array(pts, dtype=float)  # shape (m,2)
        x_idx = self._time_to_sample(x_snap)
        # Compute distance in pixels
        x_sec_dists = np.abs(pts[:, 0] - x_idx)

        idx_min = int(np.argmin(x_sec_dists))
        if x_sec_dists[idx_min] <= 10:
            return int(pts[idx_min][1])
        return None

    def _add_point(self, button_str: str, lead_idx: int, x: float, y: float) -> None:
        lead_state = self.annotation_session.lead_states[lead_idx]
        if button_str == "shift_left":
            lead_state.blue_points.append((x, y))
        elif button_str == "left":
            lead_state.red_points.append((x, y))
        elif button_str == "right":
            lead_state.green_points.append((x, y))

    # ---------- Event handlers ----------
    def on_click(self, x_click: float, y_click: float, button_str: str) -> None:
        """Handle adding/removing red/green points on ECG."""
        if self.patient_file_idx == -1:
            return
        view_start, view_end = self.plotter.get_view()
        if x_click < view_start or x_click > view_end:
            return

        # Find closest lead
        lead_idx = np.argmin(
            [
                abs(
                    y_click - self.cached_scaled_leads[i][self._time_to_sample(x_click)]
                )
                for i in range(len(self.cached_scaled_leads))
            ]
        )

        if self.plotter.get_ignored_leads()[lead_idx]:
            return

        # Snap to nearest point on line
        close, x_snap, y_snap = self.is_close_to_line(x_click, y_click, lead_idx)
        if not close:
            return

        # --- Remove points if near ---
        lead_state = self.annotation_session.lead_states[lead_idx]
        for pts in (
            lead_state.red_points,
            lead_state.green_points,
            lead_state.blue_points,
        ):
            idx = self.find_near_point(x_snap, pts)
            if idx is not None:
                pts.pop(idx)
                self._update_segments()
                return

        self._add_point(button_str, lead_idx, x_snap, y_snap)

        self._update_segments()

    # ---------- Saving / Next ----------
    def validate_points(self):
        """Validate red/green points before saving."""
        validate_annotation_session(
            self.annotation_session,
            self.ecg.start_hms,
            self.distance_threshold,
        )

    def save_current_patient(self):
        patient_file = self.ecg_data_loader.get_path(self.patient_file_idx)
        rows_to_add = build_annotation_rows(
            self.annotation_session,
            patient_id=self.ecg.patient.id,
            patient_file=patient_file,
            time_to_sample=self._time_to_sample,
        )
        self.annotations_loader.replace_file_annotations(patient_file, rows_to_add)
        self.annotations_loader.save(self.output_file)
        log.info(
            "Saved annotations for patient %s to %s",
            self.ecg.patient.id,
            self.output_file,
        )

    def save(self, idx, update_idx):
        # Save current patient
        self.save_current_patient()
        self._clear_and_load_patient_by_idx(idx, update_idx)

    def _clear_and_load_patient_by_idx(self, idx, update_idx):
        self.view_start = 0.0
        self.plotter.reset_controls()

        x_range, y_range = self.plotter.get_view_range()

        self.patient_file_idx = idx
        self.ecg = None
        self.load_patient_signals(idx, update_idx)
        self._wire_lead_controls()
        self._apply_session_to_controls()

        self._init_plot()

        self.plotter.restore_view_range(x_range, y_range)
        self.plotter.repaint()

    def prev_signal(self, event):
        prev_patient_file_idx = self.patient_file_idx - 1
        if prev_patient_file_idx == -2:
            prev_patient_file_idx = (
                self.ecg_data_loader.get_file_count() - 1
            )  # Stay at first file

        self.validate_and_save(prev_patient_file_idx, lambda x: x - 1)

    def next_signal(self, event):
        next_patient_file_idx = self.patient_file_idx + 1
        if next_patient_file_idx >= self.ecg_data_loader.get_file_count():
            next_patient_file_idx = -1  # No more files
        self.validate_and_save(next_patient_file_idx, lambda x: x + 1)

    def browse_for_patient(self, directory: str) -> None:
        """Change the working directory to `directory`, chosen via the browse
        dialog, and load its first unannotated patient."""
        if self.plotter.is_popup_visible():
            return

        if self.patient_file_idx == -1:
            self._switch_directory_and_load(directory)
            return

        try:
            self.validate_points()
        except ValueError as ve:
            self.plotter.show_popup(
                str(ve),
                directory,
                None,
                self._save_anyway_and_switch_directory,
                self.annotation_session.annotation_type,
            )
            return

        self.save_current_patient()
        self._switch_directory_and_load(directory)

    def _save_anyway_and_switch_directory(
        self, directory: str, _update_idx=None
    ) -> None:
        self.save_current_patient()
        self._switch_directory_and_load(directory)
        self.plotter.close_popup()

    def _switch_directory_and_load(self, directory: str) -> None:
        self.ecg_data_loader = DataLoader(directory)
        self._clear_and_load_patient_by_idx(
            self.get_next_unnanotated_patient_file_idx(-1),
            self.get_next_unnanotated_patient_file_idx,
        )

    def _current_browse_dir(self) -> str:
        current_path = self.ecg_data_loader.get_path(self.patient_file_idx)
        if current_path:
            return str(Path(current_path).parent)
        if self.ecg_data_loader.patient_file_paths:
            return str(Path(self.ecg_data_loader.patient_file_paths[0]).parent)
        return str(Path.home())

    def validate_and_save(self, next_patient_file_idx, update_idx):
        if self.patient_file_idx == -1:
            self._clear_and_load_patient_by_idx(next_patient_file_idx, update_idx)
            return
        if self.plotter.is_popup_visible():
            return

        log.debug("Moving to next file")
        try:
            self.validate_points()
        except ValueError as ve:
            self.plotter.show_popup(
                str(ve),
                next_patient_file_idx,
                update_idx,
                self.save_anyway,
                self.annotation_session.annotation_type,
            )
            return

        self.save(next_patient_file_idx, update_idx)

    # ---------- Popup ----------
    def save_anyway(self, next_patient_file_idx, update_idx=None):
        self.save(next_patient_file_idx, update_idx)
        self.plotter.close_popup()
