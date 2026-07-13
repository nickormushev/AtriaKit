from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QGraphicsDropShadowEffect,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ecg_annotator.utils.time import sec_to_hms, time_to_sample

LeadSignals = Sequence[np.ndarray]


@dataclass(frozen=True)
class WindowSlice:
    start_time: float
    end_time: float
    indices: np.ndarray
    x_values: np.ndarray


def _snap_to_step(value: float, step: float) -> float:
    return float(round(round(value / step) * step, 6))


def _build_parallel_line_data(
    values: np.ndarray,
    start: float,
    end: float,
    *,
    vertical: bool,
) -> tuple[list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []

    for value in values:
        if vertical:
            x_values.extend([value, value, np.nan])
            y_values.extend([start, end, np.nan])
        else:
            x_values.extend([start, end, np.nan])
            y_values.extend([value, value, np.nan])

    return x_values, y_values


class ECGPlotter(QtCore.QObject):
    """
    Handles all plotting and UI elements of ECGAnnotator.
    The annotator delegates visualization logic here.
    """

    DEFAULT_FS = 2000
    DEFAULT_WINDOW_DURATION = 30.0
    DEFAULT_SEGMENT_SIZE = 2.0
    DEFAULT_VERTICAL_SPACING = -5
    OVERVIEW_MAX_POINTS = 2000
    WINDOW_PAN_STEP_SEC = 1.0
    WINDOW_PAN_FRACTION_Y = 0.1
    LABEL_X_OFFSET = 0.3
    PFA_EXTRA_GAP_LEADS = 1.5

    PFA_PEN = pg.mkPen(color=(80, 80, 80), width=0.8)

    CONFIDENCE_LEVELS = ("Low", "Med", "High")
    ANNOTATION_TYPES = ("Before", "After", "Both")

    LEAD_PEN_ACTIVE = pg.mkPen(color="k", width=1)
    LEAD_PEN_IGNORED = pg.mkPen(
        color=(150, 150, 150),
        width=1.0,
        style=Qt.PenStyle.DashLine,
    )
    SEGMENT_PENS = {
        "red": pg.mkPen(color="#e74c3c", width=3.0, style=Qt.PenStyle.DashLine),
        "green": pg.mkPen(color="#16a085", width=3.0, style=Qt.PenStyle.DashLine),
        "blue": pg.mkPen(color="#3498db", width=3.0, style=Qt.PenStyle.DashLine),
    }

    NO_PATIENT_MESSAGE = (
        "No DICOM files found in the selected directory, or all patients have been annotated.\n"
        "Please select a directory containing DICOM (.IMA, .dcm) files."
    )

    def __init__(
        self,
        parent_widget=None,
        time_sec=None,
        amplitude_scale: float = 10.0,
        plot=None,
    ):
        super().__init__()
        self.parent_widget = parent_widget
        self.amplitude_scale = amplitude_scale
        self.vertical_spacing: float = self.DEFAULT_VERTICAL_SPACING

        self.plot = plot
        self.overview_plot = None
        self.overview_window = None
        self.popup_widget = None

        self.red_segments: list[pg.PlotDataItem] = []
        self.green_segments: list[pg.PlotDataItem] = []
        self.blue_segments: list[pg.PlotDataItem] = []

        self.pfa_lines: list[pg.PlotDataItem] = []
        self.pfa_lead_labels: list[pg.TextItem] = []
        self._cached_pfa_leads: list[np.ndarray] = []
        self._pfa_visible: bool = True

        self.spacing_slider: QSlider | None = None
        self.hover_annot: pg.TextItem | None = None
        self.comment_box: QLineEdit | None = None

        self.lines: list[pg.PlotDataItem] = []
        self.lead_labels: list[pg.TextItem] = []

        self._grid_v_minor: pg.PlotDataItem | None = None
        self._grid_v_major: pg.PlotDataItem | None = None
        self._grid_h_minor: pg.PlotDataItem | None = None
        self._grid_h_major: pg.PlotDataItem | None = None

        self.ignore_widget: QWidget | None = None
        self.ignore_all_checkbox: QCheckBox | None = None
        self.show_only_main_checkbox: QCheckBox | None = None
        self._lead_rows_layout: QVBoxLayout | None = None
        self.ignore_checks: list[QCheckBox] = []
        self.radio_buttons: list[QButtonGroup] = []
        self.type_group: QButtonGroup | None = None
        self.type_buttons: list[QRadioButton] = []

        self.main_layout: QHBoxLayout | None = None
        self.main_container: QWidget | None = None

        self._on_key_cb = None

        self.view_start = 0.0
        self.view_end = self.DEFAULT_WINDOW_DURATION
        self.segment_size = self.DEFAULT_SEGMENT_SIZE

        self.time_sec = np.asarray(
            time_sec if time_sec is not None else [],
            dtype=float,
        )
        self.fs = self.DEFAULT_FS
        self.window_duration = self.DEFAULT_WINDOW_DURATION
        self._cached_indices = np.arange(len(self.time_sec))

    def scale_lead(self, lead_data: np.ndarray, lead_idx: int) -> np.ndarray:
        """Apply amplitude scale and vertical offset to a raw lead signal."""
        return lead_data * self.amplitude_scale + lead_idx * self.vertical_spacing

    def scale_pfa_lead(self, pfa_data: np.ndarray, pfa_idx: int, main_lead_count: int) -> np.ndarray:
        """Position a PFA lead below the main leads with an extra visual gap, no amplitude scaling."""
        pfa_offset = (main_lead_count + self.PFA_EXTRA_GAP_LEADS) * self.vertical_spacing
        return pfa_data + pfa_idx * self.vertical_spacing + pfa_offset

    def _has_time_axis(self) -> bool:
        return self.time_sec.size > 0

    def _update_time_axis(self, time_sec: np.ndarray) -> None:
        self.time_sec = np.asarray(time_sec, dtype=float)
        self._cached_indices = np.arange(len(self.time_sec))

    def _lead_offset(self, lead_idx: int, vertical_spacing: float) -> float:
        return lead_idx * vertical_spacing

    def _lead_pen(self, lead_idx: int, ignored_leads: Sequence[bool]) -> pg.QtGui.QPen:
        if lead_idx < len(ignored_leads) and ignored_leads[lead_idx]:
            return self.LEAD_PEN_IGNORED
        return self.LEAD_PEN_ACTIVE

    def _update_line_visibility(self) -> None:
        for curve in self.lines:
            curve.setVisible(True)
        for item in self.pfa_lines + self.pfa_lead_labels:
            item.setVisible(self._pfa_visible)

    def _clamp_window_start(self, start_time: float) -> float:
        if not self._has_time_axis():
            return 0.0

        lower_bound = float(self.time_sec[0])
        upper_bound = max(lower_bound, float(self.time_sec[-1]) - self.window_duration)
        return float(max(lower_bound, min(start_time, upper_bound)))

    def _window_slice(self, start_time: float) -> WindowSlice:
        start_time = self._clamp_window_start(start_time)
        end_time = start_time + self.window_duration

        start_idx = int(np.searchsorted(self.time_sec, start_time, side="left"))
        end_idx = int(np.searchsorted(self.time_sec, end_time, side="right"))
        indices = self._cached_indices[start_idx:end_idx]
        x_values = self.time_sec[indices] - start_time

        return WindowSlice(
            start_time=start_time,
            end_time=end_time,
            indices=indices,
            x_values=x_values,
        )

    def _window_ticks(
        self,
        window: WindowSlice,
        content_start_time,
    ) -> list[tuple[float, str]]:
        tick_count = max(int(self.window_duration), 2)
        tick_positions = np.linspace(window.start_time, window.end_time, tick_count)
        return [
            (time_point - window.start_time, sec_to_hms(time_point, content_start_time))
            for time_point in tick_positions
        ]

    def _segment_items(
        self,
    ) -> tuple[list[pg.PlotDataItem], list[pg.PlotDataItem], list[pg.PlotDataItem]]:
        return self.red_segments, self.green_segments, self.blue_segments

    def _assert_segments_initialized(self) -> None:
        n = len(self.red_segments)
        if n == 0 or len(self.green_segments) != n or len(self.blue_segments) != n:
            raise RuntimeError(
                "Annotation segments are not initialized. "
                "Call `_init_segments()` before updating segments."
            )

    def _create_segment_item(self, color_name: str) -> pg.PlotDataItem:
        return pg.PlotDataItem([], [], pen=self.SEGMENT_PENS[color_name])

    def _plot_label_x(self, x_values: np.ndarray) -> float:
        if x_values.size == 0:
            return -self.LABEL_X_OFFSET
        return float(x_values[0]) - self.LABEL_X_OFFSET

    def _build_lead_label(
        self, lead_name: str, label_x: float, label_y: float
    ) -> pg.TextItem:
        label = pg.TextItem(
            text=lead_name,
            color="w",
            anchor=(1, 0.5),
            border="k",
            fill=(50, 50, 50, 150),
        )
        label.setPos(label_x, label_y)
        return label

    def _plot_single_lead(
        self,
        lead_idx: int,
        lead_name: str,
        lead: np.ndarray,
        window: WindowSlice,
        vertical_spacing: float,
        ignored_leads: Sequence[bool],
    ) -> None:
        line = self.plot.plot(
            window.x_values,
            lead[window.indices],
            pen=self._lead_pen(lead_idx, ignored_leads),
            name=lead_name,
            autoDownsample=True,
        )
        self.lines.append(line)

        label = self._build_lead_label(
            lead_name,
            self._plot_label_x(window.x_values),
            self._lead_offset(lead_idx, vertical_spacing),
        )
        self.plot.addItem(label)
        self.lead_labels.append(label)

    def _build_lead_controls_header(self, layout_main: QVBoxLayout) -> None:
        header_layout = QHBoxLayout()
        lbl_ignore = QLabel("Ignore")
        lbl_conf = QLabel("Confidence")
        lbl_ignore.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_conf.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(lbl_ignore)
        header_layout.addWidget(lbl_conf)
        header_layout.setStretch(0, 1)
        header_layout.setStretch(1, 1)
        layout_main.addLayout(header_layout)

    def _build_global_controls(
        self,
        layout_main: QVBoxLayout,
        on_show_only_main_toggle,
    ) -> None:
        checkbox_layout = QHBoxLayout()

        self.ignore_all_checkbox = QCheckBox("Ignore All")
        self.ignore_all_checkbox.toggled.connect(self.on_ignore_all_toggle)
        checkbox_layout.addWidget(self.ignore_all_checkbox)

        self.show_only_main_checkbox = QCheckBox("Hide PFA")
        self.show_only_main_checkbox.setChecked(False)
        self.show_only_main_checkbox.setEnabled(False)  # enabled by set_pfa_available when leads exist
        self.show_only_main_checkbox.stateChanged.connect(on_show_only_main_toggle)
        checkbox_layout.addWidget(self.show_only_main_checkbox)

        layout_main.addLayout(checkbox_layout)

    def _build_lead_controls_row(self, lead_idx: int, lead_name: str) -> QWidget:
        row_widget = QWidget()
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(2)
        row_widget.setLayout(row_layout)

        checkbox = QCheckBox(lead_name)
        checkbox.setChecked(False)
        checkbox.toggled.connect(
            lambda checked, idx=lead_idx: self.toggle_ignore_lead(checked, idx)
        )
        self.ignore_checks.append(checkbox)
        row_layout.addWidget(checkbox)

        radio_group = QButtonGroup(row_widget)
        for button_idx, level in enumerate(self.CONFIDENCE_LEVELS):
            radio_button = QRadioButton(level)
            row_layout.addWidget(radio_button)
            radio_group.addButton(radio_button, button_idx)
            if level == "High":
                radio_button.setChecked(True)

        self.radio_buttons.append(radio_group)
        return row_widget

    def _build_annotation_type_group(self) -> QGroupBox:
        group_box = QGroupBox("Annotation Type")
        layout = QHBoxLayout()

        self.type_buttons = []
        self.type_group = QButtonGroup(self)
        self.type_group.setExclusive(True)

        for label in self.ANNOTATION_TYPES:
            button = QRadioButton(label)
            layout.addWidget(button)
            self.type_group.addButton(button)
            self.type_buttons.append(button)

        group_box.setLayout(layout)
        group_box.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        return group_box

    def _navigation_button(self, label: str, callback) -> QPushButton:
        button = QPushButton(label)
        button.setFixedHeight(25)
        button.clicked.connect(callback)
        return button

    def _build_navigation_controls(
        self,
        next_signal_callback,
        prev_signal_callback,
        browse_callback,
        get_browse_dir,
    ) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.addWidget(self._navigation_button("Previous", prev_signal_callback))
        layout.addWidget(
            self._navigation_button(
                "Browse...",
                lambda: self._on_browse_clicked(browse_callback, get_browse_dir),
            )
        )

        next_button = self._navigation_button("Save and Next", next_signal_callback)
        next_button.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Fixed,
        )
        layout.addWidget(next_button)
        return layout

    def _on_browse_clicked(self, browse_callback, get_browse_dir) -> None:
        """Open a directory picker and forward the chosen directory to the callback.

        Uses Qt's own dialog (not the native OS one) so patient files are
        listed for context while browsing, even though only a directory can
        ultimately be selected.
        """
        dialog = QFileDialog(self.parent_widget, "Browse for directory")
        dialog.setDirectory(get_browse_dir())
        dialog.setFileMode(QFileDialog.FileMode.Directory)
        dialog.setOption(QFileDialog.Option.ShowDirsOnly, False)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)

        if dialog.exec() != QFileDialog.DialogCode.Accepted:
            return

        selected = dialog.selectedFiles()
        if selected:
            browse_callback(selected[0])

    def _build_popup_button(
        self,
        label: str,
        base_color: str,
        hover_color: str,
        callback,
    ) -> QPushButton:
        button = QPushButton(label)
        button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {base_color};
                color: white;
                font-weight: bold;
                border-radius: 5px;
                padding: 6px 12px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
        """
        )
        button.clicked.connect(callback)
        return button

    def _init_segments(self, lead_count: int):
        """Initialize red/green/blue annotation segment items."""
        self.red_segments = []
        self.green_segments = []
        self.blue_segments = []

        for segment_group, color_name in zip(
            self._segment_items(),
            ("red", "green", "blue"),
        ):
            for _ in range(lead_count):
                segment_item = self._create_segment_item(color_name)
                self.plot.addItem(segment_item)
                segment_group.append(segment_item)

    def _clear_plot(self):
        """Clear plot items and reset cached plotting state."""
        if self.plot is not None:
            self.plot.clear()

        self.lines.clear()
        self.pfa_lines.clear()
        self.pfa_lead_labels.clear()
        self._cached_pfa_leads = []
        self.lead_labels.clear()
        self.red_segments.clear()
        self.green_segments.clear()
        self.blue_segments.clear()
        self.hover_annot = None
        if self.overview_plot is not None:
            self.parent_widget.removeItem(self.overview_plot)
            self.overview_plot = None
        self.overview_window = None

    def _add_spacing_slider(self, on_spacing_change):
        if self.main_layout is None:
            raise RuntimeError(
                "Spacing slider requires the main layout to be initialized."
            )

        self.spacing_slider = QSlider()
        self.spacing_slider.setRange(0, 100)
        self.spacing_slider.setValue(-int(self.vertical_spacing))
        self.spacing_slider.setTickInterval(10)

        label = QLabel("Vertical\nSpacing")
        spacing_layout = QVBoxLayout()
        spacing_layout.addWidget(label, alignment=Qt.AlignmentFlag.AlignHCenter)
        spacing_layout.addWidget(
            self.spacing_slider,
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )
        self.main_layout.insertLayout(0, spacing_layout)
        self.spacing_slider.valueChanged.connect(on_spacing_change)

    def _get_segments(self, lead_idx, points_list, leads):
        """
        Convert annotation points to vertical segments in the current window.
        Returns x and y arrays suitable for PlotDataItem.setData(connect='finite').
        """
        seg_x: list[float] = []
        seg_y: list[float] = []
        scaled_lead = leads[lead_idx]

        for x_value, _ in points_list[lead_idx]:
            if not (self.view_start <= x_value <= self.view_end):
                continue

            sample_idx = time_to_sample(x_value, self.fs, len(scaled_lead))
            y_center = scaled_lead[sample_idx]
            seg_x.extend([x_value - self.view_start, x_value - self.view_start, np.nan])
            seg_y.extend(
                [y_center - self.segment_size, y_center + self.segment_size, np.nan]
            )

        return np.asarray(seg_x, dtype=float), np.asarray(seg_y, dtype=float)

    def update_segments(self, red_points, green_points, blue_points, leads):
        """Draw all annotation segments at once."""
        self._assert_segments_initialized()

        empty = np.asarray([], dtype=float)
        segment_sets = (
            (red_points, self.red_segments),
            (green_points, self.green_segments),
            (blue_points, self.blue_segments),
        )

        for lead_idx in range(len(self.red_segments)):
            for points, segments in segment_sets:
                if (
                    not points
                    or lead_idx >= len(points)
                    or lead_idx >= len(segments)
                    or not points[lead_idx]
                ):
                    if lead_idx < len(segments):
                        segments[lead_idx].setData(empty, empty)
                    continue

                x_values, y_values = self._get_segments(lead_idx, points, leads)
                segments[lead_idx].setData(x_values, y_values, connect="finite")

    def effective_lead_count(self, lead_count: int, pfa_lead_count: int = 0) -> float:
        """Total vertical lead slots to plan for, including the PFA gap and leads."""
        if pfa_lead_count > 0:
            return lead_count + self.PFA_EXTRA_GAP_LEADS + pfa_lead_count
        return lead_count

    def configure_axes(self, patient_id, file_idx, file_count, lead_count: int, pfa_lead_count: int = 0):
        """Configure the main ECG plot axes and title."""
        pg.setConfigOptions(useOpenGL=True)

        self.plot.setLabel("bottom", "Time (HH:MM:SS)")
        self.plot.setTitle(
            f"ECG of patient {patient_id} File: {file_idx + 1}/{file_count}\n"
            "Left click=Red (Onset), Right click=Green (Offset), "
            "Shift + Left click=Blue(QRS onset), Click again=Remove"
        )
        self.plot.getAxis("left").setTicks([])

        effective_count = self.effective_lead_count(lead_count, pfa_lead_count)
        y_min = 0
        y_max = self.vertical_spacing * effective_count - 40
        self.plot.setYRange(y_min, y_max, padding=0)
        self.plot.setXRange(0, self.window_duration, padding=0)

    def add_hover_annotation(self):
        """Create the tooltip item shown while hovering over the signal."""
        self.hover_annot = pg.TextItem(
            "",
            anchor=(0, 1),
            border="w",
            fill=(50, 50, 50, 150),
        )
        self.plot.addItem(self.hover_annot)
        self.hover_annot.hide()

    def on_hover(self, pos, leads, start_time, lead_names: list[str] | None = None):
        if not leads or self.hover_annot is None:
            return

        vb = self.plot.getViewBox()
        if not vb.sceneBoundingRect().contains(pos):
            self.hover_annot.hide()
            return

        mouse_point = vb.mapSceneToView(pos)
        x_data = mouse_point.x()
        y_data = mouse_point.y()

        x_global = x_data + self.view_start
        if not (self.view_start <= x_global <= self.view_end):
            self.hover_annot.hide()
            return

        sample_idx = time_to_sample(x_global, self.fs, len(leads[0]))
        lead_idx = int(np.argmin([abs(y_data - lead[sample_idx]) for lead in leads]))

        x_snap = self.time_sec[sample_idx] - self.view_start
        y_snap = leads[lead_idx][sample_idx]
        hms_time = sec_to_hms(x_snap + self.view_start, start_time, True)

        name = lead_names[lead_idx] if lead_names and lead_idx < len(lead_names) else f"Lead {lead_idx + 1}"
        self.hover_annot.setText(f"Lead {name}\ntime={hms_time}")
        self.hover_annot.setPos(x_snap, y_snap)
        self.hover_annot.show()

    def toggle_ignore_lead(self, checked, lead_idx):
        """Toggle ignoring a lead by updating its line appearance."""
        if lead_idx >= len(self.lines):
            return

        curve = self.lines[lead_idx]
        curve.setPen(self.LEAD_PEN_IGNORED if checked else self.LEAD_PEN_ACTIVE)

    def on_overview_region_changed(self, content_start_time, leads):
        """Update the main plot when the overview region is dragged."""
        if self.overview_window is None or not self._has_time_axis():
            return

        region = self.overview_window.getRegion()
        start_time = self._clamp_window_start(region[0])
        self.set_time_window(content_start_time, leads, start_time=start_time)
        self.overview_window.blockSignals(True)
        self.overview_window.setRegion((start_time, start_time + self.window_duration))
        self.overview_window.blockSignals(False)

    def add_keyboard_listener(self, callback) -> None:
        """Enable arrow-key navigation. Calls callback(direction) where direction is 'left'/'right'/'up'/'down'."""
        self._on_key_cb = callback
        self.parent_widget.installEventFilter(self)
        self.parent_widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.parent_widget.setFocus()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.KeyPress:
            key_map = {
                Qt.Key.Key_Left: "left",
                Qt.Key.Key_Right: "right",
                Qt.Key.Key_Up: "up",
                Qt.Key.Key_Down: "down",
            }
            direction = key_map.get(event.key())
            if direction is not None and self._on_key_cb is not None:
                self._on_key_cb(direction)
                return True
        return super().eventFilter(obj, event)

    def pan(self, direction: str, content_start_time, leads) -> None:
        """Pan the ECG view. direction is 'left'/'right'/'up'/'down'."""
        vb = self.plot.getViewBox()
        _, y_range = vb.viewRange()
        y_span = y_range[1] - y_range[0]

        def move_time_window(delta: float) -> None:
            new_start = self._clamp_window_start(self.view_start + delta)
            if new_start == self.view_start:
                return
            if self.overview_window is not None:
                self.overview_window.blockSignals(True)
                self.overview_window.setRegion(
                    (new_start, new_start + self.window_duration)
                )
                self.overview_window.blockSignals(False)
            self.set_time_window(content_start_time, leads, new_start)

        if direction == "right":
            move_time_window(self.WINDOW_PAN_STEP_SEC)
        elif direction == "left":
            move_time_window(-self.WINDOW_PAN_STEP_SEC)
        elif direction == "up":
            vb.setYRange(
                y_range[0] + self.WINDOW_PAN_FRACTION_Y * y_span,
                y_range[1] + self.WINDOW_PAN_FRACTION_Y * y_span,
                padding=0,
            )
        elif direction == "down":
            vb.setYRange(
                y_range[0] - self.WINDOW_PAN_FRACTION_Y * y_span,
                y_range[1] - self.WINDOW_PAN_FRACTION_Y * y_span,
                padding=0,
            )

    def on_key(self, event, content_start_time, leads):
        """Pan ECG view with arrow keys (legacy entry point — delegates to pan)."""
        key_map = {
            Qt.Key.Key_Left: "left",
            Qt.Key.Key_Right: "right",
            Qt.Key.Key_Up: "up",
            Qt.Key.Key_Down: "down",
        }
        direction = key_map.get(event.key())
        if direction is None:
            return
        self.pan(direction, content_start_time, leads)
        event.accept()

    def set_time_window(self, content_start_time, leads, start_time=None):
        """Update ECG traces for the current fixed-duration time window."""
        if not self._has_time_axis():
            raise RuntimeError("Cannot update time window without a valid time axis.")
        if not self.lines:
            raise RuntimeError(
                "Cannot update time window before ECG lines are plotted."
            )

        if start_time is None:
            start_time = self.view_start

        window = self._window_slice(start_time)
        self.view_start = window.start_time
        self.view_end = window.end_time

        for curve, lead in zip(self.lines + self.pfa_lines, list(leads) + self._cached_pfa_leads):
            curve.setData(window.x_values, lead[window.indices], connect="finite")

        self._update_line_visibility()
        self.plot.getAxis("bottom").setTicks(
            [self._window_ticks(window, content_start_time)]
        )

    def on_overview_clicked(self, event, content_start_time, leads):
        """Center the overview window around a click on the overview plot."""
        if self.overview_plot is None or not self._has_time_axis():
            return

        vb = self.overview_plot.getViewBox()
        if not vb.sceneBoundingRect().contains(event.scenePos()):
            return

        mouse_point = vb.mapSceneToView(event.scenePos())
        x_value = mouse_point.x()
        if not (self.time_sec[0] <= x_value <= self.time_sec[-1]):
            return

        start_time = self._clamp_window_start(x_value - self.window_duration / 2)
        self.overview_window.blockSignals(True)
        self.overview_window.setRegion((start_time, start_time + self.window_duration))
        self.overview_window.blockSignals(False)
        self.set_time_window(content_start_time, leads, start_time=start_time)

    def draw_ecg_grid(self, extend_x=1.0, extend_y=200.0):
        """Draw the ECG grid behind the visible plot area."""
        small_box_sec, large_box_sec = 0.04, 0.20
        small_box_mv, large_box_mv = 0.1, 0.5

        amp_scale = self.amplitude_scale
        x_range, y_range = self.plot.viewRange()
        x_min, x_max = x_range
        y_min, y_max = y_range

        x_min -= extend_x
        x_max += extend_x
        y_min -= extend_y
        y_max += extend_y

        # Minor vertical lines (every small box)
        minor_v_ticks = np.arange(
            small_box_sec * np.floor(x_min / small_box_sec),
            small_box_sec * np.ceil(x_max / small_box_sec) + small_box_sec,
            small_box_sec,
        )
        x_v_minor, y_v_minor = _build_parallel_line_data(minor_v_ticks, y_min, y_max, vertical=True)
        self._grid_v_minor = pg.PlotDataItem(x_v_minor, y_v_minor, pen=pg.mkPen(color="#ffcccc", width=0.5))
        self._grid_v_minor.setZValue(-100)
        self.plot.addItem(self._grid_v_minor)

        # Major vertical lines (every large box)
        major_v_ticks = np.arange(
            large_box_sec * np.floor(x_min / large_box_sec),
            large_box_sec * np.ceil(x_max / large_box_sec) + large_box_sec,
            large_box_sec,
        )
        x_v_major, y_v_major = _build_parallel_line_data(major_v_ticks, y_min, y_max, vertical=True)
        self._grid_v_major = pg.PlotDataItem(x_v_major, y_v_major, pen=pg.mkPen(color="#ffcccc", width=0.5))
        self._grid_v_major.setZValue(-99)
        self.plot.addItem(self._grid_v_major)

        minor_h_ticks = (
            np.arange(
                _snap_to_step(y_min / amp_scale, small_box_mv),
                _snap_to_step(y_max / amp_scale, small_box_mv) + small_box_mv,
                small_box_mv,
            )
            * amp_scale
        )
        major_h_ticks = (
            np.arange(
                _snap_to_step(y_min / amp_scale, large_box_mv),
                _snap_to_step(y_max / amp_scale, large_box_mv) + large_box_mv,
                large_box_mv,
            )
            * amp_scale
        )

        x_h_minor, y_h_minor = _build_parallel_line_data(minor_h_ticks, x_min, x_max, vertical=False)
        self._grid_h_minor = pg.PlotDataItem(x_h_minor, y_h_minor, pen=pg.mkPen(color="#ffcccc", width=0.5))
        self._grid_h_minor.setZValue(-98)
        self.plot.addItem(self._grid_h_minor)

        x_h_major, y_h_major = _build_parallel_line_data(major_h_ticks, x_min, x_max, vertical=False)
        self._grid_h_major = pg.PlotDataItem(x_h_major, y_h_major, pen=pg.mkPen(color="#cc0000", width=0.5))
        self._grid_h_major.setZValue(-97)
        self.plot.addItem(self._grid_h_major)

        try:
            self.plot.getViewBox().sigXRangeChanged.disconnect(self._update_grid_visibility)
        except (RuntimeError, TypeError):
            pass
        self.plot.getViewBox().sigXRangeChanged.connect(self._update_grid_visibility)
        self._update_grid_visibility()

    def _update_grid_visibility(self, *_):
        vb = self.plot.getViewBox()
        view_width_px = vb.width()
        x_span = self.plot.viewRange()[0]
        x_span = x_span[1] - x_span[0]
        if x_span <= 0 or view_width_px <= 0:
            return
        px_per_small_box = view_width_px * 0.04 / x_span
        show_minor = px_per_small_box >= 12
        if self._grid_v_minor is not None:
            self._grid_v_minor.setVisible(show_minor)
        if self._grid_h_minor is not None:
            self._grid_h_minor.setVisible(show_minor)

    def get_comment(self):
        return self.comment_box.text() if self.comment_box is not None else ""

    def set_comment(self, comment):
        if self.comment_box is not None:
            self.comment_box.setText(comment or "")

    def get_selected_annotation_type(self):
        for radio_button in self.type_buttons:
            if radio_button.isChecked():
                return radio_button.text()
        return None

    def set_selected_annotation_type(self, annotation_type):
        if self.type_group is None:
            return

        self.type_group.setExclusive(False)
        for radio_button in self.type_buttons:
            radio_button.setChecked(radio_button.text() == annotation_type)
        self.type_group.setExclusive(True)

    def clear_annotation_type(self):
        self.set_selected_annotation_type(None)

    def set_lead_confidence(self, lead_idx, confidence):
        buttons = self.radio_buttons[lead_idx].buttons()
        for button in buttons:
            button.setChecked(button.text() == confidence)

    def set_confidence_levels(self, confidences):
        for lead_idx, confidence in enumerate(confidences):
            self.set_lead_confidence(lead_idx, confidence)

    def reset_confidence_levels(self, default="High"):
        self.set_confidence_levels([default] * len(self.radio_buttons))

    def set_ignored_leads(self, ignored_leads):
        for checkbox, ignored in zip(self.ignore_checks, ignored_leads):
            checkbox.setChecked(ignored)

    def reset_ignored_leads(self):
        self.set_ignored_leads([False] * len(self.ignore_checks))

    def is_show_only_main_checked(self):
        return (
            self.show_only_main_checkbox.isChecked()
            if self.show_only_main_checkbox is not None
            else False
        )

    def clear_lines(self):
        """Remove only the signal lines and lead labels from the plot."""
        if self.plot is None:
            self.lines = []
            self.pfa_lines = []
            self.lead_labels = []
            return

        all_items = self.lines + self.pfa_lines + self.pfa_lead_labels + self.lead_labels
        for item in all_items:
            self.plot.removeItem(item)
        self.lines = []
        self.pfa_lines = []
        self.pfa_lead_labels = []
        self.lead_labels = []

    def reset_controls(self):
        self.clear_comment()
        self.reset_ignored_leads()
        self.reset_confidence_levels()
        self.clear_annotation_type()

    def on_ignore_all_toggle(self, checked):
        for checkbox in self.ignore_checks:
            checkbox.setChecked(checked)

    def add_ignore_checkbuttons(
        self,
        on_show_only_main_toggle,
        next_signal_callback,
        prev_signal_callback,
        lead_names: list[str],
        browse_callback,
        get_browse_dir,
    ):
        """Add ignore/confidence controls next to the ECG plot."""
        self.ignore_widget = QWidget()
        layout_main = QVBoxLayout()
        layout_main.setSpacing(2)
        self.ignore_widget.setLayout(layout_main)

        self.ignore_checks = []
        self.radio_buttons = []

        self._build_lead_controls_header(layout_main)
        self._build_global_controls(layout_main, on_show_only_main_toggle)

        self._lead_rows_layout = QVBoxLayout()
        self._lead_rows_layout.setSpacing(2)
        self._lead_rows_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        lead_rows_container = QWidget()
        lead_rows_container.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        lead_rows_container.setLayout(self._lead_rows_layout)
        layout_main.addWidget(lead_rows_container)
        for lead_idx, lead_name in enumerate(lead_names):
            self._lead_rows_layout.addWidget(self._build_lead_controls_row(lead_idx, lead_name))

        layout_main.addWidget(self._build_annotation_type_group())
        layout_main.addLayout(
            self._build_navigation_controls(
                next_signal_callback,
                prev_signal_callback,
                browse_callback,
                get_browse_dir,
            )
        )

        self.ignore_widget.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Expanding,
        )

        self.main_layout = QHBoxLayout()
        self.main_container = QWidget()
        self.main_container.setLayout(self.main_layout)
        self.main_layout.addWidget(self.parent_widget)
        self.main_layout.addWidget(self.ignore_widget)
        self.main_container.show()

    def get_click_pos_from_event(self, event):
        pos = event.scenePos()
        vb = self.plot.getViewBox()
        if vb is None:
            raise RuntimeError("Cannot map click: Plot has no valid ViewBox.")

        mouse_point = vb.mapSceneToView(pos)
        x_click = mouse_point.x() + self.view_start
        y_click = mouse_point.y()
        return x_click, y_click

    def add_click_listener(self, callback) -> None:
        """Register callback(x, y, button_str) for each click inside the plot view."""
        self.plot.scene().sigMouseClicked.connect(
            lambda event: self._dispatch_click(event, callback)
        )

    def _dispatch_click(self, event, callback) -> None:
        x, y = self.get_click_pos_from_event(event)
        view_start, view_end = self.get_view()
        if not (view_start <= x <= view_end):
            return
        mods = event.modifiers()
        if event.button() == Qt.MouseButton.LeftButton and mods == Qt.KeyboardModifier.ShiftModifier:
            button = "shift_left"
        elif event.button() == Qt.MouseButton.LeftButton:
            button = "left"
        elif event.button() == Qt.MouseButton.RightButton:
            button = "right"
        else:
            return
        callback(x, y, button)

    def get_view(self):
        return self.view_start, self.view_end

    def get_view_range(self) -> tuple:
        """Return current ((xmin, xmax), (ymin, ymax)) from the main plot."""
        ranges = self.plot.viewRange()
        return ranges[0], ranges[1]

    def restore_view_range(self, x_range, y_range) -> None:
        self.plot.setXRange(*x_range, padding=0)
        self.plot.setYRange(*y_range, padding=0)

    def repaint(self) -> None:
        self.parent_widget.repaint()

    def show(self) -> None:
        self.parent_widget.show()

    def is_popup_visible(self) -> bool:
        return self.popup_widget is not None and self.popup_widget.isVisible()

    def wire_hover(self, callback) -> None:
        """Wire hover display. callback(pos) is called on each mouse move."""
        self.plot.scene().sigMouseMoved.connect(callback)

    def distance_to_line(self, x_data, y_data, x_line, y_line):
        """Return pixel distance between a data point and a line point."""
        vb = self.plot.getViewBox()
        p_data = vb.mapViewToScene(pg.Point(x_data, y_data))
        p_line = vb.mapViewToScene(pg.Point(x_line, y_line))
        return np.hypot(p_data.x() - p_line.x(), p_data.y() - p_line.y())

    def snap_to_line(self, x_click, signal_len):
        """Snap a click timestamp to the nearest sample index."""
        idx = int(np.argmin(np.abs(self.time_sec - x_click)))
        idx = int(np.clip(idx, 0, signal_len - 1))
        return float(self.time_sec[idx]), idx

    def get_confidence(self):
        return [
            (
                button_group.checkedButton().text()
                if button_group.checkedButton() is not None
                else None
            )
            for button_group in self.radio_buttons
        ]

    def get_ignored_leads(self):
        """Return the current ignore checkbox states."""
        return [checkbox.isChecked() for checkbox in self.ignore_checks]

    def update_lead_controls(self, lead_names: list[str]) -> None:
        """Clear and rebuild the per-lead sidebar rows for a new set of leads."""
        if self._lead_rows_layout is None:
            return
        while self._lead_rows_layout.count():
            item = self._lead_rows_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.ignore_checks = []
        self.radio_buttons = []
        for lead_idx, lead_name in enumerate(lead_names):
            self._lead_rows_layout.addWidget(
                self._build_lead_controls_row(lead_idx, lead_name)
            )

    def on_show_only_main_toggle(self, state, _signal_count=None):
        self._pfa_visible = (state == 0)  # unchecked = show PFA
        x_range, y_range = self.plot.viewRange()
        self._update_line_visibility()
        self.plot.setXRange(*x_range, padding=0)
        self.plot.setYRange(*y_range, padding=0)

    def set_pfa_available(self, available: bool) -> None:
        """Enable or gray out the Hide PFA checkbox depending on whether PFA leads exist."""
        if self.show_only_main_checkbox is None:
            return
        self.show_only_main_checkbox.setEnabled(available)
        self.show_only_main_checkbox.blockSignals(True)
        self.show_only_main_checkbox.setChecked(False)  # always default to showing PFA
        self.show_only_main_checkbox.blockSignals(False)
        self._pfa_visible = available

    def clear_comment(self):
        if self.comment_box is not None:
            self.comment_box.clear()

    def add_comment_box(self):
        comment_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        comment_widget.setLayout(layout)

        label = QLabel("Comment")
        label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(label)

        self.comment_box = QLineEdit()
        self.comment_box.setPlaceholderText("Enter comment here...")
        layout.addWidget(self.comment_box)

        proxy = QtWidgets.QGraphicsProxyWidget()
        proxy.setWidget(comment_widget)
        self.parent_widget.addItem(proxy, row=2, col=0)

    def plot_ecg_leads(self, leads, lead_names: list[str]):
        if not leads:
            raise RuntimeError("No leads passed to plot ecg leads")

        self._update_time_axis(np.arange(len(leads[0]), dtype=float) / self.fs)
        window = self._window_slice(self.view_start)
        self.view_start = window.start_time
        self.view_end = window.end_time

        ignored_leads = self.get_ignored_leads()
        self.clear_lines()

        for lead_idx, lead in enumerate(leads):
            name = lead_names[lead_idx] if lead_idx < len(lead_names) else f"Lead {lead_idx + 1}"
            self._plot_single_lead(
                lead_idx,
                name,
                lead,
                window,
                self.vertical_spacing,
                ignored_leads,
            )

        self._update_line_visibility()

    def plot_pfa_leads(self, pfa_leads: list[np.ndarray], pfa_lead_names: list[str], lead_offset: float) -> None:
        """Append PFA leads below the main leads. Always call after plot_ecg_leads."""
        if not pfa_leads:
            return
        self._cached_pfa_leads = list(pfa_leads)
        window = self._window_slice(self.view_start)
        label_x = self._plot_label_x(window.x_values)
        for lead_idx, (lead, name) in enumerate(zip(pfa_leads, pfa_lead_names)):
            line = self.plot.plot(window.x_values, lead[window.indices], pen=self.PFA_PEN, name=name, autoDownsample=True)
            self.pfa_lines.append(line)
            label = self._build_lead_label(name, label_x, (lead_offset + lead_idx) * self.vertical_spacing)
            self.plot.addItem(label)
            self.pfa_lead_labels.append(label)
        self._update_line_visibility()

    def plot_no_patient_plot(self):
        self.plot.showAxis("left", show=False)

        text_item = pg.TextItem(
            self.NO_PATIENT_MESSAGE,
            anchor=(0.5, 0.5),
            color="k",
        )
        view_rect = self.plot.viewRect()
        text_item.setPos(view_rect.center().x(), view_rect.center().y())
        self.plot.addItem(text_item)
        self.plot.update()

    def create_overview_axis_pg(
        self,
        lead_I,
        content_start_time,
        on_overview_region_changed,
        on_overview_clicked,
    ):
        """Create the overview plot shown under the main ECG plot."""
        self.overview_plot = pg.PlotItem()
        self.overview_plot.setMaximumHeight(80)
        self.overview_plot.setMenuEnabled(False)
        self.overview_plot.showAxis("left", False)
        self.overview_plot.setLabel("bottom", "Overview", **{"size": "9pt"})
        self.parent_widget.addItem(self.overview_plot, row=1, col=0)

        vb = self.overview_plot.getViewBox()
        vb.setMouseEnabled(x=False, y=False)
        vb.setMenuEnabled(False)
        vb.setDefaultPadding(0.0)

        stride = max(1, len(self.time_sec) // self.OVERVIEW_MAX_POINTS)
        overview_time = self.time_sec[::stride]
        overview_signal = lead_I[::stride]

        overview_curve = pg.PlotDataItem(
            overview_time,
            overview_signal,
            pen=pg.mkPen(color="gray", width=0.8),
        )
        self.overview_plot.addItem(overview_curve)

        self.overview_window = pg.LinearRegionItem(
            values=(self.view_start, self.view_start + self.window_duration),
            brush=pg.mkBrush((255, 165, 0, 100)),
            movable=True,
            bounds=(self.time_sec[0], self.time_sec[-1]),
        )
        self.overview_plot.addItem(self.overview_window)

        self.overview_window.sigRegionChanged.connect(on_overview_region_changed)
        self.overview_plot.scene().sigMouseClicked.connect(on_overview_clicked)

        self.overview_plot.getAxis("bottom").setTicks(
            [
                [
                    (time_point, sec_to_hms(time_point, content_start_time))
                    for time_point in np.linspace(
                        self.time_sec[0],
                        self.time_sec[-1],
                        20,
                    )
                ]
            ]
        )
        self.overview_plot.setYRange(min(overview_signal), max(overview_signal))

    def show_popup(
        self,
        msg: str,
        next_patient_file_idx: int,
        update_idx,
        save_anyway_callback,
        annotation_type,
    ):
        """Show a styled popup overlay with validation feedback."""
        self.close_popup()

        self.popup_widget = QWidget(self.parent_widget)
        self.popup_widget.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.popup_widget.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            False,
        )
        self.popup_widget.setStyleSheet(
            """
            background-color: #ffffff;
            border: 2px solid #FF8C00;
            border-radius: 8px;
        """
        )

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setOffset(3, 3)
        shadow.setColor(QColor(0, 0, 0, 120))
        self.popup_widget.setGraphicsEffect(shadow)

        width = self.parent_widget.width()
        height = self.parent_widget.height()
        self.popup_widget.setGeometry(
            int(width * 0.25),
            int(height * 0.35),
            int(width * 0.5),
            int(height * 0.25),
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        self.popup_widget.setLayout(layout)

        label = QLabel(msg)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size: 12pt; color: #333333;")
        layout.addWidget(label)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(25)
        layout.addLayout(button_layout)

        button_layout.addWidget(
            self._build_popup_button("Close", "#FF8C00", "#FFA500", self.close_popup)
        )

        if annotation_type is not None:
            button_layout.addWidget(
                self._build_popup_button(
                    "Save Anyway",
                    "#4CAF50",
                    "#45A049",
                    lambda: save_anyway_callback(next_patient_file_idx, update_idx),
                )
            )

        self.popup_widget.show()

    def close_popup(self, event=None):
        if self.popup_widget is None:
            return

        self.popup_widget.hide()
        self.popup_widget.deleteLater()
        self.popup_widget = None
