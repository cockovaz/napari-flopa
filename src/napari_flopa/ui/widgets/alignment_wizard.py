import numpy as np
from qtpy.QtCore import Qt, QThreadPool, Signal, Slot
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
)

from napari_flopa.core.io.config import US
from napari_flopa.core.processing.alignment import (
    fit_failed,
    optimize_shift,
    prealign,
    segment_profiles,
)
from napari_flopa.ui.style import MPL, C, S, apply_style
from napari_flopa.ui.utils.threading import Worker
from napari_flopa.ui.widgets.line_plot import LinePlotWidget
from napari_flopa.ui.widgets.status_label import StatusLabel


class AlignmentWizard(QDialog):
    """Tune the line marker delays of a bidirectional scan.

    Estimate align runs the coarse pre-alignment, Optimize refines it by
    correlating the two scan directions, and Show align reconstructs a probe
    chunk at the delays currently entered. Delays are entered in µs and
    emitted in seconds.

    Signals:
        applied(float, float) — line start and stop delay, in seconds
    """

    applied = Signal(float, float)

    def __init__(self, ptu_data, settings, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowTitle("Alignment Wizard")
        self.resize(720, 640)

        self._ptu_data = ptu_data
        self._settings = settings
        self._sync_rate = float(ptu_data["constants"]["repetition_rate"])
        self._syncing = False
        self.threadpool = QThreadPool()
        self._build_ui()

        start_us, stop_us = (
            settings.line_start_marker_delay / US,
            settings.line_stop_marker_delay / US,
        )
        self._start_spin.setValue(start_us)
        self._stop_spin.setValue(stop_us)
        self._pixels_spin.setValue(int(settings.pixels))
        self._sync_shift_from_delays()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        self._start_spin = self._delay_spin()
        self._stop_spin = self._delay_spin()
        self._shift_spin = self._delay_spin()

        self._fix_shift = QCheckBox("Fix shift")
        self._fix_shift.setToolTip(
            "Hold the shift and derive the stop delay from it, instead of "
            "editing the stop delay directly"
        )
        shift_row = QHBoxLayout()
        shift_row.setContentsMargins(0, 0, 0, 0)
        shift_row.addWidget(self._shift_spin, 1)
        shift_row.addWidget(self._fix_shift)

        self._pixels_spin = QSpinBox()
        self._pixels_spin.setRange(16, 4096)
        self._pixels_spin.setSingleStep(16)

        self._chunk_spin = QSpinBox()
        self._chunk_spin.setRange(100, 20_000)
        self._chunk_spin.setSingleStep(100)
        self._chunk_spin.setValue(500)

        self._skip_spin = QSpinBox()
        self._skip_spin.setRange(0, 100)
        self._skip_spin.setValue(1)

        self._max_shift_spin = QDoubleSpinBox()
        self._max_shift_spin.setRange(0.1, 1000.0)
        self._max_shift_spin.setDecimals(3)
        self._max_shift_spin.setSingleStep(0.5)
        self._max_shift_spin.setValue(3.0)

        self._steps_spin = QSpinBox()
        self._steps_spin.setRange(3, 101)
        self._steps_spin.setSingleStep(2)
        self._steps_spin.setValue(11)

        form = QFormLayout()
        form.setContentsMargins(6, 2, 6, 4)
        form.setSpacing(4)
        form.addRow("Line start marker delay (µs):", self._start_spin)
        form.addRow("Line stop marker delay (µs):", self._stop_spin)
        form.addRow("Shift (µs):", shift_row)
        form.addRow("Pixels per line:", self._pixels_spin)
        form.addRow("Chunk size (×1000 records):", self._chunk_spin)
        form.addRow("Chunks to skip:", self._skip_spin)
        form.addRow("Max shift (µs):", self._max_shift_spin)
        form.addRow("Optimization steps:", self._steps_spin)
        root.addLayout(form)

        self._estimate_btn = QPushButton("Estimate align")
        self._estimate_btn.setToolTip(
            "Coarse pre-alignment of the forward and backward sweeps"
        )
        self._estimate_btn.clicked.connect(self._on_estimate)
        self._optimize_btn = QPushButton("Optimize")
        self._optimize_btn.setToolTip(
            "Refine the delays by correlating the two scan directions"
        )
        self._optimize_btn.clicked.connect(self._on_optimize)
        self._show_btn = QPushButton("Show align")
        self._show_btn.setToolTip(
            "Reconstruct a probe chunk at the delays entered above"
        )
        self._show_btn.clicked.connect(self._on_show)
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setToolTip("Send both delays to the File tab")
        apply_style(self._apply_btn, S.BTN_SUCCESS)
        self._apply_btn.clicked.connect(self._on_apply)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)

        buttons = QHBoxLayout()
        for button in (self._estimate_btn, self._optimize_btn, self._show_btn):
            buttons.addWidget(button)
        buttons.addStretch()
        buttons.addWidget(self._apply_btn)
        buttons.addWidget(close_btn)
        root.addLayout(buttons)

        # Drawn where the markers actually are, not where the delays put the
        # acceptance window — that is the offset you are tuning out.
        self._marker_checks = {}
        marker_row = QHBoxLayout()
        marker_row.setContentsMargins(0, 0, 0, 0)
        for name, label in (
            ("line_start", "Line start markers"),
            ("line_stop", "Stop markers"),
        ):
            check = QCheckBox(label)
            check.setChecked(True)
            check.toggled.connect(self._update_marker_visibility)
            self._marker_checks[name] = check
            marker_row.addWidget(check)
        marker_row.addStretch()
        root.addLayout(marker_row)

        self._alignment_plot = LinePlotWidget()
        self._alignment_plot.set_labels(
            "Time (µs)", "Photon count", "Click Estimate align or Show align"
        )
        self._correlation_plot = LinePlotWidget()
        self._correlation_plot.set_labels(
            "Shift (µs)", "Score", "Click Optimize"
        )
        self._tabs = QTabWidget()
        self._tabs.addTab(self._alignment_plot, "Alignment")
        self._tabs.addTab(self._correlation_plot, "Correlation")
        root.addWidget(self._tabs, stretch=1)

        #: x-axis unit per plot, for the cursor readouts below them.
        self._plot_units = {
            self._alignment_plot: "µs",
            self._correlation_plot: "µs",
        }
        #: optional second unit per plot, as ``(label, convert)``.
        self._plot_top = {
            self._alignment_plot: None,
            self._correlation_plot: None,
        }
        self._selector_label = QLabel("Cursor: --")
        self._selection_label = QLabel("Selection Δ: --")
        self._selection_label.setMinimumWidth(140)
        readout_row = QHBoxLayout()
        readout_row.setContentsMargins(0, 0, 0, 0)
        readout_row.addWidget(self._selector_label)
        readout_row.addWidget(self._selection_label)
        readout_row.addStretch()
        root.addLayout(readout_row)

        self._status = StatusLabel("Ready.")
        root.addWidget(self._status)

        for plot in (self._alignment_plot, self._correlation_plot):
            plot.selector_changed.connect(self._on_selector_moved)
            plot.selection_range_changed.connect(self._on_range_selected)
            plot.selection_cleared.connect(self._on_selection_cleared)

        # With the shift fixed it drives the stop delay; otherwise the delays
        # are edited directly and the shift is left exactly as it was.
        self._start_spin.valueChanged.connect(self._on_start_changed)
        self._shift_spin.valueChanged.connect(self._on_shift_changed)
        self._fix_shift.toggled.connect(self._on_fix_shift_toggled)
        self._on_fix_shift_toggled(self._fix_shift.isChecked())

    @staticmethod
    def _delay_spin():
        spin = QDoubleSpinBox()
        spin.setRange(-100_000.0, 100_000.0)
        spin.setDecimals(3)
        spin.setSingleStep(1.0)
        return spin

    def _sync_shift_from_delays(self):
        if self._syncing:
            return
        self._syncing = True
        self._shift_spin.setValue(
            self._start_spin.value() + self._stop_spin.value()
        )
        self._syncing = False

    def _sync_stop_from_shift(self):
        if self._syncing:
            return
        self._syncing = True
        self._stop_spin.setValue(
            self._shift_spin.value() - self._start_spin.value()
        )
        self._syncing = False

    def _on_start_changed(self):
        if self._fix_shift.isChecked():
            self._sync_stop_from_shift()

    def _on_shift_changed(self):
        if self._fix_shift.isChecked():
            self._sync_stop_from_shift()

    def _on_fix_shift_toggled(self, checked: bool):
        self._stop_spin.setReadOnly(checked)
        self._shift_spin.setReadOnly(not checked)
        if checked:
            self._sync_stop_from_shift()

    def _on_selector_moved(self, value: float):
        unit = self._plot_units.get(self.sender(), "")
        self._selector_label.setText(
            f"Cursor: {value:.4g} {unit}{self._top_readout(value)}"
        )

    def _top_readout(self, value: float) -> str:
        """The same position in the top axis' unit, when there is one."""
        top = self._plot_top.get(self.sender())
        if top is None:
            return ""
        label, convert = top
        return f" ({convert(value):.4g} {label})"

    def _on_range_selected(self, start: float, stop: float):
        unit = self._plot_units.get(self.sender(), "")
        width = abs(stop - start)
        top = self._plot_top.get(self.sender())
        extra = ""
        if top is not None:
            label, convert = top
            extra = f" ({abs(convert(stop) - convert(start)):.4g} {label})"
        self._selection_label.setText(
            f"Selection Δ: {width:.4g} {unit}{extra}"
        )

    def _on_selection_cleared(self):
        self._selection_label.setText("Selection Δ: --")

    def _update_marker_visibility(self):
        self._alignment_plot.set_marker_visibility(
            {name: c.isChecked() for name, c in self._marker_checks.items()}
        )

    def _current_config(self):
        return self._settings.replace(
            bidirectional=True,
            pixels=self._pixels_spin.value(),
            line_start_marker_delay=self._start_spin.value() * US,
            line_stop_marker_delay=self._stop_spin.value() * US,
        ).to_scan_config(self._sync_rate)

    def _probe_kwargs(self) -> dict:
        return {
            "chunk_records": self._chunk_spin.value() * 1000,
            "skip_chunks": self._skip_spin.value(),
        }

    def _buttons(self):
        return (
            self._estimate_btn,
            self._optimize_btn,
            self._show_btn,
            self._apply_btn,
        )

    def _begin_run(self, message: str):
        for button in self._buttons():
            button.setEnabled(False)
        self._status.info(message)

    def _clear_plot(self, plot):
        """Drop a stale result rather than leave it next to a new one."""
        plot.clear()
        plot.set_top_axis(None)
        self._plot_top[plot] = None
        self._selector_label.setText("Cursor: --")
        self._selection_label.setText("Selection Δ: --")

    def _start(self, fn, on_result, message, **kwargs):
        self._begin_run(message)
        worker = Worker(fn, self._ptu_data, self._current_config(), **kwargs)
        worker.signals.result.connect(on_result)
        worker.signals.error.connect(self._on_error)
        worker.signals.finished.connect(self._on_finished)
        self.threadpool.start(worker)

    def _on_estimate(self):
        self._clear_plot(self._alignment_plot)
        self._clear_plot(self._correlation_plot)
        self._start(
            prealign,
            self._on_prealign_result,
            "Estimating alignment…",
            **self._probe_kwargs(),
        )

    def _on_optimize(self):
        self._clear_plot(self._correlation_plot)
        self._start(
            optimize_shift,
            self._on_optimize_result,
            "Optimizing shift…",
            max_shift_s=self._max_shift_spin.value() * US,
            steps=self._steps_spin.value(),
            **self._probe_kwargs(),
        )

    def _on_show(self):
        self._clear_plot(self._alignment_plot)
        self._clear_plot(self._correlation_plot)
        self._start(
            segment_profiles,
            self._on_profiles_result,
            "Reconstructing at the current delays…",
            **self._probe_kwargs(),
        )

    @Slot(object)
    def _on_prealign_result(self, result):
        time_us = np.asarray(result["time_axis"].values, dtype=float) / US
        self._alignment_plot.set_labels(
            "Time (µs)", "Photon count", "Click Estimate align"
        )
        self._show_profiles(
            time_us,
            [
                (result["forward"].values, "forward"),
                (result["backward"].values, "backward"),
                (result["backward_aligned"].values, "backward aligned"),
            ],
            np.asarray(result["durations_s"].values, dtype=float) / US,
        )
        self._plot_units[self._alignment_plot] = "µs"
        time_shift_us = float(result["time_shift"].item()) / US
        # The coarse result is a shift, so pin it and let the stop follow.
        self._fix_shift.setChecked(True)
        self._set_shift(time_shift_us)
        self._tabs.setCurrentWidget(self._alignment_plot)
        self._status.info(
            f"Coarse shift {time_shift_us:.3f} µs "
            f"({int(result['pixel_shift'].item())} probe pixels)."
        )

    @Slot(object)
    def _on_optimize_result(self, result):
        best_us = float(result["best_shift"].item()) / US
        series = [
            (
                np.asarray(result["test_shift"].values, dtype=float) / US,
                np.asarray(result["scores"].values, dtype=float),
                C.SERIES[0],
                "correlation",
            )
        ]
        converged = not fit_failed(result)
        if converged:
            series.append(
                (
                    np.asarray(result["fit_shift"].values, dtype=float) / US,
                    np.asarray(result["fit"].values, dtype=float),
                    C.SERIES[2],
                    "Gaussian fit",
                )
            )
        x_values = np.concatenate([np.asarray(s[0]) for s in series])
        self._correlation_plot.set_series(
            series,
            x_min=float(np.min(x_values)),
            x_max=float(np.max(x_values)),
            markers={"best": ([best_us], MPL.SELECTION)},
        )

        # The optimum is relative to the delays it was measured at, so it
        # accumulates. Either route moves the stop delay by the same amount.
        if self._fix_shift.isChecked():
            self._shift_spin.setValue(self._shift_spin.value() + best_us)
        else:
            self._stop_spin.setValue(self._stop_spin.value() + best_us)

        self._tabs.setCurrentWidget(self._correlation_plot)
        note = "" if converged else " (fit did not converge)"
        self._status.info(
            f"Shift {best_us:+.3f} µs applied; stop delay is now "
            f"{self._stop_spin.value():.3f} µs{note}."
        )

    @Slot(object)
    def _on_profiles_result(self, result):
        forward, backward, time_axis = result
        time_us = np.asarray(time_axis, dtype=float) / US

        # Pixel index is a linear relabelling of the same axis, so the second
        # axis just inverts the even spacing of the pixel centres.
        first = float(time_us[0])
        step = float(time_us[1] - time_us[0]) if len(time_us) > 1 else 1.0
        self._plot_units[self._alignment_plot] = "µs"
        self._plot_top[self._alignment_plot] = (
            "px",
            lambda t: (t - first) / step,
        )
        self._alignment_plot.set_labels(
            "Time (µs)", "Photon count", "Click Show align"
        )
        self._alignment_plot.set_top_axis(
            "pixel",
            lambda t: (t - first) / step,
            lambda px: first + px * step,
        )
        self._show_profiles(
            time_us, [(forward, "forward"), (backward, "backward")]
        )
        self._tabs.setCurrentWidget(self._alignment_plot)
        self._status.info(
            f"Reconstructed at start {self._start_spin.value():.3f} µs / "
            f"stop {self._stop_spin.value():.3f} µs."
        )

    def _show_profiles(self, x_values, labelled, stop_markers_us=None):
        """Draw the profiles, with marker overlays when *stop_markers_us*."""
        series = [
            (x_values, np.asarray(values, dtype=float), C.SERIES[index], label)
            for index, (values, label) in enumerate(labelled)
        ]
        bounds = [float(np.min(x_values)), float(np.max(x_values))]
        markers = None

        if stop_markers_us is not None:
            # The line start marker defines t = 0 and the stop markers sit at
            # the line durations — both undelayed, so a delay reads as an
            # offset. The data only spans the pixel centres, so widen the axis
            # rather than clip a marker away.
            stops = np.asarray(stop_markers_us, dtype=float)
            bounds.append(0.0)
            if stops.size:
                bounds += [float(np.min(stops)), float(np.max(stops))]
            markers = {
                "line_start": ([0.0], MPL.MARKER_LINE_START),
                "line_stop": (stops, MPL.MARKER_LINE_STOP),
            }

        self._alignment_plot.set_series(
            series,
            x_min=min(bounds),
            x_max=max(bounds),
            markers=markers,
        )
        self._update_marker_visibility()

    def _set_shift(self, shift_us: float):
        self._syncing = True
        self._shift_spin.setValue(shift_us)
        self._stop_spin.setValue(shift_us - self._start_spin.value())
        self._syncing = False

    @Slot(tuple)
    def _on_error(self, error_tuple):
        _, value, tb = error_tuple
        self._status.error(str(value))
        print(tb)

    @Slot()
    def _on_finished(self):
        for button in self._buttons():
            button.setEnabled(True)

    def _on_apply(self):
        start_s = self._start_spin.value() * US
        stop_s = self._stop_spin.value() * US
        self.applied.emit(start_s, stop_s)
        self._status.info("Delays sent to the File tab.")
