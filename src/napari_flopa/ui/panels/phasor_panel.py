import traceback
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from qtpy.QtCore import Qt, Slot
from qtpy.QtGui import QBrush, QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg as FigureCanvas,
    )
except ImportError:
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvasQTAgg as FigureCanvas,
    )

from napari_flopa.core.processing.phasor import (
    apply_calibration,
    calibrate_mean_arrival_time,
    calibration_factor,
    irf_delay_from_calibration,
)
from napari_flopa.ui.state import FlopaState
from napari_flopa.ui.style import MPL, S, apply_style
from napari_flopa.ui.widgets.phasor_axes import (
    count_outside,
    draw_lifetime_ticks,
    draw_semicircle,
    finalise_phasor_axes,
)
from napari_flopa.ui.widgets.phasor_roi import (
    PhasorRoiSelector,
    napari_label_color,
)
from napari_flopa.ui.widgets.status_label import StatusLabel

# Scatter subsample cap — display only; the CSV export always holds every
# valid pixel. 300k covers a full 500×500 frame without thinning.
_MAX_PX = 300_000


class PhasorPanel(QWidget):
    """Phasor tab - phasor analysis

    Plot rules::

        Per pixel, no mask   → Scatter (white) | Intensity Weighted | Density
        Per pixel, mask      → "Labels" mode, colours from the napari layer
        Per pixel, mask + "As filter"
                             → mask used as binary filter, sub-modes apply again
        Per object, no mask  → single centroid (white)
        Per object, mask     → one centroid per label, napari layer colours

    The calibration factor is complex and applied as ``(g + i·s) · factor``;
    "Revert" restores the factor that came with the loaded config (1+0j if
    none).

    Table export: dataset_name, label_id, g, s, photon_count_sum,
    area_pixels - holding every valid point even when the scatter
    was subsampled for display.
    """

    # Pixel plot sub-modes (only relevant for per-pixel, no mask)
    PM_SCATTER = 0
    PM_INTENSITY = 1  # alpha-weighted by photon count
    PM_DENSITY = 2  # 2D histogram imshow

    def __init__(self, state: FlopaState, viewer, parent=None):
        super().__init__(parent)
        self.state = state
        self.viewer = viewer

        self._current_view: dict = {}
        self._plotted_settings: dict | None = None
        self._final_plot_data: dict | None = None  # points actually drawn
        self._export_data: dict | None = None  # every valid point, for CSV
        self._calib_factor: complex = 1.0 + 0j
        # Factor handed over by the last reconstruction (from the config file),
        # i.e. what "Revert" goes back to; 1+0j when the config carried none.
        self._loaded_calib_factor: complex = 1.0 + 0j

        # (ys, xs) of each plotted point in per-pixel mode, for ROI painting
        self._roi_pixel_yx = None

        self._build_ui()
        self._roi = PhasorRoiSelector(viewer, self._canvas, self._roi_context)
        self.state.dataset_changed.connect(self._on_dataset_changed)
        self.state.calib_factor_changed.connect(self._on_state_calib_changed)
        self.viewer.layers.events.inserted.connect(
            lambda _: self._refresh_mask_layers()
        )
        self.viewer.layers.events.removed.connect(
            lambda _: self._refresh_mask_layers()
        )

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        tb = QHBoxLayout()
        self._plot_btn = QPushButton("Plot")
        self._plot_btn.setStyleSheet(S.BTN_RUN)
        self._plot_btn.setEnabled(False)
        self._plot_btn.setToolTip(
            "Plot the phasor of the slice shown in FLIM View and fill the table"
        )
        tb.addWidget(self._plot_btn)

        self._stale = QLabel("●")
        self._stale.setFixedWidth(14)
        self._stale.setAlignment(Qt.AlignCenter)
        self._stale.setStyleSheet(S.STALE_INACTIVE)
        self._stale.setVisible(False)
        tb.addWidget(self._stale)

        self._roi_btn = QPushButton("⬡ ROI Select")
        self._roi_btn.setCheckable(True)
        self._roi_btn.setToolTip(
            "Toggle interactive lasso ROI selection on the phasor plot"
        )
        self._roi_btn.setEnabled(False)  # enabled after first plot
        self._roi_btn.setVisible(
            False
        )  # TODO: re-enable when ROI Select is ready
        tb.addWidget(self._roi_btn)

        self._roi_done_btn = QPushButton("✓ Done")
        self._roi_done_btn.setVisible(False)
        self._roi_done_btn.setStyleSheet(S.BTN_SUCCESS)
        self._roi_done_btn.setToolTip(
            "Finish ROI selection, the next selection starts a new Labels layer"
        )
        tb.addWidget(self._roi_done_btn)

        self._roi_btn.toggled.connect(self._on_roi_toggled)
        self._roi_done_btn.clicked.connect(self._on_roi_done)

        tb.addSpacing(10)
        # tb.addStretch()

        # Monoexponential-lifetime reference overlay — short label here (between
        # Plot and Save), full explanation in the tooltip.
        self._lifetimes_check = QCheckBox("τ marks")
        self._lifetimes_check.setToolTip(
            "Overlay mono-exponential lifetime reference marks (ns) on the "
            "universal semicircle."
        )
        tb.addWidget(self._lifetimes_check)
        tb.addStretch()
        self._export_combo = QComboBox()
        self._export_combo.addItem("Save plot…", "plot")
        self._export_combo.addItem("Save table…", "table")
        self._export_combo.setToolTip(
            "Plot: PNG / SVG / PDF of the figure\n"
            "Table: CSV of every plotted point (g, s, photons, area per label)"
        )
        tb.addWidget(self._export_combo)
        self._save_btn = QPushButton("Save")
        self._save_btn.setToolTip("Save the plot or table chosen on the left")
        self._save_btn.clicked.connect(self._on_export)
        tb.addWidget(self._save_btn)

        root.addLayout(tb)

        # ── Mode — row 1: Per Object / Per Pixel; row 2: per-pixel display ─
        mode_box = QGroupBox("Mode")
        apply_style(mode_box, S.GROUP_PRIMARY)
        mode_outer = QVBoxLayout(mode_box)
        mode_outer.setSpacing(3)  # minimal gap between the two rows
        mode_outer.setContentsMargins(6, 2, 6, 2)

        # Row 1 — object vs pixel
        mode_row = QHBoxLayout()
        mode_row.setSpacing(10)
        self._per_object_radio = QRadioButton("Per Object")
        self._per_pixel_radio = QRadioButton("Per Pixel")
        self._per_object_radio.setChecked(True)
        self._per_object_radio.setToolTip(
            "One photon-weighted mean point per label (or the whole image)"
        )
        self._per_pixel_radio.setToolTip("One point per pixel")
        mode_row.addWidget(self._per_object_radio)
        mode_row.addWidget(self._per_pixel_radio)
        mode_row.addStretch()
        self._per_object_radio.toggled.connect(
            lambda _: self._cancel_roi_if_active()
        )
        self._per_pixel_radio.toggled.connect(
            lambda _: self._cancel_roi_if_active()
        )
        mode_outer.addLayout(mode_row)

        # Row 2 — per-pixel display controls (enabled only in Per Pixel mode)
        px_row = QHBoxLayout()
        px_row.setSpacing(6)
        self._pixel_mode_group = QButtonGroup(self)
        self._pm_scatter = QRadioButton("Scatter")
        self._pm_intensity = QRadioButton("Intensity α")
        self._pm_density = QRadioButton("Density")
        self._pm_scatter.setChecked(True)
        self._pm_scatter.setToolTip("Scatter plot — each pixel as a point")
        self._pm_intensity.setToolTip(
            "Scatter with alpha-channel weighted by photon count"
        )
        self._pm_density.setToolTip("2D histogram density ")
        for rb, mode_id in (
            (self._pm_scatter, self.PM_SCATTER),
            (self._pm_intensity, self.PM_INTENSITY),
            (self._pm_density, self.PM_DENSITY),
        ):
            self._pixel_mode_group.addButton(rb, mode_id)
            px_row.addWidget(rb)

        self._hexbin_check = QCheckBox("Hex")
        self._hexbin_check.setToolTip(
            "Density bin shape — unchecked: square histogram (imshow); "
            "checked: hexagonal bins (hexbin). Density mode only."
        )
        px_row.addWidget(self._hexbin_check)

        # px_row.addWidget(QLabel("Cmap:"))
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(
            ["hot", "inferno", "plasma", "viridis", "magma", "rainbow"]
        )
        self._cmap_combo.setMaximumWidth(85)
        self._cmap_combo.setToolTip("Colormap of the Density plot")
        px_row.addWidget(self._cmap_combo)
        px_row.addStretch()
        mode_outer.addLayout(px_row)

        self._pixel_submodes = [
            self._pm_scatter,
            self._pm_intensity,
            self._pm_density,
        ]
        root.addWidget(mode_box)

        mask_box = QGroupBox("Mask")
        apply_style(mask_box, S.GROUP_PRIMARY)
        mask_lay = QHBoxLayout(mask_box)
        mask_lay.setSpacing(6)
        mask_lay.setContentsMargins(6, 2, 6, 2)
        mask_lay.addWidget(QLabel("Mask layer:"))
        self._mask_combo = QComboBox()
        self._mask_combo.addItem("None")
        self._mask_combo.setMinimumWidth(100)
        self._mask_combo.setToolTip(
            "napari Labels layer: each label is plotted in its layer colour"
        )
        # Give the combo the row's spare width (stretch factor 1) instead of an
        # addStretch() spacer, so it expands from the label to the end and shows
        # long layer names — no fixed width.
        self._mask_combo.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        mask_lay.addWidget(self._mask_combo, 1)

        self._mask_filter_check = QCheckBox("As filter")
        self._mask_filter_check.setToolTip(
            "Treat the mask as binary: plot only the pixels inside it, but as "
            "one population instead of colouring per label.\n"
            "That makes the Per Pixel sub-modes (Intensity α, Density) usable "
            "with a mask."
        )
        mask_lay.addWidget(self._mask_filter_check)
        root.addWidget(mask_box)

        # -- Smoothing (own row, above Calibration) --
        smooth_box = QGroupBox("Smoothing")
        apply_style(smooth_box, S.GROUP_CHECKABLE)
        smooth_box.setCheckable(True)
        smooth_box.setChecked(False)
        smooth_box.setToolTip(
            "Photon-weighted smoothing of G and S before plotting"
        )
        smooth_lay = QFormLayout(smooth_box)
        self._smooth_spin = QSpinBox()
        self._smooth_spin.setRange(3, 19)
        self._smooth_spin.setSingleStep(2)  # odd kernel sizes only
        self._smooth_spin.setValue(3)
        self._smooth_spin.setToolTip("Smoothing kernel size in px")
        kernel_row = QHBoxLayout()
        kernel_row.setContentsMargins(0, 0, 0, 0)
        kernel_row.addWidget(self._smooth_spin)
        kernel_row.addStretch()  # keep the spinbox compact
        smooth_lay.addRow("Kernel:", kernel_row)
        self._smooth_group = smooth_box
        root.addWidget(smooth_box)

        cal_box = QGroupBox("Calibration")
        apply_style(cal_box, S.GROUP_CHECKABLE)
        cal_box.setCheckable(True)
        cal_box.setChecked(False)
        cal_box.setToolTip(
            "Apply the calibration factor to the plot\n"
            "Enables Calibrated lifetime in FLIM View"
        )
        cal_vlay = QVBoxLayout(cal_box)
        cal_vlay.setSpacing(2)
        cal_vlay.setContentsMargins(6, 4, 6, 4)
        self._cal_info = QLabel("")
        self._cal_info.setStyleSheet(S.HINT)
        self._cal_info.setWordWrap(True)
        cal_vlay.addWidget(self._cal_info)
        cal_lay = QHBoxLayout()
        cal_lay.setSpacing(4)
        cal_vlay.addLayout(cal_lay)

        cal_lay.addWidget(QLabel("Factor:"))
        self._calib_display = QLineEdit(self._fmt_complex(self._calib_factor))
        self._calib_display.setReadOnly(True)
        self._calib_display.setStyleSheet(S.DISPLAY)
        self._calib_display.setToolTip(
            "Current calibration factor (complex number). Applied as: phasor × factor"
        )
        self._calib_display.setMaximumWidth(160)
        cal_lay.addWidget(self._calib_display)

        # hidden τ_ref — kept for auto-calibrate; value set via Calculate dialog
        self._tau_ref_spin = QDoubleSpinBox()
        self._tau_ref_spin.setRange(0.001, 1000.0)
        self._tau_ref_spin.setValue(3.834)
        self._tau_ref_spin.setDecimals(3)
        self._tau_ref_spin.setVisible(False)

        calc_btn = QPushButton("Calculate")
        calc_btn.setToolTip(
            "Open dialog: enter τ_ref (ns) + measured G, S → computes factor"
        )
        calc_btn.clicked.connect(self._on_calculate_cal_dialog)
        cal_lay.addWidget(calc_btn)

        custom_btn = QPushButton("Custom")
        custom_btn.setToolTip(
            "Enter a custom calibration factor directly (real + imaginary parts)"
        )
        custom_btn.clicked.connect(self._on_custom_factor_dialog)
        cal_lay.addWidget(custom_btn)

        self._revert_btn = QPushButton("Revert")
        self._revert_btn.clicked.connect(self._revert_calib)
        cal_lay.addWidget(self._revert_btn)
        self._update_revert_tooltip()

        unity_btn = QPushButton("Reset")
        unity_btn.setToolTip("Set the calibration factor to 1+0j")
        unity_btn.clicked.connect(self._reset_calib)
        cal_lay.addWidget(unity_btn)

        cal_lay.addStretch()
        self._cal_group = cal_box
        root.addWidget(cal_box)

        splitter = QSplitter(Qt.Vertical)

        cw = QWidget()
        self._plot_area = cw  # hidden until the first plot is drawn
        cw_lay = QVBoxLayout(cw)
        cw_lay.setContentsMargins(0, 0, 0, 0)
        # Small default size (keeps the initial dock narrow); the Expanding
        # size policy + widgetResizable scroll area stretch it to fill the tab.
        self._fig = Figure(figsize=(3.6, 3.6), tight_layout=True)
        self._fig.patch.set_facecolor(MPL.FIG_BG)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        cw_lay.addWidget(self._canvas)
        splitter.addWidget(cw)
        cw.setVisible(False)  # revealed on the first successful plot

        self._table = QTableWidget(0, 8)
        self._table.setHorizontalHeaderLabels(
            [
                "Color",
                "Label",
                "Pixels",
                "Valid px",
                "Mean G",
                "Mean S",
                "Mean τ (ns)",
                "Total Int.",
            ]
        )
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Fixed
        )
        self._table.setColumnWidth(0, 24)
        self._table.setFixedHeight(130)
        self._table.setVisible(False)
        splitter.addWidget(self._table)

        root.addWidget(splitter, stretch=1)

        self._status = StatusLabel(
            "Load and reconstruct a PTU file with 'All' output to enable phasor."
        )
        root.addWidget(self._status)

        # Initial empty axes
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor(MPL.AXES_BG)
        self._draw_empty()

        self._plot_btn.clicked.connect(self._on_compute_clicked)

        for w in [
            self._per_object_radio,
            self._per_pixel_radio,
            self._pm_scatter,
            self._pm_intensity,
            self._pm_density,
            self._hexbin_check,
            self._mask_combo,
            self._mask_filter_check,
            self._cmap_combo,
            self._lifetimes_check,
            self._smooth_group,
            self._smooth_spin,
            self._cal_group,
        ]:
            if isinstance(w, (QRadioButton, QCheckBox, QGroupBox)):
                w.toggled.connect(self._on_setting_changed)
            elif isinstance(w, QComboBox):
                w.currentTextChanged.connect(self._on_setting_changed)
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                w.valueChanged.connect(self._on_setting_changed)

        self._cal_group.toggled.connect(self.state.set_calib_enabled)
        self.state.set_calib_enabled(self._cal_group.isChecked())

        self._per_pixel_radio.toggled.connect(self._update_control_states)
        self._mask_combo.currentTextChanged.connect(
            self._update_control_states
        )
        self._pm_density.toggled.connect(self._update_control_states)
        self._hexbin_check.toggled.connect(self._update_control_states)
        self._mask_filter_check.toggled.connect(self._update_control_states)

        self._update_control_states()

    def _update_control_states(self):
        """Enable/disable controls based on current mode + mask selection."""
        mask_active = self._mask_combo.currentText() != "None"
        per_pixel = self._per_pixel_radio.isChecked()
        # "As filter" turns a label mask into a binary one: same pixels, but
        # one population — which is what the sub-modes need. Per Pixel only.
        self._mask_filter_check.setEnabled(per_pixel and mask_active)
        color_by_label = mask_active and not self._mask_as_filter()

        # Pixel sub-modes: whenever points are not coloured per label
        sub_modes_ok = per_pixel and not color_by_label
        for rb in self._pixel_submodes:
            rb.setEnabled(sub_modes_ok)

        # Cmap + hex bin: only for density
        density = sub_modes_ok and self._pm_density.isChecked()
        self._cmap_combo.setEnabled(density)
        self._hexbin_check.setEnabled(density)

    def _mask_as_filter(self) -> bool:
        """Mask set to act as a plain binary filter — Per Pixel mode only."""
        return (
            self._per_pixel_radio.isChecked()
            and self._mask_combo.currentText() != "None"
            and self._mask_filter_check.isChecked()
        )

    def _on_setting_changed(self, *_):
        self._update_control_states()
        self._mark_stale()

    def _mark_stale(self):
        if self._plotted_settings is not None:
            if self._get_settings() != self._plotted_settings:
                self._stale.setStyleSheet(S.STALE_STALE)
                self._stale.setToolTip(
                    "Settings changed since last plot — click Plot to refresh."
                )
                self._stale.setVisible(True)

    def _get_settings(self) -> dict:
        return {
            **self._current_view,
            "per_object": self._per_object_radio.isChecked(),
            "pixel_mode": self._pixel_mode_group.checkedId(),
            "mask": self._mask_combo.currentText(),
            "mask_filter": self._mask_filter_check.isChecked(),
            "cmap": self._cmap_combo.currentText(),
            "smooth": self._smooth_group.isChecked(),
            "smooth_k": self._smooth_spin.value(),
            "calibrate": self._cal_group.isChecked(),
            "calib_re": self._calib_factor.real,
            "calib_im": self._calib_factor.imag,
            "lifetimes": self._lifetimes_check.isChecked(),
            "hexbin": self._hexbin_check.isChecked(),
        }

    @Slot()
    def _on_dataset_changed(self):
        ds = self.state.dataset
        has_phasor = ds is not None and "phasor_g" in ds and "phasor_s" in ds
        self._plot_btn.setEnabled(has_phasor)
        self._plotted_settings = None
        self._stale.setVisible(False)
        self._final_plot_data = None
        self._export_data = None
        self._plot_area.setVisible(False)  # hide until (re)plotted
        self._update_cal_info(ds)
        if not has_phasor:
            self._status.info("No phasor data. Reconstruct with 'All' output.")
            self._draw_empty()
        else:
            self._status.info("Phasor data ready. Click 'Plot'.")

    def _update_cal_info(self, ds) -> None:
        """Show the rate, resolution and omega the phasor was built with."""
        if ds is None or "omega" not in ds:
            self._cal_info.setText("")
            self._cal_info.setToolTip("")
            return
        ip = ds.attrs.get("instrument_params", {})
        rate_mhz = float(ip.get("repetition_rate", 0) or 0) / 1e6
        omega = float(ds["omega"].values)
        res_ps = (
            float(ds["tcspc_resolution"].values) * 1e12
            if "tcspc_resolution" in ds
            else float("nan")
        )
        text = (
            f"Dataset: f_rep {rate_mhz:.3f} MHz, resolution {res_ps:.1f} ps, "
            f"ω {omega:.4e} rad/channel"
        )
        tip = "The phasor and the calibration use these values."
        warn = ""
        header_ps = float(ip.get("tcspc_resolution_ns", 0) or 0) * 1e3
        if ip.get("resolution_unit", "ns") != "ns":
            warn = "Resolution not in the file header, a default is used."
        elif header_ps and not np.isclose(res_ps, header_ps, rtol=1e-6):
            warn = (
                f"Built with {res_ps:.1f} ps but the header says "
                f"{header_ps:.1f} ps. Reconstruct again."
            )
        if warn:
            text += f"\n{warn}"
            tip += f"\n{warn}"
        self._cal_info.setText(text)
        self._cal_info.setStyleSheet(S.STATUS_WARN if warn else S.HINT)
        self._cal_info.setToolTip(tip)

    @Slot(dict)
    def on_view_changed(self, settings: dict):
        self._current_view = settings
        self._mark_stale()

    def _refresh_mask_layers(self):
        cur = self._mask_combo.currentText()
        self._mask_combo.blockSignals(True)
        self._mask_combo.clear()
        self._mask_combo.addItem("None")
        for layer in self.viewer.layers:
            if type(layer).__name__ == "Labels":
                self._mask_combo.addItem(layer.name)
        idx = self._mask_combo.findText(cur)
        self._mask_combo.setCurrentIndex(max(0, idx))
        self._mask_combo.blockSignals(False)
        self._update_control_states()

    def _f_rep_hz(self) -> float:
        """Repetition rate the loaded dataset was reconstructed with, in Hz.

        Always taken from the file header via ``instrument_params`` — the same
        value that produced ``omega`` and therefore the phasor coordinates — so
        the reference marks and the calibration describe the data on screen.
        """
        ds = self.state.dataset
        if ds is not None:
            rate = ds.attrs.get("instrument_params", {}).get("repetition_rate")
            if rate:
                return float(rate)
        return float(self.state.frep_mhz) * 1e6

    def _on_calculate_cal_dialog(self):
        """Open dialog: user enters τ_ns + measured G, S → compute factor."""
        dlg = CalibrationDialog(
            default_tau=self._tau_ref_spin.value(), parent=self
        )
        if dlg.exec_() != QDialog.Accepted:
            return
        vals = dlg.get_values()
        try:
            # Persist τ_ref for subsequent Auto calls
            self._tau_ref_spin.setValue(vals["tau_ns"])
            measured = complex(vals["g"], vals["s"])
            self._calib_factor = calibration_factor(
                vals["tau_ns"], measured, self._f_rep_hz()
            )
            self._calib_display.setText(self._fmt_complex(self._calib_factor))
            self.state.set_calib_factor(self._calib_factor)
            self._mark_stale()
        except Exception as e:
            self._status.error(f"Calibration error: {e}")

    def _on_custom_factor_dialog(self):
        """Open dialog: user types real + imaginary parts directly."""
        dlg = CustomFactorDialog(self._calib_factor, parent=self)
        if dlg.exec_() != QDialog.Accepted:
            return
        r, i = dlg.get_values()
        self._calib_factor = complex(r, i)
        self._calib_display.setText(self._fmt_complex(self._calib_factor))
        self.state.set_calib_factor(self._calib_factor)
        self._mark_stale()

    @Slot(object)
    def _on_state_calib_changed(self, factor):
        """Adopt a factor set elsewhere (e.g. a config loaded in the PTU tab).

        It also becomes the new "Revert" target. The echo of this panel's own
        setters is filtered out (the value is already current).
        """
        factor = complex(factor)
        if factor == self._calib_factor:
            return
        self._loaded_calib_factor = factor
        self._update_revert_tooltip()
        self._calib_factor = factor
        self._calib_display.setText(self._fmt_complex(factor))
        # A factor from the loaded config switches calibration on, which also
        # enables "Calibrated" lifetime in FLIM View without a phasor plot.
        if factor != 1 + 0j:
            self._cal_group.setChecked(True)
        self._mark_stale()

    def _revert_calib(self):
        """Back to the factor loaded with the config (1+0j if none)."""
        self._set_calib_factor(self._loaded_calib_factor)

    def _reset_calib(self):
        """Drop calibration — factor 1+0j."""
        self._set_calib_factor(1.0 + 0j)

    def _set_calib_factor(self, factor: complex):
        self._calib_factor = complex(factor)
        self._calib_display.setText(self._fmt_complex(self._calib_factor))
        self.state.set_calib_factor(self._calib_factor)
        self._mark_stale()

    def _update_revert_tooltip(self):
        self._revert_btn.setToolTip(
            "Restore the calibration factor loaded with the config: "
            f"{self._fmt_complex(self._loaded_calib_factor)}"
        )

    @staticmethod
    def _fmt_complex(c: complex) -> str:
        sign = "+" if c.imag >= 0 else "-"
        return f"{c.real:.4f} {sign} {abs(c.imag):.4f}j"

    def _on_compute_clicked(self):
        ds = self.state.dataset
        if ds is None or "phasor_g" not in ds:
            return
        self._refresh_mask_layers()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._do_compute(ds)
            self._plot_area.setVisible(True)  # reveal on first/any plot
        except Exception as e:
            traceback.print_exc()
            self._status.error(f"Error: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def _do_compute(self, ds):
        sel, dims_to_sum = _parse_settings(ds, self._current_view)
        sliced = ds.isel(**sel) if sel else ds
        if dims_to_sum:
            from napari_flopa.core.processing.image_utils import (
                aggregate_dataset,
            )

            sliced = aggregate_dataset(sliced, dims_to_sum)

        phasor_g = sliced["phasor_g"].values.squeeze().astype(np.float32)
        phasor_s = sliced["phasor_s"].values.squeeze().astype(np.float32)
        photon_count = (
            sliced["photon_count"].values.squeeze()
            if "photon_count" in sliced
            else None
        )

        ip = ds.attrs.get("instrument_params", {})
        res_ns = ip.get("tcspc_resolution_ns", 1.0)
        lt_2d = (
            sliced["mean_arrival_time"].values.squeeze().astype(np.float32)
            * res_ns
            if "mean_arrival_time" in sliced
            else None
        )
        # IRF-delay correction of τ, active together with the phasor
        # calibration. Means are taken on the raw lifetime and corrected
        # once, so pixels wrapped by a period cannot skew an object's mean.
        lt_corr = (
            irf_delay_from_calibration(ds, self._calib_factor)
            if self._cal_group.isChecked() and lt_2d is not None
            else None
        )

        def _cal_tau(value: float) -> float:
            if lt_corr is None or not np.isfinite(value):
                return value
            return float(
                calibrate_mean_arrival_time(np.asarray(value), *lt_corr)
            )

        if self._smooth_group.isChecked() and photon_count is not None:
            k = self._smooth_spin.value()
            if (
                k % 2 == 0
            ):  # tttrkit smooth_phasor shifts ½px with even kernels
                k += 1
            from tttrkit.analysis.phasor import smooth_phasor

            phasor_c_smooth = smooth_phasor(
                phasor_g + 1j * phasor_s,
                photon_count.astype(np.uint32),
                size=k,
            )
            phasor_g = phasor_c_smooth.real
            phasor_s = phasor_c_smooth.imag

        if self._cal_group.isChecked() and self._calib_factor != (1.0 + 0j):
            phasor_g, phasor_s = apply_calibration(
                phasor_g, phasor_s, self._calib_factor
            )
        phasor_g = np.asarray(phasor_g, dtype=np.float32)
        phasor_s = np.asarray(phasor_s, dtype=np.float32)

        shape_2d = phasor_g.shape

        mask_name = self._mask_combo.currentText()
        mask_active = mask_name != "None"
        label_2d = None
        mask_layer = None
        if mask_active and mask_name in self.viewer.layers:
            mask_layer = self.viewer.layers[mask_name]
            ldata = np.asarray(mask_layer.data)
            if ldata.ndim > 2:
                ldata = ldata.reshape(-1, ldata.shape[-2], ldata.shape[-1])[0]
            if ldata.shape == shape_2d:
                label_2d = ldata.astype(np.int32)
            else:
                self._status.warn(
                    f"Mask shape {ldata.shape} ≠ phasor shape {shape_2d}. "
                    "Mask ignored."
                )
                mask_active = False

        per_object = self._per_object_radio.isChecked()
        # Binary-filter mode (Per Pixel only): the mask still restricts which
        # pixels are plotted — that happens below via `valid` — but they are
        # drawn as one population instead of coloured per label.
        # `mask_active` (not the combo) so a shape-mismatched mask, which was
        # just discarded above, cannot leave filter mode switched on.
        mask_as_filter = mask_active and self._mask_as_filter()
        color_by_label = mask_active and not mask_as_filter

        (
            g_out,
            s_out,
            labels_out,
            photons_out,
            areas_out,
            valid_out,
            colors_out,
            lt_out,
        ) = ([], [], [], [], [], [], [], [])

        if per_object:
            from tttrkit.analysis.phasor import average_phasor

            phasor_c = phasor_g + 1j * phasor_s
            pc_arr = (
                photon_count.astype(np.float64)
                if photon_count is not None
                else np.ones(phasor_c.shape)
            )

            if not mask_active:
                # whole image → single centroid
                avg = average_phasor(phasor_c, pc_arr)
                valid = (
                    np.isfinite(phasor_g)
                    & np.isfinite(phasor_s)
                    & (pc_arr > 0)
                )
                g_out.append(float(avg.real))
                s_out.append(float(avg.imag))
                labels_out.append(np.nan)
                photons_out.append(float(pc_arr[valid].sum()))
                areas_out.append(int(phasor_g.size))
                valid_out.append(int(valid.sum()))
                colors_out.append((1.0, 1.0, 1.0, 0.9))  # white
                lt_out.append(
                    _cal_tau(float(average_phasor(lt_2d, pc_arr, mask=valid)))
                    if lt_2d is not None
                    else float("nan")
                )
            else:
                for lbl in np.unique(label_2d):
                    if lbl == 0:
                        continue
                    m = label_2d == lbl
                    avg = average_phasor(phasor_c, pc_arr, mask=m)
                    if not np.isfinite(avg):
                        continue
                    valid = (
                        m
                        & np.isfinite(phasor_g)
                        & np.isfinite(phasor_s)
                        & (pc_arr > 0)
                    )
                    g_out.append(float(avg.real))
                    s_out.append(float(avg.imag))
                    labels_out.append(int(lbl))
                    photons_out.append(float(pc_arr[valid].sum()))
                    areas_out.append(int(m.sum()))
                    valid_out.append(int(valid.sum()))
                    lt_out.append(
                        _cal_tau(
                            float(average_phasor(lt_2d, pc_arr, mask=valid))
                        )
                        if lt_2d is not None
                        else float("nan")
                    )
                    rgba = napari_label_color(mask_layer, int(lbl))
                    colors_out.append(rgba)

        else:  # per pixel
            valid = np.isfinite(phasor_g) & np.isfinite(phasor_s)
            if photon_count is not None:
                valid &= photon_count > 0
            if mask_active and label_2d is not None:
                valid &= label_2d > 0

            yx_valid = np.argwhere(
                valid
            )  # (N, 2) — row, col of every valid pixel

            g_pix = phasor_g[valid]
            s_pix = phasor_s[valid]
            pc_pix = (
                photon_count[valid].astype(np.float32)
                if photon_count is not None
                else None
            )
            lt_raw = lt_2d[valid] if lt_2d is not None else None
            lt_pix = (
                calibrate_mean_arrival_time(lt_raw, *lt_corr)
                if lt_raw is not None and lt_corr is not None
                else lt_raw
            )
            lbl_pix = (
                label_2d[valid]
                if (mask_active and label_2d is not None)
                else None
            )

            # Save full arrays for table stats (before subsampling)
            g_full = g_pix
            s_full = s_pix
            pc_full = pc_pix
            lt_full = lt_pix
            lbl_full = lbl_pix
            # Size of each label (or the whole masked / unmasked image) for
            # the summary table; independent of which pixels hold photons.
            if mask_active and label_2d is not None:
                ids, counts = np.unique(label_2d, return_counts=True)
                area_by_label = dict(
                    zip(ids.tolist(), counts.tolist(), strict=True)
                )
                area_all = int((label_2d > 0).sum())
            else:
                area_by_label, area_all = {}, int(phasor_g.size)

            # sub-sample if too large (for plotting only)
            if len(g_pix) > _MAX_PX:
                rng = np.random.default_rng(0)
                idx = rng.choice(len(g_pix), _MAX_PX, replace=False)
                g_pix = g_pix[idx]
                s_pix = s_pix[idx]
                pc_pix = pc_pix[idx] if pc_pix is not None else None
                lt_pix = lt_pix[idx] if lt_pix is not None else None
                lbl_pix = lbl_pix[idx] if lbl_pix is not None else None
                yx_plot = yx_valid[idx]
            else:
                yx_plot = yx_valid

            self._roi_pixel_yx = (yx_plot[:, 0], yx_plot[:, 1])

            g_out = g_pix
            s_out = s_pix
            photons_out = pc_pix if pc_pix is not None else np.ones(len(g_pix))
            labels_out = (
                lbl_pix if lbl_pix is not None else np.full(len(g_pix), np.nan)
            )
            areas_out = np.full(len(g_pix), np.nan)
            valid_out = np.full(len(g_pix), np.nan)
            lt_out = (
                lt_pix if lt_pix is not None else np.full(len(g_pix), np.nan)
            )

            if color_by_label and lbl_pix is not None:
                # build RGBA per point from napari label colors — one lookup
                # per plotted point, so skip it entirely in filter mode
                colors_out = np.array(
                    [
                        napari_label_color(mask_layer, int(lbl))
                        for lbl in lbl_pix
                    ]
                )  # shape (N, 4)
            else:
                colors_out = "white"

        # ── Store the drawn points (ROI selection indexes into these) ───
        dataset_name = ds.attrs.get("source_filename", "N/A")
        self._final_plot_data = {
            "g_coords": np.array(g_out, dtype=np.float32).ravel(),
            "s_coords": np.array(s_out, dtype=np.float32).ravel(),
            "photon_counts": np.array(photons_out, dtype=np.float32).ravel(),
            "labels": np.array(labels_out).ravel(),
            "areas": np.array(areas_out).ravel(),
            "valid": np.array(valid_out).ravel(),
            "lifetimes": np.array(lt_out, dtype=np.float32).ravel(),
            "dataset_name": dataset_name,
        }

        # The CSV export gets every valid point, including the ones the plot
        # dropped when subsampling.
        if per_object:
            self._export_data = self._final_plot_data
        else:
            n_full = len(g_full)
            self._export_data = {
                "g_coords": np.asarray(g_full, dtype=np.float32).ravel(),
                "s_coords": np.asarray(s_full, dtype=np.float32).ravel(),
                "photon_counts": (
                    np.asarray(pc_full, dtype=np.float32).ravel()
                    if pc_full is not None
                    else np.ones(n_full, dtype=np.float32)
                ),
                "labels": (
                    np.asarray(lbl_full).ravel()
                    if lbl_full is not None
                    else np.full(n_full, np.nan)
                ),
                "areas": np.full(n_full, np.nan),
                "valid": np.full(n_full, np.nan),
                "lifetimes": (
                    np.asarray(lt_full, dtype=np.float32).ravel()
                    if lt_full is not None
                    else np.full(n_full, np.nan)
                ),
                "dataset_name": dataset_name,
            }

        if per_object:
            self._roi_pixel_yx = None
        sync_hz = self._f_rep_hz()
        self._draw_phasor(
            np.array(g_out).ravel(),
            np.array(s_out).ravel(),
            colors_out,
            sync_hz=float(sync_hz),
            per_object=per_object,
            color_by_label=color_by_label,
            photon_counts=(
                np.array(photons_out).ravel() if not per_object else None
            ),
            labels=np.array(labels_out).ravel(),
            mask_layer=mask_layer,
            lt_arr=np.array(lt_out).ravel(),
        )

        if per_object:
            self._fill_table_per_object(
                np.array(g_out),
                np.array(s_out),
                np.array(lt_out),
                np.array(photons_out),
                np.array(labels_out),
                np.array(areas_out),
                np.array(valid_out),
                colors_out,
            )
        else:
            self._fill_table_summary(
                g_full,
                s_full,
                lt_raw,
                pc_full,
                # Filter mode → no labels, so the table summarises the masked
                # pixels as the single population the plot shows.
                None if mask_as_filter else lbl_full,
                mask_layer,
                area_by_label,
                area_all,
                _cal_tau,
            )
        self._table.setVisible(True)

        self._plotted_settings = self._get_settings()
        self._stale.setStyleSheet(S.STALE_FRESH)
        self._stale.setToolTip("Plot is up to date.")
        self._stale.setVisible(True)
        self._roi_btn.setEnabled(True)
        # Points outside the fixed axes are drawn but not visible, so they
        # are left out of the plotted count and reported separately.
        n_plot = len(np.array(g_out).ravel())
        n_vis = n_plot - count_outside(g_out, s_out)
        if per_object:
            msg, warn = f"Plotted {n_vis:,} objects.", False
            n_out = n_plot - n_vis
        else:
            n_full = len(g_full)
            scope = " inside the mask" if mask_as_filter else ""
            warn = n_plot < n_full
            msg = (
                f"Plot subsampled: {n_vis:,} of {n_full:,} valid pixels"
                f"{scope} drawn. 'Save table' still exports all of them."
                if warn
                else f"Plotted {n_vis:,} pixels{scope}."
            )
            n_out = count_outside(g_full, s_full)
        if n_out:
            msg += f" {n_out:,} outside the plot area (G/S beyond the axes)."
            warn = True
        (self._status.warn if warn else self._status.info)(msg)

    def _draw_phasor(
        self,
        g,
        s,
        colors,
        *,
        sync_hz,
        per_object,
        color_by_label,
        photon_counts,
        labels,
        mask_layer,
        lt_arr,
    ):
        """Draw the phasor plot.

        *color_by_label* means "a mask is active and should tint the points";
        it is False when the mask acts as a plain binary filter, which routes
        per-pixel drawing to the Scatter / Intensity / Density sub-modes.
        """
        fig = self._fig
        fig.clear()
        ax = fig.add_subplot(111)
        self._ax = ax
        ax.set_facecolor(MPL.AXES_BG)
        fig.patch.set_facecolor(MPL.FIG_BG)

        draw_semicircle(ax)
        if self._lifetimes_check.isChecked():
            draw_lifetime_ticks(ax, sync_hz)

        if len(g) == 0:
            self._finalise_axes(ax)
            self._canvas.draw_idle()
            return

        if per_object:
            # Only centroids — no background scatter
            marker_size = 60 if len(g) == 1 else 50
            ax.scatter(
                g,
                s,
                c=colors,
                s=marker_size,
                edgecolors="white",
                linewidths=0.6,
                alpha=0.9,
                zorder=10,
            )
            # Legend (≤12 labels)
            if color_by_label and labels is not None and len(g) <= 12:
                import matplotlib.patches as mpatches

                handles = []
                for i, lbl in enumerate(labels):
                    fc = (
                        colors[i]
                        if isinstance(colors, list)
                        else (1, 1, 1, 0.9)
                    )
                    lbl_str = f"#{int(lbl)}" if np.isfinite(lbl) else "All"
                    handles.append(mpatches.Patch(color=fc, label=lbl_str))
                ax.legend(
                    handles=handles,
                    fontsize=7,
                    loc="upper right",
                    facecolor=MPL.AXES_BG,
                    edgecolor=MPL.SPINE,
                    labelcolor=MPL.TICK,
                    framealpha=0.8,
                )

        elif color_by_label:
            # Per pixel + mask → color by label
            ax.scatter(g, s, c=colors, s=2, alpha=0.4, linewidths=0, zorder=1)
            # Legend (≤12 unique labels)
            unique_lbls = (
                np.unique(labels[np.isfinite(labels)].astype(int))
                if labels is not None
                else []
            )
            if (
                len(unique_lbls) <= 12
                and len(unique_lbls) > 0
                and mask_layer is not None
            ):
                import matplotlib.patches as mpatches

                handles = []
                for lbl in unique_lbls:
                    rgba = napari_label_color(mask_layer, int(lbl))
                    handles.append(mpatches.Patch(color=rgba, label=f"#{lbl}"))
                ax.legend(
                    handles=handles,
                    fontsize=7,
                    loc="upper right",
                    facecolor=MPL.AXES_BG,
                    edgecolor=MPL.SPINE,
                    labelcolor=MPL.TICK,
                    framealpha=0.8,
                )

        else:
            # Per pixel, no mask — use sub-mode
            pixel_mode = self._pixel_mode_group.checkedId()

            if pixel_mode == self.PM_SCATTER:
                ax.scatter(
                    g, s, c="white", s=2, alpha=0.5, linewidths=0, zorder=1
                )

            elif pixel_mode == self.PM_INTENSITY:
                # Alpha modulated by photon count — color stays white (or label color)
                if photon_counts is not None and photon_counts.max() > 0:
                    alpha_vals = np.clip(
                        photon_counts / photon_counts.max(), 0.02, 1.0
                    )
                else:
                    alpha_vals = np.full(len(g), 0.5)
                # Build RGBA array (white with per-point alpha)
                rgba = np.ones((len(g), 4), dtype=np.float32)
                rgba[:, 3] = alpha_vals
                ax.scatter(g, s, c=rgba, s=2, linewidths=0, zorder=1)

            elif pixel_mode == self.PM_DENSITY:
                cmap_name = self._cmap_combo.currentText()
                if self._hexbin_check.isChecked():
                    hb = ax.hexbin(
                        g,
                        s,
                        gridsize=60,
                        cmap=cmap_name,
                        mincnt=1,
                        extent=[-0.1, 1.1, -0.1, 0.85],
                        zorder=1,
                    )
                    cb = fig.colorbar(
                        hb,
                        ax=ax,
                        orientation="horizontal",
                        fraction=0.046,
                        pad=0.12,
                    )
                else:
                    H, xedges, yedges = np.histogram2d(
                        g, s, bins=150, range=[[-0.1, 1.1], [-0.1, 0.85]]
                    )
                    alpha_img = (H > 0).astype(float)
                    im = ax.imshow(
                        H.T,
                        origin="lower",
                        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                        cmap=cmap_name,
                        aspect="auto",
                        alpha=alpha_img.T,
                        zorder=1,
                    )
                    cb = fig.colorbar(
                        im,
                        ax=ax,
                        orientation="horizontal",
                        fraction=0.046,
                        pad=0.12,
                    )
                cb.ax.xaxis.set_tick_params(colors=MPL.TICK, labelsize=7)
                cb.ax.set_facecolor(MPL.AXES_BG)
                cb.outline.set_edgecolor(MPL.SPINE)
                cb.set_label("Count", color=MPL.TICK, fontsize=8)

        self._finalise_axes(ax)

        # _roi_pixel_yx was set by _do_compute for these same points
        self._roi.set_points(ax, g, s, self._roi_pixel_yx)

        self._canvas.draw_idle()

    def _finalise_axes(self, ax):
        finalise_phasor_axes(ax)

    def _draw_empty(self):
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        self._ax = ax
        ax.set_facecolor(MPL.AXES_BG)
        self._fig.patch.set_facecolor(MPL.FIG_BG)
        draw_semicircle(ax, color="#555555", alpha=1.0)
        ax.text(
            0.5,
            0.25,
            "No data",
            ha="center",
            va="center",
            color="#555555",
            fontsize=11,
        )
        self._finalise_axes(ax)
        self._canvas.draw_idle()

    def _fill_table_per_object(
        self, g, s, lt, photons, labels, areas, valid, colors
    ):
        self._table.setRowCount(len(g))
        for r in range(len(g)):
            rgba = colors[r] if isinstance(colors, list) else (1, 1, 1)
            self._set_table_row(
                r,
                rgba,
                str(int(labels[r])) if np.isfinite(labels[r]) else "all",
                int(areas[r]),
                int(valid[r]),
                g[r],
                s[r],
                (
                    lt[r]
                    if lt is not None and np.isfinite(lt[r])
                    else float("nan")
                ),
                float(photons[r]),
            )
        self._table.resizeColumnsToContents()
        self._table.setColumnWidth(0, 24)

    def _fill_table_summary(
        self,
        g,
        s,
        lt,
        photons,
        labels,
        mask_layer,
        area_by_label,
        area_all,
        cal_tau,
    ):
        """For per-pixel mode: one row per unique label (or one row for whole image).

        Receives the full (non-subsampled) valid pixels and the *raw* lifetime;
        *cal_tau* corrects each averaged τ. Pixels = label size
        (*area_by_label* / *area_all*), Valid px = pixels that entered the mean.
        """
        if g is None or len(g) == 0:
            self._table.setRowCount(0)
            return
        unique = (
            np.unique(labels[np.isfinite(labels)].astype(int))
            if labels is not None and np.any(np.isfinite(labels))
            else []
        )
        from tttrkit.analysis.phasor import average_phasor

        # Photon-weighted, exactly like per-object mode and Batch.
        g = np.asarray(g, dtype=np.float64)
        w = (
            np.asarray(photons, dtype=np.float64)
            if photons is not None
            else np.ones(g.shape)
        )
        phasor_c = g + 1j * np.asarray(s, dtype=np.float64)

        def _row_stats(m):
            avg = average_phasor(phasor_c, w, mask=m)
            lt_val = (
                cal_tau(
                    float(average_phasor(np.asarray(lt, float), w, mask=m))
                )
                if lt is not None
                else float("nan")
            )
            used = (m if m is not None else True) & np.isfinite(phasor_c)
            return (
                float(avg.real),
                float(avg.imag),
                lt_val,
                float(w[used].sum()),
            )

        if len(unique) == 0:
            self._table.setRowCount(1)
            g_val, s_val, lt_val, total = _row_stats(None)
            self._set_table_row(
                0,
                (1.0, 1.0, 1.0),
                "All",
                area_all,
                len(g),
                g_val,
                s_val,
                lt_val,
                total,
            )
        else:
            self._table.setRowCount(len(unique))
            for r, lbl in enumerate(unique):
                m = labels == lbl
                rgba = (
                    napari_label_color(mask_layer, int(lbl))
                    if mask_layer
                    else (0.5, 0.5, 0.5)
                )
                g_val, s_val, lt_val, total = _row_stats(m)
                self._set_table_row(
                    r,
                    rgba,
                    str(lbl),
                    int(area_by_label.get(int(lbl), m.sum())),
                    int(m.sum()),
                    g_val,
                    s_val,
                    lt_val,
                    total,
                )
        self._table.resizeColumnsToContents()
        self._table.setColumnWidth(0, 24)

    def _set_table_row(
        self,
        r,
        rgba,
        label_str,
        pixels,
        valid_px,
        mean_g,
        mean_s,
        mean_lt,
        total_int,
    ):
        swatch = QTableWidgetItem()
        swatch.setBackground(
            QBrush(
                QColor(
                    int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
                )
            )
        )
        swatch.setFlags(Qt.ItemIsEnabled)
        self._table.setItem(r, 0, swatch)
        for c, val in enumerate(
            [
                label_str,
                str(pixels),
                str(valid_px),
                f"{mean_g:.4f}",
                f"{mean_s:.4f}",
                f"{mean_lt:.3f}",
                f"{total_int:.1f}",
            ]
        ):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(r, 1 + c, item)

    def _on_export(self):
        fmt = self._export_combo.currentData()
        if fmt == "plot":
            self._save_plot()
        else:
            self._save_table()

    def _save_plot(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Phasor Plot",
            "phasor.png",
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)",
        )
        if not path:
            return
        try:
            self._fig.savefig(
                path, dpi=150, facecolor=MPL.FIG_BG, bbox_inches="tight"
            )
            self._status.info(f"Plot saved: {Path(path).name}")
        except Exception as e:
            self._status.error(f"Save error: {e}")

    def _save_table(self):
        # Exports every valid point, not just the subsample that was drawn.
        if not self._export_data:
            self._status.info("No data — click 'Plot' first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Table", "phasor_table.csv", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            import pandas as pd

            d = self._export_data
            pd.DataFrame(
                {
                    "dataset_name": d["dataset_name"],
                    "label_id": d["labels"],
                    "g": d["g_coords"],
                    "s": d["s_coords"],
                    "photon_count_sum": d["photon_counts"],
                    "area_pixels": d["areas"],
                    "n_valid_pixels": d["valid"],
                    "mean_arrival_time_ns": d["lifetimes"],
                }
            ).to_csv(path, index=False)
            self._status.info(
                f"Table saved: {Path(path).name} ({len(d['g_coords']):,} rows)"
            )
        except Exception as e:
            self._status.error(f"Save error: {e}")

    def hideEvent(self, event):
        """Cancel ROI selection when this panel is hidden (e.g. tab switch)."""
        self._cancel_roi_if_active()
        super().hideEvent(event)

    def _cancel_roi_if_active(self):
        if self._roi.active:
            self._on_roi_done()

    def _on_roi_toggled(self, checked: bool):
        """Toggle interactive lasso ROI mode on/off."""
        self._roi_btn.setText(
            "⬡ ROI Select" if not checked else "● ROI Active"
        )
        self._roi_done_btn.setVisible(checked)
        if checked:
            self._roi.start()
        else:
            self._roi.stop()

    def _on_roi_done(self):
        """Finish the ROI session; the next one starts a new Labels layer."""
        self._roi_btn.setChecked(False)
        self._roi_btn.setText("⬡ ROI Select")
        self._roi_done_btn.setVisible(False)
        self._roi.finish()

    def _roi_context(self) -> dict:
        """Panel state the ROI selector reads when a lasso closes."""
        fd = self._final_plot_data
        return {
            "per_object": self._per_object_radio.isChecked(),
            "object_labels": fd.get("labels") if fd is not None else None,
            "mask_layer_name": self._mask_combo.currentText(),
            "dataset": self.state.dataset,
        }


class CalibrationDialog(QDialog):
    """Enter reference lifetime + measured G, S → computes calibration factor."""

    def __init__(self, default_tau: float = 3.834, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calculate Calibration Factor")
        lay = QVBoxLayout(self)
        form = QFormLayout()

        self._tau = QDoubleSpinBox()
        self._tau.setRange(0.001, 1000.0)
        self._tau.setDecimals(4)
        self._tau.setValue(default_tau)
        form.addRow("Reference lifetime τ (ns):", self._tau)

        self._g = QDoubleSpinBox()
        self._g.setRange(-2.0, 2.0)
        self._g.setDecimals(5)
        self._g.setSingleStep(0.001)
        form.addRow("Measured phasor G:", self._g)

        self._s = QDoubleSpinBox()
        self._s.setRange(-2.0, 2.0)
        self._s.setDecimals(5)
        self._s.setSingleStep(0.001)
        form.addRow("Measured phasor S:", self._s)

        lay.addLayout(form)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def get_values(self) -> dict:
        return {
            "tau_ns": self._tau.value(),
            "g": self._g.value(),
            "s": self._s.value(),
        }


class CustomFactorDialog(QDialog):
    """Enter a calibration factor directly as real + imaginary parts."""

    def __init__(self, current: complex = 1 + 0j, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set Custom Calibration Factor")
        lay = QVBoxLayout(self)
        form = QFormLayout()

        self._real = QDoubleSpinBox()
        self._real.setRange(-100.0, 100.0)
        self._real.setDecimals(6)
        self._real.setSingleStep(0.001)
        self._real.setValue(current.real)
        form.addRow("Real part:", self._real)

        self._imag = QDoubleSpinBox()
        self._imag.setRange(-100.0, 100.0)
        self._imag.setDecimals(6)
        self._imag.setSingleStep(0.001)
        self._imag.setValue(current.imag)
        form.addRow("Imaginary part (j):", self._imag)

        lay.addLayout(form)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def get_values(self) -> tuple:
        return (self._real.value(), self._imag.value())


def _parse_settings(ds, settings: dict):
    sel, dims_to_sum = {}, []
    for dim in ["frame", "sequence", "channel"]:
        sum_key = f"sum_{dim}s"
        if dim in ds.sizes and settings.get(sum_key, False):
            dims_to_sum.append(dim)
        elif dim in ds.sizes:
            sel[dim] = settings.get(dim, 0)
    return sel, dims_to_sum
