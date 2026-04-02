"""
Phasor analysis panel — modelled on IMCF-Biocev/FLOPA.

Color / plot rules (matching old FLOPA behaviour):
  Per pixel, no mask   → sub-modes: Scatter (white) | Intensity Weighted (alpha) | Density
  Per pixel, mask      → forced "Labels" mode; colors from napari layer .get_color()
  Per object, no mask  → single centroid (white); no background scatter
  Per object, mask     → one centroid per label; colors from napari layer; no background scatter
  Cmap selector        → enabled only for Density (no mask)

Calibration:
  • Factor stored as complex number; applied as phasor_complex * factor
  • "Calculate" button → CalibrationDialog: enter τ_ref (ns) + measured G, S → computes factor
  • "Auto from data"   → uses TCSPC histogram or weighted avg G/S for measured phasor
  • "Reset"            → 1+0j

Stale indicator (●):  turns red on ANY change to view selection OR plot settings after last plot.

Export:
  • Plot  → PNG / SVG / PDF
  • Table → CSV (pandas): dataset_name, label_id, g, s, photon_count_sum, area_pixels
            (populated after every plot, not only per-object)
"""

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
    QGridLayout,
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

from napari_flopa.state import AppState

_DARK_BG = "#1e1e1e"
_AXES_BG = "#2b2b2b"
_TICK_CLR = "#cccccc"
_SPINE_CLR = "#555555"
_MAX_PX = 80_000  # scatter subsample cap


class PhasorPanel(QWidget):
    """Tab 1 — Phasor analysis."""

    # Pixel plot sub-modes (only relevant for per-pixel, no mask)
    PM_SCATTER = 0
    PM_INTENSITY = 1  # alpha-weighted by photon count
    PM_DENSITY = 2  # 2D histogram imshow

    def __init__(self, state: AppState, viewer, parent=None):
        super().__init__(parent)
        self.state = state
        self.viewer = viewer

        self._current_view: dict = {}
        self._plotted_settings: dict | None = None
        self._final_plot_data: dict | None = None  # for CSV export
        self._calib_factor: complex = 1.0 + 0j

        self._build_ui()
        self.state.dataset_changed.connect(self._on_dataset_changed)
        self.viewer.layers.events.inserted.connect(
            lambda _: self._refresh_mask_layers()
        )
        self.viewer.layers.events.removed.connect(
            lambda _: self._refresh_mask_layers()
        )

    # ------------------------------------------------------------------ #
    # UI                                                                   #
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── Toolbar ────────────────────────────────────────────────────
        tb = QHBoxLayout()
        self._plot_btn = QPushButton("Plot Phasor from Current View")
        self._plot_btn.setEnabled(False)
        tb.addWidget(self._plot_btn)

        self._stale = QLabel("●")
        self._stale.setFixedWidth(14)
        self._stale.setAlignment(Qt.AlignCenter)
        self._stale.setStyleSheet("color: #555555; font-size: 16px;")
        self._stale.setVisible(False)
        tb.addWidget(self._stale)

        tb.addStretch()

        self._export_combo = QComboBox()
        self._export_combo.addItem("Save plot…", "plot")
        self._export_combo.addItem("Save table…", "table")
        tb.addWidget(self._export_combo)
        self._save_btn = QPushButton("Save")
        self._save_btn.clicked.connect(self._on_export)
        tb.addWidget(self._save_btn)

        root.addLayout(tb)

        # ── Setup row ──────────────────────────────────────────────────
        setup_row = QHBoxLayout()

        # -- Mode group --
        mode_box = QGroupBox("Mode")
        mode_lay = QVBoxLayout(mode_box)
        mode_lay.setSpacing(2)
        self._per_object_radio = QRadioButton("Per Object")
        self._per_pixel_radio = QRadioButton("Per Pixel")
        self._per_object_radio.setChecked(True)
        mode_lay.addWidget(self._per_object_radio)
        mode_lay.addWidget(self._per_pixel_radio)

        self._pixel_mode_group = QButtonGroup(self)
        self._pm_scatter = QRadioButton("Scatter")
        self._pm_intensity = QRadioButton("Intensity α")
        self._pm_density = QRadioButton("Density")
        self._pm_scatter.setChecked(True)
        self._pm_scatter.setToolTip(
            "Scatter plot — each pixel as a point (white)"
        )
        self._pm_intensity.setToolTip(
            "Scatter with alpha-channel weighted by photon count\n"
            "(brighter = more photons, color stays the same)"
        )
        self._pm_density.setToolTip(
            "2D histogram density (imshow, uses Cmap below)"
        )
        for rb in (self._pm_scatter, self._pm_intensity, self._pm_density):
            self._pixel_mode_group.addButton(
                rb,
                {
                    self._pm_scatter: self.PM_SCATTER,
                    self._pm_intensity: self.PM_INTENSITY,
                    self._pm_density: self.PM_DENSITY,
                }[rb],
            )
            if rb is self._pm_density:
                density_row = QHBoxLayout()
                density_row.setSpacing(4)
                density_row.addWidget(self._pm_density)
                self._hexbin_check = QCheckBox("Hex")
                self._hexbin_check.setToolTip(
                    "Unchecked: square histogram (imshow)\n"
                    "Checked: hexagonal bins (hexbin)"
                )
                density_row.addWidget(self._hexbin_check)
                mode_lay.addLayout(density_row)
            else:
                mode_lay.addWidget(rb)

        self._pixel_submode_widget = QWidget()
        # pixel sub-mode widgets are already added to mode_lay above,
        # keep a reference to enable/disable them together
        self._pixel_submodes = [
            self._pm_scatter,
            self._pm_intensity,
            self._pm_density,
        ]

        setup_row.addWidget(mode_box)

        # -- Mask + cmap --
        mask_box = QGroupBox("Mask / Colors")
        mask_lay = QGridLayout(mask_box)
        mask_lay.setVerticalSpacing(3)

        mask_lay.addWidget(QLabel("Mask layer:"), 0, 0)
        self._mask_combo = QComboBox()
        self._mask_combo.addItem("None")
        self._mask_combo.setMinimumWidth(100)
        mask_lay.addWidget(self._mask_combo, 0, 1)

        mask_lay.addWidget(QLabel("Cmap (density):"), 1, 0)
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(
            ["hot", "inferno", "plasma", "viridis", "magma", "rainbow"]
        )
        self._cmap_combo.setMaximumWidth(85)
        mask_lay.addWidget(self._cmap_combo, 1, 1)

        self._lifetimes_check = QCheckBox("Monoexp. τ on circle")
        self._lifetimes_check.setToolTip(
            "Draw 1 ns, 2 ns, … tick marks on the universal semicircle"
        )
        mask_lay.addWidget(self._lifetimes_check, 2, 0, 1, 2)

        setup_row.addWidget(mask_box)

        # -- Smoothing --
        smooth_box = QGroupBox("Smoothing")
        smooth_box.setCheckable(True)
        smooth_box.setChecked(False)
        smooth_lay = QFormLayout(smooth_box)
        self._smooth_spin = QSpinBox()
        self._smooth_spin.setRange(2, 20)
        self._smooth_spin.setValue(3)
        smooth_lay.addRow("Kernel:", self._smooth_spin)
        self._smooth_group = smooth_box
        setup_row.addWidget(smooth_box)

        root.addLayout(setup_row)

        # ── Calibration ────────────────────────────────────────────────
        cal_box = QGroupBox("Calibration")
        cal_box.setCheckable(True)
        cal_box.setChecked(False)
        cal_lay = QHBoxLayout(cal_box)
        cal_lay.setSpacing(4)
        cal_lay.setContentsMargins(6, 4, 6, 4)

        cal_lay.addWidget(QLabel("Factor:"))
        self._calib_display = QLineEdit(self._fmt_complex(self._calib_factor))
        self._calib_display.setReadOnly(True)
        self._calib_display.setStyleSheet(
            "color: #aaaaaa; font-family: monospace;"
        )
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

        auto_btn = QPushButton("Auto")
        auto_btn.setToolTip(
            "Compute measured phasor from TCSPC histogram (or weighted avg G/S)\n"
            "using τ_ref set in the Calculate dialog"
        )
        auto_btn.clicked.connect(self._on_auto_calibrate)
        cal_lay.addWidget(auto_btn)

        custom_btn = QPushButton("Custom")
        custom_btn.setToolTip(
            "Enter a custom calibration factor directly (real + imaginary parts)"
        )
        custom_btn.clicked.connect(self._on_custom_factor_dialog)
        cal_lay.addWidget(custom_btn)

        reset_btn = QPushButton("Reset")
        reset_btn.setToolTip("Reset calibration factor to 1+0j")
        reset_btn.clicked.connect(self._reset_calib)
        cal_lay.addWidget(reset_btn)

        cal_lay.addStretch()
        self._cal_group = cal_box
        root.addWidget(cal_box)

        # ── Canvas + table splitter ────────────────────────────────────
        splitter = QSplitter(Qt.Vertical)

        cw = QWidget()
        cw_lay = QVBoxLayout(cw)
        cw_lay.setContentsMargins(0, 0, 0, 0)
        self._fig = Figure(tight_layout=True)
        self._fig.patch.set_facecolor(_DARK_BG)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        cw_lay.addWidget(self._canvas)
        splitter.addWidget(cw)

        self._table = QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels(
            [
                "Color",
                "Label",
                "Pixels",
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

        # ── Status ─────────────────────────────────────────────────────
        self._status = QLabel(
            "Load and reconstruct a PTU file with 'All' output to enable phasor."
        )
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #888888; font-size: 10px;")
        root.addWidget(self._status)

        # Initial empty axes
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor(_AXES_BG)
        self._draw_empty()

        # ── Signal wiring ──────────────────────────────────────────────
        self._plot_btn.clicked.connect(self._on_compute_clicked)

        for w in [
            self._per_object_radio,
            self._per_pixel_radio,
            self._pm_scatter,
            self._pm_intensity,
            self._pm_density,
            self._hexbin_check,
            self._mask_combo,
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

        self._per_pixel_radio.toggled.connect(self._update_control_states)
        self._mask_combo.currentTextChanged.connect(
            self._update_control_states
        )
        self._pm_density.toggled.connect(self._update_control_states)
        self._hexbin_check.toggled.connect(self._update_control_states)

        self._update_control_states()

    # ------------------------------------------------------------------ #
    # State management                                                     #
    # ------------------------------------------------------------------ #

    def _update_control_states(self):
        """Enable/disable controls based on current mode + mask selection."""
        mask_active = self._mask_combo.currentText() != "None"
        per_pixel = self._per_pixel_radio.isChecked()
        density_no_mask = (
            per_pixel and not mask_active and self._pm_density.isChecked()
        )

        # Pixel sub-modes: only relevant + enabled for per-pixel, no mask
        for rb in self._pixel_submodes:
            rb.setEnabled(per_pixel and not mask_active)

        # Cmap + hex bin: only for density without mask
        self._cmap_combo.setEnabled(density_no_mask)
        self._hexbin_check.setEnabled(density_no_mask)

    def _on_setting_changed(self, *_):
        self._update_control_states()
        self._mark_stale()

    def _mark_stale(self):
        if self._plotted_settings is not None:
            if self._get_settings() != self._plotted_settings:
                self._stale.setStyleSheet("color: #ff4444; font-size: 16px;")
                self._stale.setToolTip(
                    "Settings or view have changed since the last plot — click Plot to refresh."
                )
                self._stale.setVisible(True)

    def _get_settings(self) -> dict:
        return {
            **self._current_view,
            "per_object": self._per_object_radio.isChecked(),
            "pixel_mode": self._pixel_mode_group.checkedId(),
            "mask": self._mask_combo.currentText(),
            "cmap": self._cmap_combo.currentText(),
            "smooth": self._smooth_group.isChecked(),
            "smooth_k": self._smooth_spin.value(),
            "calibrate": self._cal_group.isChecked(),
            "calib_re": self._calib_factor.real,
            "calib_im": self._calib_factor.imag,
            "lifetimes": self._lifetimes_check.isChecked(),
            "hexbin": self._hexbin_check.isChecked(),
        }

    # ------------------------------------------------------------------ #
    # Dataset / view signals                                               #
    # ------------------------------------------------------------------ #

    @Slot()
    def _on_dataset_changed(self):
        ds = self.state.dataset
        has_phasor = ds is not None and "phasor_g" in ds and "phasor_s" in ds
        self._plot_btn.setEnabled(has_phasor)
        self._plotted_settings = None
        self._stale.setVisible(False)
        self._final_plot_data = None
        if not has_phasor:
            self._status.setText(
                "No phasor data. Reconstruct with 'All (+ Phasor & TCSPC)' output."
            )
            self._draw_empty()
        else:
            self._status.setText(
                "Phasor data ready. Click 'Plot Phasor from Current View'."
            )

    @Slot(dict)
    def on_view_changed(self, settings: dict):
        self._current_view = settings
        self._mark_stale()

    # ------------------------------------------------------------------ #
    # Mask layers                                                          #
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    # Calibration                                                          #
    # ------------------------------------------------------------------ #

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
            f_rep_hz = self.state.frep_mhz * 1e6
            measured = complex(vals["g"], vals["s"])
            self._calib_factor = _calc_calib_factor(
                vals["tau_ns"], measured, f_rep_hz
            )
            self._calib_display.setText(self._fmt_complex(self._calib_factor))
            self.state.set_calib_factor(self._calib_factor)
            self._mark_stale()
        except Exception as e:
            self._status.setText(f"Calibration error: {e}")

    def _on_auto_calibrate(self):
        """Compute measured phasor from TCSPC or weighted avg G/S."""
        ds = self.state.dataset
        if ds is None:
            self._status.setText("No dataset.")
            return
        try:
            tau_ns = self._tau_ref_spin.value()
            f_rep_hz = self.state.frep_mhz * 1e6
            ip = ds.attrs.get("instrument_params", {})
            sel, dims_to_sum = _parse_settings(ds, self._current_view)
            sliced = ds.isel(**sel) if sel else ds
            if dims_to_sum:
                from napari_flopa.processing.image_utils import (
                    aggregate_dataset,
                )

                sliced = aggregate_dataset(sliced, dims_to_sum)

            measured = None

            if "tcspc_histogram" in sliced:
                res_ns = ip.get("tcspc_resolution_ns", None)
                if res_ns is not None:
                    decay = sliced["tcspc_histogram"].values.squeeze()
                    if decay.ndim > 1:
                        decay = decay.sum(axis=tuple(range(decay.ndim - 1)))
                    decay = decay.astype(np.float64)
                    total = decay.sum()
                    if total > 0:
                        omega = 2 * np.pi * f_rep_hz
                        t = np.arange(len(decay)) * (res_ns * 1e-9)
                        measured = (
                            np.dot(decay, np.exp(1j * omega * t)) / total
                        )

            if (
                measured is None
                and "phasor_g" in sliced
                and "phasor_s" in sliced
            ):
                g2 = sliced["phasor_g"].values.squeeze().astype(np.float64)
                s2 = sliced["phasor_s"].values.squeeze().astype(np.float64)
                w = None
                if "photon_count" in sliced:
                    w = (
                        sliced["photon_count"]
                        .values.squeeze()
                        .astype(np.float64)
                    )
                valid = np.isfinite(g2) & np.isfinite(s2)
                if w is not None:
                    wv = w[valid]
                    gv = g2[valid]
                    sv = s2[valid]
                    wsum = wv.sum()
                    g_avg = (
                        np.dot(wv, gv) / wsum
                        if wsum > 0
                        else np.nanmean(g2[valid])
                    )
                    s_avg = (
                        np.dot(wv, sv) / wsum
                        if wsum > 0
                        else np.nanmean(s2[valid])
                    )
                else:
                    g_avg, s_avg = np.nanmean(g2[valid]), np.nanmean(s2[valid])
                measured = complex(g_avg, s_avg)

            if measured is None or not np.isfinite(measured):
                self._status.setText(
                    "Cannot compute measured phasor from current data."
                )
                return

            self._calib_factor = _calc_calib_factor(tau_ns, measured, f_rep_hz)
            self._calib_display.setText(self._fmt_complex(self._calib_factor))
            self.state.set_calib_factor(self._calib_factor)
            self._mark_stale()
            self._status.setText(
                f"Calibration factor: {self._fmt_complex(self._calib_factor)}  "
                f"(τ_ref={tau_ns:.3f} ns)"
            )
        except Exception as e:
            traceback.print_exc()
            self._status.setText(f"Auto-calibration error: {e}")

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

    def _reset_calib(self):
        self._calib_factor = 1.0 + 0j
        self._calib_display.setText(self._fmt_complex(self._calib_factor))
        self.state.set_calib_factor(self._calib_factor)
        self._mark_stale()

    @staticmethod
    def _fmt_complex(c: complex) -> str:
        sign = "+" if c.imag >= 0 else "-"
        return f"{c.real:.4f} {sign} {abs(c.imag):.4f}j"

    # ------------------------------------------------------------------ #
    # Main compute / plot                                                  #
    # ------------------------------------------------------------------ #

    def _on_compute_clicked(self):
        ds = self.state.dataset
        if ds is None or "phasor_g" not in ds:
            return
        self._refresh_mask_layers()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._do_compute(ds)
        except Exception as e:
            traceback.print_exc()
            self._status.setText(f"Error: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def _do_compute(self, ds):
        sel, dims_to_sum = _parse_settings(ds, self._current_view)
        sliced = ds.isel(**sel) if sel else ds
        if dims_to_sum:
            from napari_flopa.processing.image_utils import aggregate_dataset

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

        # ── Smoothing ───────────────────────────────────────────────────
        if self._smooth_group.isChecked() and photon_count is not None:
            k = self._smooth_spin.value()
            from napari_flopa.processing.image_utils import smooth_weighted

            phasor_g, _ = smooth_weighted(
                phasor_g, photon_count.astype(np.uint32), size=k
            )
            phasor_s, _ = smooth_weighted(
                phasor_s, photon_count.astype(np.uint32), size=k
            )

        # ── Calibration ─────────────────────────────────────────────────
        phasor_c = phasor_g + 1j * phasor_s
        if self._cal_group.isChecked() and self._calib_factor != (1.0 + 0j):
            phasor_c = phasor_c * self._calib_factor
        phasor_g = phasor_c.real.astype(np.float32)
        phasor_s = phasor_c.imag.astype(np.float32)

        shape_2d = phasor_g.shape

        # ── Mask ────────────────────────────────────────────────────────
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
                self._status.setText(
                    f"Mask shape {ldata.shape} ≠ phasor shape {shape_2d}. Mask ignored."
                )
                mask_active = False

        per_object = self._per_object_radio.isChecked()

        # ── Build final g/s arrays ──────────────────────────────────────
        (
            g_out,
            s_out,
            labels_out,
            photons_out,
            areas_out,
            colors_out,
            lt_out,
        ) = ([], [], [], [], [], [], [])

        if per_object:
            if not mask_active:
                # whole image → single centroid
                valid = np.isfinite(phasor_g) & np.isfinite(phasor_s)
                if photon_count is not None:
                    valid &= photon_count > 0
                pc_v = (
                    photon_count[valid].astype(np.float64)
                    if photon_count is not None
                    else None
                )
                g_v = phasor_g[valid].astype(np.float64)
                s_v = phasor_s[valid].astype(np.float64)
                g_avg = (
                    np.dot(pc_v, g_v) / pc_v.sum()
                    if pc_v is not None and pc_v.sum() > 0
                    else float(np.nanmean(g_v))
                )
                s_avg = (
                    np.dot(pc_v, s_v) / pc_v.sum()
                    if pc_v is not None and pc_v.sum() > 0
                    else float(np.nanmean(s_v))
                )
                g_out.append(g_avg)
                s_out.append(s_avg)
                labels_out.append(np.nan)
                photons_out.append(
                    float(pc_v.sum())
                    if pc_v is not None
                    else float(valid.sum())
                )
                areas_out.append(int(valid.sum()))
                colors_out.append((1.0, 1.0, 1.0, 0.9))  # white
                lt_v = lt_2d[valid] if lt_2d is not None else None
                lt_out.append(
                    float(np.nanmean(lt_v))
                    if lt_v is not None
                    else float("nan")
                )
            else:
                for lbl in np.unique(label_2d):
                    if lbl == 0:
                        continue
                    m = label_2d == lbl
                    valid = m & np.isfinite(phasor_g) & np.isfinite(phasor_s)
                    if not valid.any():
                        continue
                    pc_v = (
                        photon_count[valid].astype(np.float64)
                        if photon_count is not None
                        else None
                    )
                    g_v = phasor_g[valid].astype(np.float64)
                    s_v = phasor_s[valid].astype(np.float64)
                    wsum = pc_v.sum() if pc_v is not None else 0.0
                    g_avg = (
                        np.dot(pc_v, g_v) / wsum
                        if wsum > 0
                        else float(np.nanmean(g_v))
                    )
                    s_avg = (
                        np.dot(pc_v, s_v) / wsum
                        if wsum > 0
                        else float(np.nanmean(s_v))
                    )
                    g_out.append(g_avg)
                    s_out.append(s_avg)
                    labels_out.append(int(lbl))
                    photons_out.append(
                        float(wsum if wsum > 0 else valid.sum())
                    )
                    areas_out.append(int(m.sum()))
                    lt_v = lt_2d[valid] if lt_2d is not None else None
                    lt_out.append(
                        float(np.nanmean(lt_v))
                        if lt_v is not None
                        else float("nan")
                    )
                    rgba = _get_napari_color(mask_layer, int(lbl))
                    colors_out.append(rgba)

        else:  # per pixel
            valid = np.isfinite(phasor_g) & np.isfinite(phasor_s)
            if photon_count is not None:
                valid &= photon_count > 0
            if mask_active and label_2d is not None:
                valid &= label_2d > 0

            g_pix = phasor_g[valid]
            s_pix = phasor_s[valid]
            pc_pix = (
                photon_count[valid].astype(np.float32)
                if photon_count is not None
                else None
            )
            lt_pix = lt_2d[valid] if lt_2d is not None else None
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

            # sub-sample if too large (for plotting only)
            if len(g_pix) > _MAX_PX:
                rng = np.random.default_rng(0)
                idx = rng.choice(len(g_pix), _MAX_PX, replace=False)
                g_pix = g_pix[idx]
                s_pix = s_pix[idx]
                pc_pix = pc_pix[idx] if pc_pix is not None else None
                lt_pix = lt_pix[idx] if lt_pix is not None else None
                lbl_pix = lbl_pix[idx] if lbl_pix is not None else None

            g_out = g_pix
            s_out = s_pix
            photons_out = pc_pix if pc_pix is not None else np.ones(len(g_pix))
            labels_out = (
                lbl_pix if lbl_pix is not None else np.full(len(g_pix), np.nan)
            )
            areas_out = np.full(len(g_pix), np.nan)
            lt_out = (
                lt_pix if lt_pix is not None else np.full(len(g_pix), np.nan)
            )

            if mask_active and lbl_pix is not None:
                # build RGBA per point from napari label colors
                colors_out = np.array(
                    [
                        _get_napari_color(mask_layer, int(lbl))
                        for lbl in lbl_pix
                    ]
                )  # shape (N, 4)
            else:
                colors_out = "white"

        # ── Store for CSV export ────────────────────────────────────────
        self._final_plot_data = {
            "g_coords": np.array(g_out, dtype=np.float32).ravel(),
            "s_coords": np.array(s_out, dtype=np.float32).ravel(),
            "photon_counts": np.array(photons_out, dtype=np.float32).ravel(),
            "labels": np.array(labels_out).ravel(),
            "areas": np.array(areas_out).ravel(),
            "dataset_name": ds.attrs.get("source_filename", "N/A"),
        }

        # ── Draw ────────────────────────────────────────────────────────
        sync_hz = ip.get("sync_rate_hz", None) or (self.state.frep_mhz * 1e6)
        self._draw_phasor(
            np.array(g_out).ravel(),
            np.array(s_out).ravel(),
            colors_out,
            sync_hz=float(sync_hz),
            per_object=per_object,
            mask_active=mask_active,
            photon_counts=(
                np.array(photons_out).ravel() if not per_object else None
            ),
            labels=(
                np.array(labels_out).ravel()
                if not per_object
                else np.array(labels_out).ravel()
            ),
            mask_layer=mask_layer,
            lt_arr=np.array(lt_out).ravel(),
        )

        # ── Table ────────────────────────────────────────────────────────
        if per_object:
            self._fill_table_per_object(
                np.array(g_out),
                np.array(s_out),
                np.array(lt_out),
                np.array(photons_out),
                np.array(labels_out),
                colors_out,
            )
        else:
            self._fill_table_summary(
                g_full,
                s_full,
                lt_full,
                pc_full,
                lbl_full,
                mask_layer,
            )
        self._table.setVisible(True)

        self._plotted_settings = self._get_settings()
        self._stale.setStyleSheet("color: #44cc44; font-size: 16px;")
        self._stale.setToolTip("Plot is up to date.")
        self._stale.setVisible(True)
        n_plot = len(np.array(g_out).ravel())
        if per_object:
            self._status.setText(f"Plotted {n_plot:,} objects.")
        else:
            n_full = len(g_full)
            if n_plot < n_full:
                self._status.setText(
                    f"Plotted {n_plot:,} pixels (subsampled from {n_full:,} valid pixels)."
                )
            else:
                self._status.setText(f"Plotted {n_plot:,} pixels.")

    # ------------------------------------------------------------------ #
    # Drawing                                                              #
    # ------------------------------------------------------------------ #

    def _draw_phasor(
        self,
        g,
        s,
        colors,
        *,
        sync_hz,
        per_object,
        mask_active,
        photon_counts,
        labels,
        mask_layer,
        lt_arr,
    ):
        fig = self._fig
        fig.clear()
        ax = fig.add_subplot(111)
        self._ax = ax
        ax.set_facecolor(_AXES_BG)
        fig.patch.set_facecolor(_DARK_BG)

        # Universal semicircle
        theta = np.linspace(0, np.pi, 300)
        ax.plot(
            0.5 + 0.5 * np.cos(theta),
            0.5 * np.sin(theta),
            color="white",
            linewidth=0.8,
            alpha=0.5,
            zorder=5,
        )

        if self._lifetimes_check.isChecked():
            _draw_lifetime_ticks(ax, sync_hz)

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
            if mask_active and labels is not None and len(g) <= 12:
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
                    facecolor=_AXES_BG,
                    edgecolor=_SPINE_CLR,
                    labelcolor=_TICK_CLR,
                    framealpha=0.8,
                )

        elif mask_active:
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
                    rgba = _get_napari_color(mask_layer, int(lbl))
                    handles.append(mpatches.Patch(color=rgba, label=f"#{lbl}"))
                ax.legend(
                    handles=handles,
                    fontsize=7,
                    loc="upper right",
                    facecolor=_AXES_BG,
                    edgecolor=_SPINE_CLR,
                    labelcolor=_TICK_CLR,
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
                    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
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
                    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.ax.yaxis.set_tick_params(colors=_TICK_CLR, labelsize=7)
                cb.ax.set_facecolor(_AXES_BG)
                cb.outline.set_edgecolor(_SPINE_CLR)
                cb.set_label("Count", color=_TICK_CLR, fontsize=8)

        self._finalise_axes(ax)
        self._canvas.draw_idle()

    def _finalise_axes(self, ax):
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 0.80)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("G", color=_TICK_CLR)
        ax.set_ylabel("S", color=_TICK_CLR)
        ax.tick_params(colors=_TICK_CLR, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(_SPINE_CLR)

    def _draw_empty(self):
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        self._ax = ax
        ax.set_facecolor(_AXES_BG)
        self._fig.patch.set_facecolor(_DARK_BG)
        theta = np.linspace(0, np.pi, 300)
        ax.plot(
            0.5 + 0.5 * np.cos(theta),
            0.5 * np.sin(theta),
            color="#555555",
            linewidth=0.8,
        )
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

    # ------------------------------------------------------------------ #
    # Table                                                                #
    # ------------------------------------------------------------------ #

    def _fill_table_per_object(self, g, s, lt, photons, labels, colors):
        self._table.setRowCount(len(g))
        fmts = ["{:.4f}", "{:.4f}", "{:.3f}", "{:.1f}"]
        for r in range(len(g)):
            rgba = colors[r] if isinstance(colors, list) else (1, 1, 1)
            self._set_table_row(
                r,
                rgba,
                str(int(labels[r])) if np.isfinite(labels[r]) else "all",
                int(photons[r]) if np.isfinite(photons[r]) else 0,
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

    def _fill_table_summary(self, g, s, lt, photons, labels, mask_layer):
        """For per-pixel mode: one row per unique label (or one row for whole image).
        Receives full (non-subsampled) arrays for accurate statistics."""
        if g is None or len(g) == 0:
            self._table.setRowCount(0)
            return
        unique = (
            np.unique(labels[np.isfinite(labels)].astype(int))
            if labels is not None and np.any(np.isfinite(labels))
            else []
        )
        if len(unique) == 0:
            # Whole image summary — intensity-weighted means to match per-object mode
            self._table.setRowCount(1)
            w = photons.astype(np.float64) if photons is not None else None
            wsum = float(np.nansum(w)) if w is not None else 0.0
            if wsum > 0:
                g_val = float(np.nansum(g.astype(np.float64) * w) / wsum)
                s_val = float(np.nansum(s.astype(np.float64) * w) / wsum)
                lt_val = (
                    float(np.nansum(lt.astype(np.float64) * w) / wsum)
                    if lt is not None
                    else float("nan")
                )
            else:
                g_val = float(np.nanmean(g))
                s_val = float(np.nanmean(s))
                lt_val = (
                    float(np.nanmean(lt)) if lt is not None else float("nan")
                )
            self._set_table_row(
                0,
                (1.0, 1.0, 1.0),
                "All",
                len(g),
                g_val,
                s_val,
                lt_val,
                wsum if wsum > 0 else float(len(g)),
            )
        else:
            self._table.setRowCount(len(unique))
            for r, lbl in enumerate(unique):
                m = labels == lbl
                rgba = (
                    _get_napari_color(mask_layer, int(lbl))
                    if mask_layer
                    else (0.5, 0.5, 0.5)
                )
                self._set_table_row(
                    r,
                    rgba,
                    str(lbl),
                    int(m.sum()),
                    float(np.nanmean(g[m])),
                    float(np.nanmean(s[m])),
                    (
                        float(np.nanmean(lt[m]))
                        if lt is not None
                        else float("nan")
                    ),
                    (
                        float(np.nansum(photons[m]))
                        if photons is not None
                        else float(m.sum())
                    ),
                )
        self._table.resizeColumnsToContents()
        self._table.setColumnWidth(0, 24)

    def _set_table_row(
        self, r, rgba, label_str, pixels, mean_g, mean_s, mean_lt, total_int
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
        for c, (val, fmt) in enumerate(
            zip(
                [
                    label_str,
                    str(pixels),
                    f"{mean_g:.4f}",
                    f"{mean_s:.4f}",
                    f"{mean_lt:.3f}",
                    f"{total_int:.1f}",
                ],
                range(6),
            )
        ):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(r, 1 + c, item)

    # ------------------------------------------------------------------ #
    # Export                                                               #
    # ------------------------------------------------------------------ #

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
                path, dpi=150, facecolor=_DARK_BG, bbox_inches="tight"
            )
            self._status.setText(f"Plot saved: {Path(path).name}")
        except Exception as e:
            self._status.setText(f"Save error: {e}")

    def _save_table(self):
        if not self._final_plot_data:
            self._status.setText("No data — click 'Plot' first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Table", "phasor_table.csv", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            import pandas as pd

            d = self._final_plot_data
            pd.DataFrame(
                {
                    "dataset_name": d["dataset_name"],
                    "label_id": d["labels"],
                    "g": d["g_coords"],
                    "s": d["s_coords"],
                    "photon_count_sum": d["photon_counts"],
                    "area_pixels": d["areas"],
                }
            ).to_csv(path, index=False)
            self._status.setText(f"Table saved: {Path(path).name}")
        except Exception as e:
            self._status.setText(f"Save error: {e}")


# ── CalibrationDialog ─────────────────────────────────────────────────────


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


# ── CustomFactorDialog ────────────────────────────────────────────────────


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
        form.addRow("Real part (i):", self._real)

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


# ── Module-level helpers ──────────────────────────────────────────────────


def _parse_settings(ds, settings: dict):
    sel, dims_to_sum = {}, []
    for dim in ["frame", "sequence", "channel"]:
        sum_key = f"sum_{dim}s"
        if dim in ds.sizes and settings.get(sum_key, False):
            dims_to_sum.append(dim)
        elif dim in ds.sizes:
            sel[dim] = settings.get(dim, 0)
    return sel, dims_to_sum


def _calc_calib_factor(
    tau_ns: float, measured: complex, f_rep_hz: float
) -> complex:
    tau_s = tau_ns * 1e-9
    omega = 2 * np.pi * f_rep_hz
    g_th = 1.0 / (1.0 + (omega * tau_s) ** 2)
    s_th = (omega * tau_s) / (1.0 + (omega * tau_s) ** 2)
    theory = complex(g_th, s_th)
    if abs(measured) < 1e-12:
        raise ValueError(
            "Measured phasor is zero — cannot compute calibration factor."
        )
    return theory / measured


def _get_napari_color(layer, label_id: int) -> tuple:
    """Return (R, G, B, A) float 0-1 for label_id from a napari Labels layer."""
    try:
        rgba = layer.get_color(label_id)
        return (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))
    except Exception:
        return (0.5, 0.5, 0.5, 0.8)


def _draw_lifetime_ticks(ax, sync_hz: float):
    omega = 2 * np.pi * sync_hz
    tau_max = max(1, int(np.ceil(0.5e9 / sync_hz)))
    center = np.array([0.5, 0.0])
    tick_len = 0.018
    for tau_ns in range(1, tau_max + 1):
        tau_s = tau_ns * 1e-9
        g_t = 1.0 / (1.0 + (omega * tau_s) ** 2)
        s_t = (omega * tau_s) / (1.0 + (omega * tau_s) ** 2)
        p = np.array([g_t, s_t])
        v = p - center
        nrm = np.linalg.norm(v)
        if nrm < 1e-9:
            continue
        v_u = v / nrm
        p1, p2 = p - tick_len / 2 * v_u, p + tick_len / 2 * v_u
        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]], "w-", lw=0.8, alpha=0.6, zorder=4
        )
        lp = p + tick_len * 1.7 * v_u
        ax.text(
            lp[0],
            lp[1],
            f"{tau_ns}ns",
            fontsize=6,
            ha="center",
            va="center",
            color="white",
            alpha=0.7,
            zorder=4,
        )
