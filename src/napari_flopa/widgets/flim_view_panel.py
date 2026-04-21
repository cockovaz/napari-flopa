import traceback

import numpy as np
import xarray as xr
from matplotlib import cm
from qtpy.QtCore import QTimer, Signal, Slot
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from napari_flopa.processing.image_utils import (
    aggregate_dataset,
    smooth_count,
    smooth_weighted,
)
from napari_flopa.state import AppState
from napari_flopa.widgets.histogram_slider import HistogramSlider
from napari_flopa.widgets.style import SS, apply_style

_CANVAS_H = 55  # histogram canvas height in compact layout
_HIST_MAX_W = 300  # histogram slider max width


class FlimViewPanel(QWidget):
    """
    Bottom dock panel for interactive FLIM image display.

    Provides:
      • Frame / sequence / detector spinbox selectors with per-dim "Agg"
        checkboxes that sum over that dimension.
      • Dual HistogramSlider for Intensity and Lifetime — cyan handles set
        display contrast, red handles define the mask threshold range.
      • Smoothing controls (kernel size) for both intensity and lifetime.
      • Colormap selectors; FLIM RGB composite mode when both layers are ON.
      • "→ Int. Mask" / "→ Lt. Mask" buttons create uniquely named Labels
        layers from the current red-slider threshold range.
      • Export buttons for Intensity (uint16 TIFF), Lifetime (float32 TIFF),
        and FLIM RGB composite (PNG or TIFF).

    The widget is hidden until update_data() is called with a valid dataset.
    Emits view_changed(dict) whenever the selection or aggregation changes so
    that PhasorPanel and DecayPanel can mark themselves stale.
    """

    view_changed = Signal(dict)

    def __init__(self, state: AppState, viewer, parent=None):
        super().__init__(parent)
        self.state = state
        self.viewer = viewer
        self.dataset = None
        self._cached_intensity = None
        self._cached_lifetime = None
        self._lut = None  # 256×3 uint8 lifetime colormap LUT
        self._sum_cache = (
            {}
        )  # (sel_tuple, frozenset(dims_to_sum)) → (raw_int, raw_lt)
        self._smooth_cache = {}  # (type, id, shape, k) → smoothed array
        self._layers_created = False
        self._layer_lock_bypass = False
        self.selectors = {}

        # Debounce timer kept as instance attr so it is not GC-collected
        self._debounce = QTimer()
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(80)

        self._build_ui()
        self.setVisible(False)

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(2, 0, 2, 0)

        self.container = QGroupBox("FLIM View")
        apply_style(self.container, SS.GROUP_TITLE)
        self.view_layout = QVBoxLayout(self.container)
        self.view_layout.setContentsMargins(2, 2, 2, 2)
        main_layout.addWidget(self.container)
        main_layout.addStretch()

    # ------------------------------------------------------------------ #

    @Slot(object)
    def update_data(self, dataset: xr.Dataset):
        """
        Called after reconstruction completes.  Stores the dataset, clears
        caches, rebuilds all controls via _create_controls(), and makes the
        widget visible.  Hides the widget if neither photon_count nor
        mean_arrival_time is present.
        """
        self.dataset = dataset
        self._sum_cache.clear()
        self._smooth_cache.clear()
        self._layers_created = False
        has_intensity = "photon_count" in dataset.data_vars
        has_lifetime = "mean_arrival_time" in dataset.data_vars
        if not has_intensity and not has_lifetime:
            self.setVisible(False)
            return
        self.setVisible(True)
        self._create_controls(has_intensity, has_lifetime)

    # ------------------------------------------------------------------ #

    def _create_controls(self, has_intensity: bool, has_lifetime: bool):
        """
        Rebuild all child widgets inside the FLIM View group box.

        Called by update_data() each time a new dataset arrives.  Removes any
        previously created widgets, then lays out:
          col 0 — frame / sequence / channel selectors with Agg checkboxes
          col 1 — Intensity HistogramSlider + controls + mask button
          col 2 — Lifetime HistogramSlider + controls + mask button
          col 3 — Export buttons (Intensity, Lifetime, FLIM RGB)

        All signal wiring (selectors → debounced _slice, sliders → _fast_display,
        colormaps, mask buttons, smoothing, visibility) is established here using
        closures that capture the local helpers defined in this method.

        Inner helpers (not accessible outside this method):
          _make_lut(cmap_name)      — build 256×3 uint8 LUT from a matplotlib cmap
          _fast_flim(ci, cl)        — compute FLIM RGB array from cached arrays + LUT
          _get_sel_key()            — return (sel dict, dims_to_sum list) from selectors
          _get_raw_arrays(sel, sums)— isel + optional aggregate with LRU cache (64 entries)
          _get_smoothed(raw_i, raw_l)— apply smoothing kernels with LRU cache (32 entries)
          _display_data()           — full layer update (data + contrast + colormap)
          _fast_display()           — contrast-only update for slider drag
          _slice()                  — re-extract arrays, update histograms, call _display_data
          _update_colormaps()       — push colormap changes to napari layers + histogram
          _visibility_changed()     — toggle layer visibility when ON checkboxes change
          _next_layer_name(base)    — return a unique layer name (base, base [1], …)
          _create_intensity_mask()  — add Labels layer from intensity red-slider range
          _create_lifetime_mask()   — add Labels layer from lifetime red-slider range
        """
        while self.view_layout.count():
            item = self.view_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        ds = self.dataset
        n_frames = ds.sizes.get("frame", 1)
        n_sequences = ds.sizes.get("sequence", 1)
        n_channels = ds.sizes.get("channel", 1)
        instrument_params = ds.attrs.get("instrument_params", {})
        tcspc_res_ns = instrument_params.get("tcspc_resolution_ns", 1.0)
        lifetime_unit = instrument_params.get("resolution_unit", "ch")

        grid = QGridLayout()
        grid.setSpacing(2)
        # col 0: selectors (narrow), cols 1-2: histogram groups (wide), col 3: export (medium)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 4)
        grid.setColumnStretch(2, 4)
        grid.setColumnStretch(3, 1)
        self.view_layout.addLayout(grid)

        # ---- Col 0: slicing selectors ----
        sel_group = QGroupBox()
        apply_style(sel_group, SS.GROUP_A)
        sel_group.setFlat(True)
        sf = QGridLayout(sel_group)
        sf.setVerticalSpacing(1)

        frame_sel = QSpinBox()
        frame_sel.setRange(0, max(0, n_frames - 1))
        frame_sel.setMaximumWidth(50)
        sum_frames = QCheckBox("Agg")
        sum_frames.setEnabled(n_frames > 1)
        sum_frames.setToolTip("Sum all frames into one image")
        seq_sel = QSpinBox()
        seq_sel.setRange(0, max(0, n_sequences - 1))
        seq_sel.setMaximumWidth(50)
        sum_seqs = QCheckBox("Agg")
        sum_seqs.setEnabled(n_sequences > 1)
        sum_seqs.setToolTip("Sum all sequences into one image")
        chan_sel = QSpinBox()
        chan_sel.setRange(0, max(0, n_channels - 1))
        chan_sel.setMaximumWidth(50)
        sum_chans = QCheckBox("Agg")
        sum_chans.setEnabled(n_channels > 1)
        sum_chans.setToolTip("Sum all detector channels into one image")

        for row, (lbl, spin, chk) in enumerate(
            [
                ("Frame:", frame_sel, sum_frames),
                ("Seq:", seq_sel, sum_seqs),
                ("Det:", chan_sel, sum_chans),
            ]
        ):
            sf.addWidget(QLabel(lbl), row, 0)
            sf.addWidget(spin, row, 1)
            sf.addWidget(chk, row, 2)

        grid.addWidget(sel_group, 0, 0)
        self.selectors = {
            "frame": frame_sel,
            "sequence": seq_sel,
            "channel": chan_sel,
            "sum_frames": sum_frames,
            "sum_sequences": sum_seqs,
            "sum_channels": sum_chans,
        }

        # ---- Col 1: Intensity ----
        int_group = QGroupBox("Intensity")
        apply_style(int_group, SS.GROUP_A)
        int_group.setEnabled(has_intensity)
        ig = QHBoxLayout(int_group)
        ig.setSpacing(1)
        ig.setContentsMargins(4, 2, 4, 2)

        self.intensity_slider = HistogramSlider(
            integer_mode=True, canvas_height=_CANVAS_H
        )
        self.intensity_slider.setMaximumWidth(_HIST_MAX_W)
        self.intensity_slider.setToolTip(
            "Cyan slider: display contrast range\n"
            "Red slider: intensity threshold for mask creation"
        )
        ig.addWidget(self.intensity_slider, stretch=3)

        self.intensity_slider.set_name("Counts")

        int_ctrl = QVBoxLayout()
        int_ctrl.setSpacing(2)
        int_ctrl.setContentsMargins(6, 0, 6, 0)

        int_row1 = QHBoxLayout()
        int_row1.setSpacing(2)
        self.show_intensity = QCheckBox("ON")
        self.show_intensity.setChecked(True)
        int_row1.addWidget(self.show_intensity)
        self.smooth_int_check = QCheckBox("Smooth")
        int_row1.addWidget(self.smooth_int_check)
        self.smooth_int_spin = QSpinBox()
        self.smooth_int_spin.setRange(2, 50)
        self.smooth_int_spin.setValue(3)
        self.smooth_int_spin.setEnabled(False)
        self.smooth_int_check.toggled.connect(self.smooth_int_spin.setEnabled)
        int_row1.addWidget(self.smooth_int_spin)
        int_ctrl.addLayout(int_row1)

        self.int_colormap = QComboBox()
        self.int_colormap.addItems(["gray", "viridis", "magma"])
        int_ctrl.addWidget(self.int_colormap)

        int_contrast = QHBoxLayout()
        int_contrast.setSpacing(2)

        self.int_auto_btn = QPushButton("Auto contrast")
        self.int_auto_btn.setToolTip("Set contrast to [p2, p98]")
        int_contrast.addWidget(self.int_auto_btn)
        self.int_minmax_btn = QPushButton("Min-Max")
        self.int_minmax_btn.setToolTip("Set contrast to full data range")
        int_contrast.addWidget(self.int_minmax_btn)
        int_ctrl.addLayout(int_contrast)

        self.int_mask_btn = QPushButton("→ Generate Int. Mask")
        self.int_mask_btn.setStyleSheet(SS.BTN_DANGER)
        self.int_mask_btn.setToolTip(
            "Create a new Labels layer from pixels within the red slider range."
        )
        self.int_mask_btn.setEnabled(False)
        int_ctrl.addWidget(self.int_mask_btn)
        int_ctrl.addStretch()
        ig.addLayout(int_ctrl)

        grid.addWidget(int_group, 0, 1)

        # ---- Col 2: Lifetime ----
        lt_group = QGroupBox(f"Lifetime ({lifetime_unit})")
        apply_style(lt_group, SS.GROUP_A)
        lt_group.setEnabled(has_lifetime)
        lg = QHBoxLayout(lt_group)
        lg.setSpacing(1)
        lg.setContentsMargins(4, 2, 4, 2)

        self.lifetime_slider = HistogramSlider(
            integer_mode=False, canvas_height=_CANVAS_H
        )
        self.lifetime_slider.setMaximumWidth(_HIST_MAX_W)
        self.lifetime_slider.setToolTip(
            "Cyan slider: display contrast range\n"
            "Red slider: lifetime threshold for mask creation"
        )
        lg.addWidget(self.lifetime_slider, stretch=3)

        self.lifetime_slider.set_name("Counts")

        lt_ctrl = QVBoxLayout()
        lt_ctrl.setSpacing(2)
        lt_ctrl.setContentsMargins(6, 0, 6, 0)

        lt_row1 = QHBoxLayout()
        lt_row1.setSpacing(2)
        self.show_lifetime = QCheckBox("ON")
        self.show_lifetime.setChecked(True)
        lt_row1.addWidget(self.show_lifetime)
        self.smooth_lt_check = QCheckBox("Smooth")
        lt_row1.addWidget(self.smooth_lt_check)
        self.smooth_lt_spin = QSpinBox()
        self.smooth_lt_spin.setRange(2, 50)
        self.smooth_lt_spin.setValue(3)
        self.smooth_lt_spin.setEnabled(False)
        self.smooth_lt_check.toggled.connect(self.smooth_lt_spin.setEnabled)
        lt_row1.addWidget(self.smooth_lt_spin)
        lt_ctrl.addLayout(lt_row1)

        self.lt_colormap = QComboBox()
        self.lt_colormap.addItems(["rainbow", "hsv", "viridis"])
        lt_ctrl.addWidget(self.lt_colormap)

        lt_contrast = QHBoxLayout()
        lt_contrast.setSpacing(2)
        self.lt_auto_btn = QPushButton("Auto contrast")
        self.lt_auto_btn.setToolTip("Set contrast to [p2, p98]")
        lt_contrast.addWidget(self.lt_auto_btn)
        self.lt_minmax_btn = QPushButton("Min-Max")
        self.lt_minmax_btn.setToolTip("Set contrast to full data range")
        lt_contrast.addWidget(self.lt_minmax_btn)
        lt_ctrl.addLayout(lt_contrast)

        self.lt_mask_btn = QPushButton("→ Generate Lt. Mask")
        self.lt_mask_btn.setStyleSheet(SS.BTN_DANGER)
        self.lt_mask_btn.setToolTip(
            "Create a new Labels layer from pixels within the red slider range."
        )
        self.lt_mask_btn.setEnabled(False)
        lt_ctrl.addWidget(self.lt_mask_btn)
        lt_ctrl.addStretch()
        lg.addLayout(lt_ctrl)

        grid.addWidget(lt_group, 0, 2)

        # ---- Col 3: Export ----
        exp_group = QGroupBox("Export")
        apply_style(exp_group, SS.GROUP_A)
        el = QVBoxLayout(exp_group)
        el.setSpacing(2)
        el.setContentsMargins(4, 2, 4, 2)
        self.export_int_btn = QPushButton("Intensity")
        self.export_lt_btn = QPushButton("Lifetime")
        self.export_flim_btn = QPushButton("FLIM RGB")
        for btn in (
            self.export_int_btn,
            self.export_lt_btn,
            self.export_flim_btn,
        ):
            btn.setEnabled(False)
            el.addWidget(btn)
        # el.addStretch()
        grid.addWidget(exp_group, 0, 3)

        self.export_int_btn.clicked.connect(self._export_intensity)
        self.export_lt_btn.clicked.connect(self._export_lifetime)
        self.export_flim_btn.clicked.connect(self._export_flim)

        self.int_auto_btn.clicked.connect(
            self.intensity_slider.set_auto_contrast
        )
        self.int_minmax_btn.clicked.connect(self.intensity_slider.set_min_max)
        self.lt_auto_btn.clicked.connect(
            self.lifetime_slider.set_auto_contrast
        )
        self.lt_minmax_btn.clicked.connect(self.lifetime_slider.set_min_max)

        def _sync_int_histogram_colormap():
            """In FLIM mode intensity histogram always shows gray; otherwise follow combo."""
            show_i = self.show_intensity.isChecked() and has_intensity
            show_l = self.show_lifetime.isChecked() and has_lifetime
            flim_mode = show_i and show_l
            self.int_colormap.setEnabled(not flim_mode)
            cmap = (
                cm.gray
                if flim_mode
                else cm.get_cmap(self.int_colormap.currentText())
            )
            self.intensity_slider.set_colormap(cmap)

        self.show_intensity.toggled.connect(
            lambda _: _sync_int_histogram_colormap()
        )
        self.show_lifetime.toggled.connect(
            lambda _: _sync_int_histogram_colormap()
        )
        self.int_colormap.currentTextChanged.connect(
            lambda _: _sync_int_histogram_colormap()
        )
        _sync_int_histogram_colormap()

        # ------------------------------------------------------------------ #
        # Helpers                                                              #
        # ------------------------------------------------------------------ #

        def _set_visible(name: str, value: bool):
            """Set layer visibility from panel code, bypassing the user lock."""
            if name in self.viewer.layers:
                self._layer_lock_bypass = True
                self.viewer.layers[name].visible = value
                self._layer_lock_bypass = False

        def _make_lut(cmap_name: str) -> np.ndarray:
            cmap = cm.get_cmap(cmap_name)
            return (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)

        def _fast_flim(ci, cl):
            lt_lo, lt_hi = self.lifetime_slider.value()
            lt_range = lt_hi - lt_lo if lt_hi > lt_lo else 1.0
            cl_f = np.where(np.isfinite(cl), cl, lt_lo).astype(np.float32)
            lt_idx = np.clip((cl_f - lt_lo) / lt_range * 255, 0, 255).astype(
                np.uint8
            )
            int_lo, int_hi = self.intensity_slider.value()
            int_range = int_hi - int_lo if int_hi > int_lo else 1.0
            ci_f = np.where(np.isfinite(ci), ci, int_lo).astype(np.float32)
            int_norm = np.clip((ci_f - int_lo) / int_range, 0, 1)
            return (self._lut[lt_idx].astype(np.float32) / 255.0) * int_norm[
                ..., np.newaxis
            ]

        def _get_sel_key():
            sel, dims_to_sum = {}, []
            for dim in ["frame", "sequence", "channel"]:
                sum_key = f"sum_{dim}s"
                if dim in ds.sizes and self.selectors[sum_key].isChecked():
                    dims_to_sum.append(dim)
                elif dim in ds.sizes:
                    sel[dim] = self.selectors[dim].value()
            return sel, dims_to_sum

        def _get_raw_arrays(sel, dims_to_sum):
            cache_key = (tuple(sorted(sel.items())), frozenset(dims_to_sum))
            if cache_key in self._sum_cache:
                return self._sum_cache[cache_key]
            sliced = ds.isel(**sel)
            final = (
                aggregate_dataset(sliced, dims_to_sum)
                if dims_to_sum
                else sliced
            )
            raw_int = (
                final["photon_count"].values.squeeze()
                if "photon_count" in final
                else None
            )
            raw_lt = (
                final["mean_arrival_time"].values.squeeze()
                if "mean_arrival_time" in final
                else None
            )
            self._sum_cache[cache_key] = (raw_int, raw_lt)
            if len(self._sum_cache) > 64:
                self._sum_cache.pop(next(iter(self._sum_cache)))
            return raw_int, raw_lt

        def _get_smoothed(raw_int, raw_lt):
            s_int = raw_int
            if (
                has_intensity
                and raw_int is not None
                and self.smooth_int_check.isChecked()
            ):
                k = self.smooth_int_spin.value()
                key = ("int", id(raw_int), raw_int.shape, k)
                if key not in self._smooth_cache:
                    self._smooth_cache[key] = smooth_count(raw_int, size=k)
                    if len(self._smooth_cache) > 32:
                        self._smooth_cache.pop(next(iter(self._smooth_cache)))
                s_int = self._smooth_cache[key]
            s_lt = raw_lt
            if (
                has_lifetime
                and raw_lt is not None
                and raw_int is not None
                and self.smooth_lt_check.isChecked()
            ):
                k = self.smooth_lt_spin.value()
                key = ("lt", id(raw_lt), raw_lt.shape, k)
                if key not in self._smooth_cache:
                    result, _ = smooth_weighted(raw_lt, raw_int, size=k)
                    self._smooth_cache[key] = result
                    if len(self._smooth_cache) > 32:
                        self._smooth_cache.pop(next(iter(self._smooth_cache)))
                s_lt = self._smooth_cache[key]
            return s_int, s_lt

        # ------------------------------------------------------------------ #
        # Display                                                              #
        # ------------------------------------------------------------------ #

        def _display_data():
            try:
                ci = self._cached_intensity
                cl = self._cached_lifetime
                show_i = self.show_intensity.isChecked() and has_intensity
                show_l = self.show_lifetime.isChecked() and has_lifetime
                flim_mode = show_i and show_l
                int_mode = show_i and not flim_mode
                lt_mode = show_l and not flim_mode

                if not self._layers_created:
                    for name in ["FLIM", "Intensity", "Lifetime"]:
                        if name in self.viewer.layers:
                            self.viewer.layers.remove(name)
                    if ci is not None and cl is not None:
                        self.viewer.add_image(
                            np.zeros((*ci.shape, 3), dtype=np.float32),
                            name="FLIM",
                            rgb=True,
                        )
                        self._lock_layer(self.viewer.layers["FLIM"])
                    if ci is not None:
                        self.viewer.add_image(
                            ci, name="Intensity", colormap="gray"
                        )
                        self._lock_layer(self.viewer.layers["Intensity"])
                    if cl is not None:
                        self.viewer.add_image(
                            cl, name="Lifetime", colormap="rainbow"
                        )
                        self._lock_layer(self.viewer.layers["Lifetime"])
                    self._layers_created = True

                _set_visible("FLIM", flim_mode)
                if "FLIM" in self.viewer.layers:
                    if (
                        flim_mode
                        and ci is not None
                        and cl is not None
                        and self._lut is not None
                    ):
                        self.viewer.layers["FLIM"].data = _fast_flim(ci, cl)

                _set_visible("Intensity", int_mode)
                if "Intensity" in self.viewer.layers:
                    if ci is not None:
                        self.viewer.layers["Intensity"].data = ci
                        if int_mode:
                            self.viewer.layers["Intensity"].contrast_limits = (
                                self.intensity_slider.value()
                            )
                            self.viewer.layers["Intensity"].colormap = (
                                self.int_colormap.currentText()
                            )

                _set_visible("Lifetime", lt_mode)
                if "Lifetime" in self.viewer.layers:
                    if cl is not None:
                        self.viewer.layers["Lifetime"].data = cl
                        if lt_mode:
                            self.viewer.layers["Lifetime"].contrast_limits = (
                                self.lifetime_slider.value()
                            )
                            self.viewer.layers["Lifetime"].colormap = (
                                self.lt_colormap.currentText()
                            )

                # enable export buttons whenever we have data
                self.export_int_btn.setEnabled(ci is not None)
                self.export_lt_btn.setEnabled(cl is not None)
                self.export_flim_btn.setEnabled(
                    ci is not None and cl is not None and self._lut is not None
                )
            except Exception:
                traceback.print_exc()

        def _fast_display():
            """Slider drag — only update contrast/FLIM, no data copy."""
            try:
                ci = self._cached_intensity
                cl = self._cached_lifetime
                show_i = self.show_intensity.isChecked() and has_intensity
                show_l = self.show_lifetime.isChecked() and has_lifetime
                flim_mode = show_i and show_l
                if (
                    flim_mode
                    and ci is not None
                    and cl is not None
                    and self._lut is not None
                ):
                    if "FLIM" in self.viewer.layers:
                        self.viewer.layers["FLIM"].data = _fast_flim(ci, cl)
                else:
                    if (
                        show_i
                        and "Intensity" in self.viewer.layers
                        and ci is not None
                    ):
                        self.viewer.layers["Intensity"].contrast_limits = (
                            self.intensity_slider.value()
                        )
                    if (
                        show_l
                        and "Lifetime" in self.viewer.layers
                        and cl is not None
                    ):
                        self.viewer.layers["Lifetime"].contrast_limits = (
                            self.lifetime_slider.value()
                        )
            except Exception:
                traceback.print_exc()

        _first_slice = [True]

        def _reset_and_slice():
            """Force a full contrast reset on the next slice (for smooth/aggregate changes)."""
            _first_slice[0] = True
            _slice()

        def _slice():
            try:
                sel, dims_to_sum = _get_sel_key()
                raw_int, raw_lt = _get_raw_arrays(sel, dims_to_sum)
                s_int, s_lt = _get_smoothed(raw_int, raw_lt)
                self._cached_intensity = (
                    np.atleast_2d(s_int) if s_int is not None else None
                )
                self._cached_lifetime = (
                    np.atleast_2d(s_lt) * tcspc_res_ns
                    if s_lt is not None
                    else None
                )
                update_slider = (
                    self.intensity_slider.update_data
                    if _first_slice[0]
                    else self.intensity_slider.update_data_keep_range
                )
                update_lt_slider = (
                    self.lifetime_slider.update_data
                    if _first_slice[0]
                    else self.lifetime_slider.update_data_keep_range
                )
                if self._cached_intensity is not None:
                    update_slider(self._cached_intensity)
                if self._cached_lifetime is not None:
                    update_lt_slider(self._cached_lifetime)
                _first_slice[0] = False
                _display_data()
                # Notify phasor/decay panels about current view settings
                self.view_changed.emit(
                    {
                        "frame": self.selectors["frame"].value(),
                        "sequence": self.selectors["sequence"].value(),
                        "channel": self.selectors["channel"].value(),
                        "sum_frames": self.selectors["sum_frames"].isChecked(),
                        "sum_sequences": self.selectors[
                            "sum_sequences"
                        ].isChecked(),
                        "sum_channels": self.selectors[
                            "sum_channels"
                        ].isChecked(),
                    }
                )
            except Exception:
                traceback.print_exc()

        def _update_colormaps():
            if has_intensity:
                self.intensity_slider.set_colormap(
                    cm.get_cmap(self.int_colormap.currentText())
                )
            if has_lifetime:
                self.lifetime_slider.set_colormap(
                    cm.get_cmap(self.lt_colormap.currentText())
                )
                self._lut = _make_lut(self.lt_colormap.currentText())
            try:
                show_i = self.show_intensity.isChecked() and has_intensity
                show_l = self.show_lifetime.isChecked() and has_lifetime
                flim_mode = show_i and show_l
                if not flim_mode:
                    if show_i and "Intensity" in self.viewer.layers:
                        self.viewer.layers["Intensity"].colormap = (
                            self.int_colormap.currentText()
                        )
                    if show_l and "Lifetime" in self.viewer.layers:
                        self.viewer.layers["Lifetime"].colormap = (
                            self.lt_colormap.currentText()
                        )
                elif (
                    "FLIM" in self.viewer.layers
                    and self._cached_intensity is not None
                    and self._cached_lifetime is not None
                ):
                    self.viewer.layers["FLIM"].data = _fast_flim(
                        self._cached_intensity, self._cached_lifetime
                    )
            except Exception:
                traceback.print_exc()

        def _visibility_changed():
            try:
                show_i = self.show_intensity.isChecked() and has_intensity
                show_l = self.show_lifetime.isChecked() and has_lifetime
                flim_mode = show_i and show_l
                _set_visible("FLIM", flim_mode)
                if "FLIM" in self.viewer.layers:
                    if (
                        flim_mode
                        and self._cached_intensity is not None
                        and self._cached_lifetime is not None
                        and self._lut is not None
                    ):
                        self.viewer.layers["FLIM"].data = _fast_flim(
                            self._cached_intensity, self._cached_lifetime
                        )
                _set_visible("Intensity", show_i and not flim_mode)
                _set_visible("Lifetime", show_l and not flim_mode)
            except Exception:
                traceback.print_exc()

        def _next_layer_name(base: str) -> str:
            existing = {layer.name for layer in self.viewer.layers}
            if base not in existing:
                return base
            i = 1
            while f"{base} [{i}]" in existing:
                i += 1
            return f"{base} [{i}]"

        def _create_intensity_mask():
            ci = self._cached_intensity
            if ci is None:
                return
            lo, hi = self.intensity_slider.mask_value()
            mask = np.where((ci >= lo) & (ci <= hi), 1, 0).astype(np.int32)
            self.viewer.add_labels(
                mask, name=_next_layer_name("Intensity Mask")
            )

        def _create_lifetime_mask():
            cl = self._cached_lifetime
            if cl is None:
                return
            lo, hi = self.lifetime_slider.mask_value()
            mask = np.where((cl >= lo) & (cl <= hi), 1, 0).astype(np.int32)
            self.viewer.add_labels(
                mask, name=_next_layer_name("Lifetime Mask")
            )

        # ---- wire signals ----
        # Spinboxes → debounce → _slice (sticky contrast on navigation)
        # Agg checkboxes → _reset_and_slice (contrast resets when aggregation changes)
        for w in self.selectors.values():
            if isinstance(w, QSpinBox):
                w.valueChanged.connect(self._debounce.start)
            if isinstance(w, QCheckBox):
                w.toggled.connect(_reset_and_slice)
        self._debounce.timeout.connect(_slice)

        self.smooth_int_check.toggled.connect(_reset_and_slice)
        self.smooth_int_spin.valueChanged.connect(_reset_and_slice)
        self.show_intensity.toggled.connect(_visibility_changed)
        self.int_colormap.currentTextChanged.connect(_update_colormaps)
        self.intensity_slider.valueChanged.connect(_fast_display)
        self.intensity_slider.sliderReleased.connect(_fast_display)
        if has_intensity:
            self.int_mask_btn.setEnabled(True)
            self.int_mask_btn.clicked.connect(_create_intensity_mask)
        if has_lifetime:
            self.lt_mask_btn.setEnabled(True)
            self.lt_mask_btn.clicked.connect(_create_lifetime_mask)

        if has_lifetime:
            self.smooth_lt_check.toggled.connect(_reset_and_slice)
            self.smooth_lt_spin.valueChanged.connect(_reset_and_slice)
            self.show_lifetime.toggled.connect(_visibility_changed)
            self.lt_colormap.currentTextChanged.connect(_update_colormaps)
            self.lifetime_slider.valueChanged.connect(_fast_display)
            self.lifetime_slider.sliderReleased.connect(_fast_display)

        self._lut = (
            _make_lut(self.lt_colormap.currentText()) if has_lifetime else None
        )
        _slice()
        _update_colormaps()

    # ------------------------------------------------------------------ #
    # Export                                                               #
    # ------------------------------------------------------------------ #

    def _lock_layer(self, layer) -> None:
        """Prevent user from modifying a managed FLIM layer from the napari UI.

        Sets editable=False and connects to the visible/opacity events so any
        change made via the layer list is immediately reverted.  Changes made
        by panel code are exempt: they set _layer_lock_bypass=True first via
        the _set_visible helper.
        """
        layer.editable = False

        def _guard_visible():
            if not self._layer_lock_bypass:
                self._layer_lock_bypass = True
                layer.visible = not layer.visible  # revert
                self._layer_lock_bypass = False

        def _guard_opacity():
            if not self._layer_lock_bypass:
                self._layer_lock_bypass = True
                layer.opacity = 1.0  # revert
                self._layer_lock_bypass = False

        layer.events.visible.connect(_guard_visible)
        layer.events.opacity.connect(_guard_opacity)

    def _export_intensity(self):
        """Save cached intensity array as a uint16 TIFF (normalised to 0–65535)."""
        if self._cached_intensity is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Intensity", "intensity.tif", "TIFF (*.tif *.tiff)"
        )
        if not path:
            return
        try:
            from skimage.io import imsave

            arr = self._cached_intensity
            # normalise to uint16 for TIFF
            mn, mx = float(arr.min()), float(arr.max())
            if mx > mn:
                scaled = ((arr - mn) / (mx - mn) * 65535).astype(np.uint16)
            else:
                scaled = np.zeros_like(arr, dtype=np.uint16)
            imsave(path, scaled, check_contrast=False)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _export_lifetime(self):
        """Save cached lifetime array as a float32 TIFF (values in ns)."""
        if self._cached_lifetime is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Lifetime", "lifetime.tif", "TIFF (*.tif *.tiff)"
        )
        if not path:
            return
        try:
            from skimage.io import imsave

            imsave(
                path,
                self._cached_lifetime.astype(np.float32),
                check_contrast=False,
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def _export_flim(self):
        """
        Save the FLIM RGB composite as PNG or TIFF (uint8).

        Lifetime is mapped through the current LUT using the lifetime slider's
        contrast range; intensity modulates brightness via the intensity slider's
        contrast range.
        """
        if (
            self._cached_intensity is None
            or self._cached_lifetime is None
            or self._lut is None
        ):
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export FLIM RGB",
            "flim.png",
            "PNG (*.png);;TIFF (*.tif *.tiff)",
        )
        if not path:
            return
        try:
            from skimage.io import imsave

            lt_lo, lt_hi = self.lifetime_slider.value()
            lt_range = lt_hi - lt_lo if lt_hi > lt_lo else 1.0
            lt_idx = np.clip(
                (self._cached_lifetime.astype(np.float32) - lt_lo)
                / lt_range
                * 255,
                0,
                255,
            ).astype(np.uint8)
            int_lo, int_hi = self.intensity_slider.value()
            int_range = int_hi - int_lo if int_hi > int_lo else 1.0
            int_norm = np.clip(
                (self._cached_intensity.astype(np.float32) - int_lo)
                / int_range,
                0,
                1,
            )
            rgb_f32 = (
                self._lut[lt_idx].astype(np.float32) / 255.0
            ) * int_norm[..., np.newaxis]
            rgb_u8 = (rgb_f32 * 255).clip(0, 255).astype(np.uint8)
            imsave(path, rgb_u8, check_contrast=False)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
