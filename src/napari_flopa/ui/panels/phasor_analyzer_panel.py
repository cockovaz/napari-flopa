import re
import traceback
from contextlib import suppress
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QCompleter,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
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

from napari_flopa.core.processing.clustering import (
    METHOD_TAGS,
    METHODS,
    cluster_points,
    method_available,
)
from napari_flopa.core.processing.phasor import (
    calibration_factor,
    component_fractions_2,
    component_fractions_3,
    lifetime_ticks,
    monoexp_phasor,
    phasor_lifetimes,
)
from napari_flopa.core.processing.phasor_stats import (
    contour_levels,
    covariance_ellipse,
    kde_grid,
)
from napari_flopa.core.processing.table_filter import (
    OPS,
    UNARY_OPS,
    build_clause,
)
from napari_flopa.core.processing.table_filter import (
    evaluate as evaluate_filter,
)
from napari_flopa.ui.state import FlopaState
from napari_flopa.ui.style import MPL, S, apply_style
from napari_flopa.ui.widgets.status_label import StatusLabel

_MAX_TABLE_ROWS = 50_000
_MAX_SCATTER = 300_000
#: Beyond this many groups the per-group overlays become unreadable.
_MAX_OVERLAY_GROUPS = 30


def _auto_point_size(n: int) -> int:
    if n <= 0:
        return 10
    return max(2, min(40, int(200 / n**0.5)))


class PhasorAnalyzerPanel(QWidget):
    """Phasor Analyzer tab — load CSV phasor tables, color-code, calibrate,
    extract groups, compute component fractions."""

    def __init__(self, state: FlopaState, viewer, parent=None):
        super().__init__(parent)
        self.state = state
        self.viewer = viewer
        self._df: pd.DataFrame | None = None
        self._calib_factor: complex = 1.0 + 0j
        self._plotted_once = False
        self._highlight_artist = None
        self._valid_mask: np.ndarray | None = None
        self._plot_g: np.ndarray | None = None
        self._plot_s: np.ndarray | None = None
        # Once the user asks for τφ/τm, the columns follow the calibration,
        # the Apply toggle and f_rep until a new CSV is loaded.
        self._lifetimes_on = False
        # (color_map, labels) of the last categorical plot, for overlays.
        self._group_colors: tuple[dict, np.ndarray] | None = None
        self._cl_param_widgets: dict = {}
        # Row filter as a query expression, re-evaluated on every use so it
        # tracks recomputed columns. Rows are never dropped from _df.
        self._filter_expr = ""
        self._table_rows = np.empty(0, dtype=int)  # df positions in the table
        self._build_ui()
        self.state.dataset_changed.connect(self._sync_frep_from_state)

    # ------------------------------------------------------------------ #
    # UI                                                                   #
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)

        # ── Toolbar ────────────────────────────────────────────────────
        tb = QHBoxLayout()
        self._load_btn = QPushButton("Load CSV")
        self._load_btn.clicked.connect(self._on_load_csv)
        tb.addWidget(self._load_btn)

        self._plot_btn = QPushButton("Plot")
        self._plot_btn.setStyleSheet(S.BTN_RUN)
        self._plot_btn.setEnabled(False)
        self._plot_btn.clicked.connect(self._on_plot)
        tb.addWidget(self._plot_btn)

        tb.addWidget(QLabel("Size:"))
        self._size_spin = QSpinBox()
        self._size_spin.setRange(1, 50)
        self._size_spin.setValue(10)
        self._size_spin.setToolTip("Scatter point size (auto-scaled on plot)")
        self._size_spin.setMaximumWidth(50)
        tb.addWidget(self._size_spin)

        self._filter_lbl = QLabel("")
        self._filter_lbl.setStyleSheet(S.HINT)
        self._filter_lbl.setVisible(False)
        tb.addWidget(self._filter_lbl)
        self._filter_clear_btn = QPushButton("✕")
        self._filter_clear_btn.setToolTip("Clear the row filter")
        self._filter_clear_btn.setStyleSheet(S.BTN_SMALL)
        self._filter_clear_btn.setFixedWidth(22)
        self._filter_clear_btn.setVisible(False)
        self._filter_clear_btn.clicked.connect(lambda: self.set_filter(""))
        tb.addWidget(self._filter_clear_btn)

        tb.addStretch()

        self._tools_btn = QPushButton("Data tools…")
        self._tools_btn.clicked.connect(self._on_open_data_tools)
        tb.addWidget(self._tools_btn)

        self._export_combo = QComboBox()
        self._export_combo.addItem("Save plot…", "plot")
        self._export_combo.addItem("Save table…", "table")
        tb.addWidget(self._export_combo)
        self._save_btn = QPushButton("Save")
        self._save_btn.clicked.connect(self._on_export)
        tb.addWidget(self._save_btn)
        root.addLayout(tb)

        root.addSpacing(8)

        # ── Color by ──────────────────────────────────────────────────
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Color by:"))
        self._color_combo = QComboBox()
        self._color_combo.addItem("None")
        self._color_combo.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        color_row.addWidget(self._color_combo, 1)
        self._unique_label = QLabel("")
        self._unique_label.setStyleSheet(S.HINT)
        self._unique_label.setFixedWidth(60)
        color_row.addWidget(self._unique_label)
        color_row.addWidget(QLabel("Cmap:"))
        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(
            ["tab10", "Set1", "viridis", "plasma", "inferno", "hot", "rainbow"]
        )
        self._cmap_combo.setMaximumWidth(85)
        color_row.addWidget(self._cmap_combo)
        self._legend_check = QCheckBox("Legend")
        self._legend_check.setChecked(True)
        color_row.addWidget(self._legend_check)
        root.addLayout(color_row)

        # ── Per-group overlays ─────────────────────────────────────────
        ov_row = QHBoxLayout()
        ov_row.setSpacing(4)
        ov_lbl = QLabel("Overlays:")
        ov_lbl.setToolTip(
            "Drawn per group of the 'Color by' column (categorical), or for "
            "all points when coloring is None / continuous."
        )
        ov_row.addWidget(ov_lbl)
        self._ov_contour = QCheckBox("Contours")
        self._ov_contour.setToolTip(
            "Density contours enclosing 25 / 50 / 75 % of each group's points"
        )
        self._ov_mean = QCheckBox("Mean")
        self._ov_mean.setToolTip("Group mean, drawn as ✕")
        self._ov_median = QCheckBox("Median")
        self._ov_median.setToolTip("Group median (per axis), drawn as ✚")
        self._ov_ellipse = QCheckBox("Ellipse")
        self._ov_ellipse.setToolTip("Covariance ellipse of each group")
        self._ov_sigma = QSpinBox()
        self._ov_sigma.setRange(1, 3)
        self._ov_sigma.setValue(2)
        self._ov_sigma.setSuffix(" σ")
        self._ov_sigma.setToolTip(
            "Ellipse size: 1σ ≈ 39 %, 2σ ≈ 86 %, 3σ ≈ 99 % of a 2-D Gaussian"
        )
        self._ov_sigma.setEnabled(False)
        self._ov_ellipse.toggled.connect(self._ov_sigma.setEnabled)
        for w in (
            self._ov_contour,
            self._ov_mean,
            self._ov_median,
            self._ov_ellipse,
            self._ov_sigma,
        ):
            ov_row.addWidget(w)
        ov_row.addStretch()
        root.addLayout(ov_row)

        # ── f_rep + τ marks ────────────────────────────────────────────
        freq_row = QHBoxLayout()
        freq_row.addWidget(QLabel("f_rep (MHz):"))
        self._frep_spin = QDoubleSpinBox()
        self._frep_spin.setRange(0.1, 1000.0)
        self._frep_spin.setDecimals(3)
        self._frep_spin.setValue(self.state.frep_mhz)
        self._frep_spin.setToolTip("Laser repetition rate in MHz")
        freq_row.addWidget(self._frep_spin)
        self._lifetimes_check = QCheckBox("τ marks")
        self._lifetimes_check.setToolTip(
            "Overlay mono-exponential lifetime reference marks on the "
            "universal semicircle."
        )
        freq_row.addWidget(self._lifetimes_check)
        freq_row.addStretch()
        root.addLayout(freq_row)

        # ── Calibration factor ─────────────────────────────────────────
        factor_row = QHBoxLayout()
        factor_row.setSpacing(4)
        factor_row.addWidget(QLabel("Factor:"))
        self._calib_display = QLineEdit(self._fmt_complex(self._calib_factor))
        self._calib_display.setReadOnly(True)
        self._calib_display.setStyleSheet(S.DISPLAY)
        self._calib_display.setMaximumWidth(160)
        factor_row.addWidget(self._calib_display)

        calc_btn = QPushButton("Calculate")
        calc_btn.setToolTip(
            "Enter τ_ref (ns) + measured G, S → compute factor"
        )
        calc_btn.clicked.connect(self._on_calculate_cal)
        factor_row.addWidget(calc_btn)

        custom_btn = QPushButton("Custom")
        custom_btn.setToolTip("Enter a custom calibration factor directly")
        custom_btn.clicked.connect(self._on_custom_factor)
        factor_row.addWidget(custom_btn)

        reset_btn = QPushButton("Reset")
        reset_btn.setToolTip("Reset calibration factor to 1+0j")
        reset_btn.clicked.connect(self._reset_calib)
        factor_row.addWidget(reset_btn)

        self._apply_cal_check = QCheckBox("Apply")
        self._apply_cal_check.setToolTip(
            "Write calibrated_g / calibrated_s columns and use them "
            "for plotting."
        )
        factor_row.addWidget(self._apply_cal_check)
        factor_row.addStretch()
        self._lt_btn = QPushButton("Phase / Mod. τ")
        self._lt_btn.setToolTip(
            "Add tau_phi_ns / tau_mod_ns columns (phase and modulation "
            "lifetimes).\nThey use calibrated G,S when Apply is on, and "
            "follow later calibration and f_rep changes."
        )
        self._lt_btn.clicked.connect(self._on_compute_lifetimes)
        factor_row.addWidget(self._lt_btn)
        root.addLayout(factor_row)

        # ── Components ─────────────────────────────────────────────────
        comp_box = QGroupBox("Component fractions")
        apply_style(comp_box, S.GROUP_CHECKABLE)
        comp_box.setCheckable(True)
        comp_box.setChecked(False)
        comp_lay = QVBoxLayout(comp_box)
        comp_lay.setSpacing(3)
        comp_lay.setContentsMargins(6, 4, 6, 4)

        tau_row = QHBoxLayout()
        self._tau_spins = []
        for i in range(3):
            sp = QDoubleSpinBox()
            sp.setRange(0.0, 1000.0)
            sp.setDecimals(3)
            sp.setSingleStep(0.1)
            sp.setValue(0.0)
            sp.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            tau_row.addWidget(QLabel(f"τ{i + 1}:"))
            tau_row.addWidget(sp, stretch=1)
            self._tau_spins.append(sp)
        comp_lay.addLayout(tau_row)

        comp_btn_row = QHBoxLayout()
        self._comp_compute_btn = QPushButton("Compute fractions")
        self._comp_compute_btn.clicked.connect(self._on_compute_fractions)
        comp_btn_row.addWidget(self._comp_compute_btn)
        self._comp_show_check = QCheckBox("Show on plot")
        self._comp_show_check.setChecked(True)
        comp_btn_row.addWidget(self._comp_show_check)
        comp_btn_row.addStretch()
        comp_lay.addLayout(comp_btn_row)

        self._comp_group = comp_box
        root.addWidget(comp_box)

        # ── Clustering ─────────────────────────────────────────────────
        cl_box = QGroupBox("Clustering")
        apply_style(cl_box, S.GROUP_PRIMARY)
        cl_lay = QVBoxLayout(cl_box)
        cl_lay.setSpacing(3)
        cl_lay.setContentsMargins(6, 4, 6, 4)

        cl_row = QHBoxLayout()
        cl_row.addWidget(QLabel("Method:"))
        self._cl_method = QComboBox()
        first_ok = -1
        for i, name in enumerate(METHODS):
            self._cl_method.addItem(name)
            if method_available(name):
                if first_ok < 0:
                    first_ok = i
                continue
            item = self._cl_method.model().item(i)
            item.setEnabled(False)
            self._cl_method.setItemData(
                i,
                f"Needs {METHODS[name]['needs']}: pip install scikit-learn",
                Qt.ToolTipRole,
            )
        self._cl_method.setCurrentIndex(max(0, first_ok))
        self._cl_method.currentTextChanged.connect(
            self._rebuild_cluster_params
        )
        cl_row.addWidget(self._cl_method)
        self._cl_run_btn = QPushButton("Run")
        self._cl_run_btn.setToolTip(
            "Cluster the plotted G,S (calibrated when Apply is on), ignoring "
            "groups.\nAdds a cluster_<method> column and colors by it."
        )
        self._cl_run_btn.clicked.connect(self._on_run_clustering)
        cl_row.addStretch()
        cl_row.addWidget(self._cl_run_btn)
        cl_lay.addLayout(cl_row)

        self._cl_params_lay = QHBoxLayout()
        self._cl_params_lay.setSpacing(4)
        cl_lay.addLayout(self._cl_params_lay)
        self._rebuild_cluster_params()

        root.addWidget(cl_box)

        # ── Canvas + table splitter ────────────────────────────────────
        splitter = QSplitter(Qt.Vertical)

        cw = QWidget()
        cw_lay = QHBoxLayout(cw)
        cw_lay.setContentsMargins(0, 0, 0, 0)
        cw_lay.setSpacing(0)
        self._fig = Figure(figsize=(3.6, 3.6), tight_layout=True)
        self._fig.patch.set_facecolor(MPL.FIG_BG)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        cw_lay.addWidget(self._canvas, stretch=1)

        self._legend_widget = QWidget()
        self._legend_layout = QVBoxLayout(self._legend_widget)
        self._legend_layout.setContentsMargins(4, 4, 4, 4)
        self._legend_layout.setSpacing(2)
        self._legend_layout.addStretch()

        self._legend_scroll = QScrollArea()
        self._legend_scroll.setWidget(self._legend_widget)
        self._legend_scroll.setWidgetResizable(True)
        self._legend_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._legend_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._legend_scroll.setFixedWidth(120)
        self._legend_scroll.setStyleSheet(
            f"QScrollArea {{ background: {MPL.FIG_BG}; border: none; }}"
        )
        self._legend_scroll.setVisible(False)
        cw_lay.addWidget(self._legend_scroll)

        splitter.addWidget(cw)

        self._table = QTableWidget(0, 0)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setFixedHeight(180)
        splitter.addWidget(self._table)

        root.addWidget(splitter, stretch=1)

        # ── Status ─────────────────────────────────────────────────────
        self._status = StatusLabel(
            "Load a CSV file with phasor G and S columns."
        )
        root.addWidget(self._status)

        # Initial empty axes
        self._ax = self._fig.add_subplot(111)
        self._ax.set_facecolor(MPL.AXES_BG)
        self._draw_empty()

        # ── Live-update wiring ─────────────────────────────────────────
        self._color_combo.currentTextChanged.connect(self._on_setting_changed)
        self._cmap_combo.currentTextChanged.connect(self._on_setting_changed)
        self._lifetimes_check.toggled.connect(self._on_setting_changed)
        self._legend_check.toggled.connect(self._on_setting_changed)
        self._size_spin.valueChanged.connect(self._on_setting_changed)
        self._frep_spin.valueChanged.connect(self._on_frep_changed)
        for w in (
            self._ov_contour,
            self._ov_mean,
            self._ov_median,
            self._ov_ellipse,
        ):
            w.toggled.connect(self._on_setting_changed)
        self._ov_sigma.valueChanged.connect(self._on_setting_changed)
        self._apply_cal_check.toggled.connect(self._on_apply_cal_toggled)
        self._comp_show_check.toggled.connect(self._on_setting_changed)
        self._comp_group.toggled.connect(self._on_setting_changed)
        for sp in self._tau_spins:
            sp.valueChanged.connect(self._on_setting_changed)

        self._table.itemSelectionChanged.connect(self._on_table_selection)

    # ------------------------------------------------------------------ #
    # Live-update                                                          #
    # ------------------------------------------------------------------ #

    def _on_setting_changed(self, *_):
        self._update_unique_label()
        if self._plotted_once and self._df is not None:
            self._on_plot()

    def _update_unique_label(self):
        col = self._color_combo.currentText()
        if self._df is None or col == "None" or col not in self._df.columns:
            self._unique_label.setText("")
            return
        n_unique = self._df[col].nunique()
        n_rows = len(self._df)
        self._unique_label.setText(f"{n_unique}/{n_rows}")

    # ------------------------------------------------------------------ #
    # CSV loading                                                          #
    # ------------------------------------------------------------------ #

    def _on_load_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Phasor CSV", "", "CSV (*.csv);;All Files (*)"
        )
        if not path:
            return
        self.load_csv(path)

    def load_csv(self, path):
        """Read a CSV into the analyzer.  Requires ``g`` and ``s`` columns."""
        try:
            df = pd.read_csv(path)
        except Exception as e:
            self._status.error(f"Failed to read CSV: {e}")
            return
        if "g" not in df.columns or "s" not in df.columns:
            self._status.error("CSV must have 'g' and 's' columns.")
            return
        self._df = df
        self._plotted_once = False
        self._lifetimes_on = False
        self._filter_expr = ""
        self._update_filter_indicator()
        self._refresh_column_combos()
        self._fill_table()
        self._plot_btn.setEnabled(True)
        self._size_spin.setValue(_auto_point_size(len(df)))
        self._status.info(
            f"Loaded {len(df):,} rows, {len(df.columns)} columns "
            f"from {Path(path).name}"
        )
        self._on_plot()

    def _sync_frep_from_state(self):
        self._frep_spin.setValue(self.state.frep_mhz)

    def _refresh_column_combos(self):
        if self._df is None:
            return
        cols = list(self._df.columns)

        cur_color = self._color_combo.currentText()
        self._color_combo.blockSignals(True)
        self._color_combo.clear()
        self._color_combo.addItem("None")
        self._color_combo.addItems(cols)
        idx = self._color_combo.findText(cur_color)
        self._color_combo.setCurrentIndex(max(0, idx))
        self._color_combo.blockSignals(False)
        self._update_unique_label()

    # ------------------------------------------------------------------ #
    # Row filter                                                           #
    # ------------------------------------------------------------------ #

    def _filter_mask(self) -> np.ndarray | None:
        """Rows passing the filter, or None when no filter is active."""
        if self._df is None or not self._filter_expr:
            return None
        try:
            return evaluate_filter(self._df, self._filter_expr)
        except ValueError as e:
            self._status.error(f"Filter ignored — {e}")
            return None

    def set_filter(self, expr: str) -> bool:
        """Activate a filter expression ("" clears it). False if invalid."""
        expr = expr.strip()
        if expr and self._df is not None:
            try:
                evaluate_filter(self._df, expr)
            except ValueError as e:
                self._status.error(f"Invalid filter: {e}")
                return False
        self._filter_expr = expr
        self._update_filter_indicator()
        self._fill_table()
        if self._plotted_once and self._df is not None:
            self._on_plot()
        return True

    def _update_filter_indicator(self):
        mask = self._filter_mask()
        active = mask is not None
        self._filter_lbl.setVisible(active)
        self._filter_clear_btn.setVisible(active)
        if not active:
            return
        expr = self._filter_expr
        short = expr if len(expr) <= 30 else expr[:29] + "…"
        self._filter_lbl.setText(
            f"Filter: {short} · {int(mask.sum()):,}/{len(mask):,}"
        )
        self._filter_lbl.setToolTip(f"Active row filter:\n{expr}")

    def _filtered_df(self) -> pd.DataFrame:
        mask = self._filter_mask()
        return self._df if mask is None else self._df[mask]

    def _fill_table(self):
        if self._df is None:
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            self._table_rows = np.empty(0, dtype=int)
            return
        df = self._df
        mask = self._filter_mask()
        rows = np.flatnonzero(mask) if mask is not None else np.arange(len(df))
        self._update_filter_indicator()
        n_rows = min(len(rows), _MAX_TABLE_ROWS)
        self._table_rows = rows[:n_rows]
        n_cols = len(df.columns)
        self._table.blockSignals(True)
        self._table.setRowCount(n_rows)
        self._table.setColumnCount(n_cols)
        self._table.setHorizontalHeaderLabels(list(df.columns))
        for r, pos in enumerate(self._table_rows):
            for c in range(n_cols):
                val = df.iloc[pos, c]
                if isinstance(val, float):
                    text = f"{val:.4f}" if abs(val) < 1e6 else f"{val:.2e}"
                else:
                    text = str(val)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                self._table.setItem(r, c, item)
        self._table.resizeColumnsToContents()
        self._table.blockSignals(False)
        if n_rows < len(rows):
            self._status.warn(
                f"Table shows first {n_rows:,} of {len(rows):,} rows."
            )

    # ------------------------------------------------------------------ #
    # Table ↔ plot selection                                               #
    # ------------------------------------------------------------------ #

    def _on_table_selection(self):
        if self._plot_g is None or self._valid_mask is None:
            return
        rows = sorted({idx.row() for idx in self._table.selectedIndexes()})
        if not rows:
            self._clear_highlight()
            return
        sel = []
        for r in rows:
            if r < len(self._table_rows):
                pos = self._table_rows[r]
                if self._valid_mask[pos]:
                    sel.append(pos)
        if not sel:
            self._clear_highlight()
            return
        sel_arr = np.array(sel)
        g_sel = self._plot_g[sel_arr]
        s_sel = self._plot_s[sel_arr]
        self._clear_highlight()
        pt_size = self._size_spin.value()
        self._highlight_artist = self._ax.scatter(
            g_sel,
            s_sel,
            s=pt_size * 6,
            facecolors="none",
            edgecolors="yellow",
            linewidths=1.5,
            zorder=20,
        )
        self._canvas.draw_idle()

    def _clear_highlight(self):
        if self._highlight_artist is not None:
            with suppress(Exception):
                self._highlight_artist.remove()

            self._highlight_artist = None
            self._canvas.draw_idle()

    # ------------------------------------------------------------------ #
    # Plotting                                                             #
    # ------------------------------------------------------------------ #

    def _on_plot(self):
        if self._df is None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._do_plot()
            self._plotted_once = True
        except Exception as e:
            traceback.print_exc()
            self._status.error(f"Plot error: {e}")
        finally:
            QApplication.restoreOverrideCursor()

    def _current_gs(self) -> tuple[np.ndarray, np.ndarray, bool]:
        """``(g, s, calibrated)`` — the pair everything plots and computes on."""
        df = self._df
        use_cal = (
            self._apply_cal_check.isChecked()
            and "calibrated_g" in df.columns
            and "calibrated_s" in df.columns
        )
        gc, sc = ("calibrated_g", "calibrated_s") if use_cal else ("g", "s")
        return (
            df[gc].values.astype(np.float64),
            df[sc].values.astype(np.float64),
            use_cal,
        )

    def _do_plot(self):
        df = self._df
        g, s, _ = self._current_gs()

        valid = np.isfinite(g) & np.isfinite(s)
        fmask = self._filter_mask()
        if fmask is not None:
            valid &= fmask
        self._valid_mask = valid
        self._plot_g = g
        self._plot_s = s
        g_v, s_v = g[valid], s[valid]

        self._legend_scroll.setVisible(False)

        fig = self._fig
        fig.clear()
        ax = fig.add_subplot(111)
        self._ax = ax
        ax.set_facecolor(MPL.AXES_BG)
        fig.patch.set_facecolor(MPL.FIG_BG)
        self._highlight_artist = None

        theta = np.linspace(0, np.pi, 300)
        ax.plot(
            0.5 + 0.5 * np.cos(theta),
            0.5 * np.sin(theta),
            color="white",
            linewidth=0.8,
            alpha=0.5,
            zorder=5,
        )

        f_rep_hz = self._frep_spin.value() * 1e6
        if self._lifetimes_check.isChecked():
            _draw_lifetime_ticks(ax, f_rep_hz)

        pt_size = self._size_spin.value()
        self._group_colors = None

        color_col = self._color_combo.currentText()
        if color_col == "None" or color_col not in df.columns:
            self._plot_white(ax, g_v, s_v, pt_size)
        else:
            color_data = df[color_col].values[valid]
            cmap_name = self._cmap_combo.currentText()

            col_series = pd.Series(color_data)
            if pd.api.types.is_numeric_dtype(col_series):
                n_unique = col_series.dropna().nunique()
                is_categorical = n_unique <= 12
            else:
                is_categorical = True

            if is_categorical:
                self._plot_categorical(
                    ax, g_v, s_v, color_data, cmap_name, pt_size
                )
            else:
                self._plot_continuous(
                    ax, g_v, s_v, color_data, cmap_name, pt_size
                )

        self._draw_group_overlays(ax, g_v, s_v)

        if self._comp_group.isChecked() and self._comp_show_check.isChecked():
            self._draw_components(ax, f_rep_hz)

        self._finalise_axes(ax)
        self._canvas.draw_idle()
        if fmask is None:
            self._status.info(f"Plotted {valid.sum():,} points.")
        else:
            self._status.info(
                f"Plotted {valid.sum():,} of {len(valid):,} points (filtered)."
            )

    def _plot_white(self, ax, g, s, pt_size):
        n = len(g)
        if n > _MAX_SCATTER:
            idx = np.random.default_rng(0).choice(
                n, _MAX_SCATTER, replace=False
            )
            g, s = g[idx], s[idx]
        ax.scatter(
            g, s, c="white", s=pt_size, alpha=0.7, linewidths=0, zorder=1
        )

    def _plot_categorical(self, ax, g, s, color_data, cmap_name, pt_size):
        import matplotlib.pyplot as plt

        str_data = [str(v) for v in color_data]
        unique = sorted(set(str_data))
        if not unique:
            self._plot_white(ax, g, s, pt_size)
            return

        cmap = plt.get_cmap(cmap_name)
        n = len(unique)
        if hasattr(cmap, "N") and cmap.N <= 20:
            color_map = {v: cmap(i % cmap.N) for i, v in enumerate(unique)}
        else:
            color_map = {
                v: cmap(i / max(n - 1, 1)) for i, v in enumerate(unique)
            }

        rgba_arr = np.array([color_map[v] for v in str_data])
        self._group_colors = (color_map, np.asarray(str_data, dtype=object))

        gp, sp, cp = g, s, rgba_arr
        if len(gp) > _MAX_SCATTER:
            idx = np.random.default_rng(0).choice(
                len(gp), _MAX_SCATTER, replace=False
            )
            gp, sp, cp = gp[idx], sp[idx], cp[idx]

        ax.scatter(
            gp,
            sp,
            c=cp,
            s=pt_size,
            alpha=0.7,
            linewidths=0,
            zorder=1,
        )

        if self._legend_check.isChecked():
            self._fill_legend_widget(color_map, unique)

    def _plot_continuous(self, ax, g, s, color_data, cmap_name, pt_size):
        vals = pd.to_numeric(pd.Series(color_data), errors="coerce").values
        finite = np.isfinite(vals)
        if not finite.any():
            self._plot_white(ax, g, s, pt_size)
            return

        gf, sf, vf = g[finite], s[finite], vals[finite]
        if len(gf) > _MAX_SCATTER:
            idx = np.random.default_rng(0).choice(
                len(gf), _MAX_SCATTER, replace=False
            )
            gf, sf, vf = gf[idx], sf[idx], vf[idx]

        sc = ax.scatter(
            gf,
            sf,
            c=vf,
            cmap=cmap_name,
            s=pt_size,
            alpha=0.7,
            linewidths=0,
            zorder=1,
        )
        cb = self._fig.colorbar(
            sc,
            ax=ax,
            orientation="vertical",
            fraction=0.046,
            pad=0.10,
            shrink=0.85,
        )
        cb.ax.yaxis.set_tick_params(colors=MPL.TICK, labelsize=7)
        cb.ax.set_facecolor(MPL.AXES_BG)
        cb.outline.set_edgecolor(MPL.SPINE)
        cb.set_label(
            self._color_combo.currentText(), color=MPL.TICK, fontsize=8
        )

    def _draw_group_overlays(self, ax, g, s):
        """Contours / centroids / ellipses per colour group (*g*, *s* finite)."""
        contours = self._ov_contour.isChecked()
        mean = self._ov_mean.isChecked()
        median = self._ov_median.isChecked()
        ellipse = self._ov_ellipse.isChecked()
        if not (contours or mean or median or ellipse) or g.size == 0:
            return
        from matplotlib.patches import Ellipse

        if self._group_colors is not None:
            color_map, labels = self._group_colors
            groups = [(c, labels == v) for v, c in color_map.items()]
        else:
            groups = [("white", np.ones(g.size, dtype=bool))]
        if len(groups) > _MAX_OVERLAY_GROUPS:
            self._status.info(
                f"Overlays skipped: {len(groups)} groups (limit "
                f"{_MAX_OVERLAY_GROUPS})."
            )
            return

        n_sigma = self._ov_sigma.value()
        for color, member in groups:
            gg, ss = g[member], s[member]
            if gg.size < 3:
                continue
            if contours:
                grid = kde_grid(gg, ss)
                if grid is not None:
                    X, Y, Z = grid
                    levels = contour_levels(Z)
                    if levels:
                        ax.contour(
                            X,
                            Y,
                            Z,
                            levels=levels,
                            colors=[color],
                            linewidths=0.9,
                            zorder=10,
                        )
            if ellipse:
                e = covariance_ellipse(gg, ss, n_sigma)
                if e is not None:
                    cx, cy, w, h, ang = e
                    ax.add_patch(
                        Ellipse(
                            (cx, cy),
                            w,
                            h,
                            angle=ang,
                            fill=False,
                            edgecolor=color,
                            linewidth=1.4,
                            zorder=11,
                        )
                    )
            for on, fn, marker in (
                (mean, np.mean, "X"),
                (median, np.median, "P"),
            ):
                if on:
                    ax.scatter(
                        [fn(gg)],
                        [fn(ss)],
                        marker=marker,
                        s=90,
                        c=[color],
                        edgecolors="black",
                        linewidths=0.8,
                        zorder=12,
                    )

        if mean or median:
            key = "   ".join(
                t for on, t in ((mean, "✕ mean"), (median, "✚ median")) if on
            )
            ax.text(
                0.98,
                0.98,
                key,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color="white",
                alpha=0.8,
                zorder=13,
            )

    def _draw_components(self, ax, f_rep_hz):
        taus = [sp.value() for sp in self._tau_spins if sp.value() > 0]
        if not taus:
            return
        points = []
        for i, tau_ns in enumerate(taus):
            g_c, s_c = monoexp_phasor(tau_ns * 1e-9, f_rep_hz)
            points.append((g_c, s_c))
            ax.plot(
                g_c,
                s_c,
                "o",
                color="yellow",
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.8,
                zorder=15,
            )
            ax.annotate(
                f"τ{i + 1}={tau_ns}ns",
                (g_c, s_c),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                color="yellow",
                zorder=15,
            )
        if len(points) == 2:
            ax.plot(
                [points[0][0], points[1][0]],
                [points[0][1], points[1][1]],
                "--",
                color="yellow",
                alpha=0.5,
                linewidth=0.8,
                zorder=4,
            )
        elif len(points) == 3:
            for a, b in [(0, 1), (1, 2), (2, 0)]:
                ax.plot(
                    [points[a][0], points[b][0]],
                    [points[a][1], points[b][1]],
                    "--",
                    color="yellow",
                    alpha=0.5,
                    linewidth=0.8,
                    zorder=4,
                )

    def _finalise_axes(self, ax):
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 0.80)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("G", color=MPL.TICK)
        ax.set_ylabel("S", color=MPL.TICK)
        ax.tick_params(colors=MPL.TICK, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(MPL.SPINE)

    def _draw_empty(self):
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        self._ax = ax
        ax.set_facecolor(MPL.AXES_BG)
        self._fig.patch.set_facecolor(MPL.FIG_BG)
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

    def _fill_legend_widget(self, color_map: dict, unique: list):
        lay = self._legend_layout
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        for val in unique:
            rgba = color_map[val]
            r, g, b = (
                int(rgba[0] * 255),
                int(rgba[1] * 255),
                int(rgba[2] * 255),
            )

            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)

            swatch = QLabel()
            swatch.setFixedSize(12, 12)
            swatch.setStyleSheet(
                f"background: rgb({r},{g},{b}); border: 1px solid #555;"
            )
            row.addWidget(swatch)

            text = QLabel(str(val))
            text.setStyleSheet(f"color: {MPL.TICK}; font-size: 9px;")
            row.addWidget(text, stretch=1)

            row_widget = QWidget()
            row_widget.setLayout(row)
            lay.addWidget(row_widget)

        lay.addStretch()
        self._legend_scroll.setVisible(True)

    # ------------------------------------------------------------------ #
    # Calibration                                                          #
    # ------------------------------------------------------------------ #

    def _on_apply_cal_toggled(self, checked):
        if checked:
            self._apply_calibration()
            return
        if self._update_lifetimes():
            self._refresh_column_combos()
            self._fill_table()
        if self._plotted_once and self._df is not None:
            self._on_plot()

    def _on_factor_changed(self):
        """Keep calibrated columns (and τφ/τm) in step with a new factor."""
        if not self._apply_cal_check.isChecked():
            return
        if self._calib_factor == (1.0 + 0j):
            # Unchecking runs _on_apply_cal_toggled, which recomputes on raw.
            self._apply_cal_check.setChecked(False)
        else:
            self._apply_calibration()

    def _on_frep_changed(self, *_):
        if self._update_lifetimes():
            self._fill_table()
        self._on_setting_changed()

    def _on_compute_lifetimes(self):
        if self._df is None:
            self._status.info("Load a CSV first.")
            return
        self._lifetimes_on = True
        self._update_lifetimes()
        self._refresh_column_combos()
        self._fill_table()
        if self._plotted_once:
            self._on_plot()

    def _update_lifetimes(self) -> bool:
        """Rewrite tau_phi_ns / tau_mod_ns if they are on; True if written."""
        if not self._lifetimes_on or self._df is None:
            return False
        g, s, calibrated = self._current_gs()
        f_mhz = self._frep_spin.value()
        tau_phi, tau_mod = phasor_lifetimes(g, s, f_mhz * 1e6)
        self._df["tau_phi_ns"] = tau_phi
        self._df["tau_mod_ns"] = tau_mod
        src = "calibrated" if calibrated else "raw"
        self._lt_btn.setToolTip(
            f"tau_phi_ns / tau_mod_ns from {src} G,S at {f_mhz:g} MHz.\n"
            "Recomputed on calibration and f_rep changes."
        )
        self._status.info(
            f"Phase / modulation lifetimes from {src} G,S at {f_mhz:g} MHz "
            f"— {np.isfinite(tau_phi).sum():,}/{len(tau_phi):,} defined."
        )
        return True

    def _apply_calibration(self):
        if self._df is None:
            return
        if self._calib_factor == (1.0 + 0j):
            self._status.info("Set a calibration factor first.")
            self._apply_cal_check.blockSignals(True)
            self._apply_cal_check.setChecked(False)
            self._apply_cal_check.blockSignals(False)
            return
        df = self._df
        g = df["g"].values.astype(np.float64)
        s = df["s"].values.astype(np.float64)
        phasor_c = (g + 1j * s) * self._calib_factor
        df["calibrated_g"] = phasor_c.real
        df["calibrated_s"] = phasor_c.imag
        self._status.info(
            "Calibration applied → calibrated_g, calibrated_s columns added."
        )
        self._update_lifetimes()
        self._refresh_column_combos()
        self._fill_table()
        if self._plotted_once:
            self._on_plot()

    def _on_calculate_cal(self):
        from napari_flopa.ui.panels.phasor_panel import CalibrationDialog

        dlg = CalibrationDialog(default_tau=3.834, parent=self)
        if dlg.exec_() != QDialog.Accepted:
            return
        vals = dlg.get_values()
        try:
            measured = complex(vals["g"], vals["s"])
            self._calib_factor = calibration_factor(
                vals["tau_ns"], measured, self._frep_spin.value() * 1e6
            )
            self._calib_display.setText(self._fmt_complex(self._calib_factor))
            self._status.info("Calibration factor updated.")
        except Exception as e:
            self._status.error(f"Calibration error: {e}")
            return
        self._on_factor_changed()

    def _on_custom_factor(self):
        from napari_flopa.ui.panels.phasor_panel import CustomFactorDialog

        dlg = CustomFactorDialog(self._calib_factor, parent=self)
        if dlg.exec_() != QDialog.Accepted:
            return
        r, i = dlg.get_values()
        self._calib_factor = complex(r, i)
        self._calib_display.setText(self._fmt_complex(self._calib_factor))
        self._on_factor_changed()

    def _reset_calib(self):
        self._calib_factor = 1.0 + 0j
        self._calib_display.setText(self._fmt_complex(self._calib_factor))
        self._on_factor_changed()

    @staticmethod
    def _fmt_complex(c: complex) -> str:
        sign = "+" if c.imag >= 0 else "-"
        return f"{c.real:.4f} {sign} {abs(c.imag):.4f}j"

    # ------------------------------------------------------------------ #
    # Component fractions                                                  #
    # ------------------------------------------------------------------ #

    def _on_compute_fractions(self):
        if self._df is None:
            self._status.info("Load a CSV first.")
            return
        taus = [sp.value() for sp in self._tau_spins if sp.value() > 0]
        if len(taus) < 2:
            self._status.error("Set at least two non-zero lifetimes.")
            return

        df = self._df
        g, s, _ = self._current_gs()

        f_rep_hz = self._frep_spin.value() * 1e6

        if len(taus) == 2:
            f1, f2, residual = component_fractions_2(
                g, s, taus[0], taus[1], f_rep_hz
            )
            df["f1"] = f1
            df["f2"] = f2
            df["residual"] = residual
            if "f3" in df.columns:
                df.drop(columns=["f3"], inplace=True)
            valid = np.isfinite(f1).sum()
            self._status.info(
                f"2-component fractions — {valid}/{len(f1)} valid "
                f"(τ₁={taus[0]}, τ₂={taus[1]} ns)."
            )
        else:
            f1, f2, f3 = component_fractions_3(
                g, s, taus[0], taus[1], taus[2], f_rep_hz
            )
            df["f1"] = f1
            df["f2"] = f2
            df["f3"] = f3
            if "residual" in df.columns:
                df.drop(columns=["residual"], inplace=True)
            valid = np.isfinite(f1).sum()
            self._status.info(
                f"3-component fractions — {valid}/{len(f1)} inside triangle "
                f"(τ₁={taus[0]}, τ₂={taus[1]}, τ₃={taus[2]} ns)."
            )

        self._refresh_column_combos()
        self._fill_table()
        if self._plotted_once:
            self._on_plot()

    def _rebuild_cluster_params(self, *_):
        while self._cl_params_lay.count():
            item = self._cl_params_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._cl_param_widgets = {}
        spec = METHODS[self._cl_method.currentText()]["params"]
        for name, (kind, default, *rest) in spec.items():
            self._cl_params_lay.addWidget(QLabel(name.replace("_", " ") + ":"))
            if kind == "choice":
                w = QComboBox()
                w.addItems(rest[0])
                w.setCurrentText(default)
            elif kind == "float":
                w = QDoubleSpinBox()
                w.setDecimals(3)
                w.setSingleStep(0.005)
                w.setRange(rest[0], rest[1])
                w.setValue(default)
            else:
                w = QSpinBox()
                w.setRange(rest[0], rest[1])
                w.setValue(default)
            self._cl_params_lay.addWidget(w)
            self._cl_param_widgets[name] = w
        self._cl_params_lay.addStretch()

    def _cluster_params(self) -> dict:
        out = {}
        for name, w in self._cl_param_widgets.items():
            out[name] = (
                w.currentText() if isinstance(w, QComboBox) else w.value()
            )
        return out

    def _on_run_clustering(self):
        if self._df is None:
            self._status.info("Load a CSV first.")
            return
        method = self._cl_method.currentText()
        g, s, calibrated = self._current_gs()
        fmask = self._filter_mask()
        if fmask is not None:
            # Rows outside the filter are left out of the fit and the labels.
            g = np.where(fmask, g, np.nan)
            s = np.where(fmask, s, np.nan)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            labels = cluster_points(g, s, method, **self._cluster_params())
        except Exception as e:
            self._status.error(f"Clustering failed: {e}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        finite = np.isfinite(g) & np.isfinite(s)
        n_clusters = int(labels.max()) + 1 if labels.size else 0
        width = max(2, len(str(n_clusters)))
        # Zero-padded strings sort correctly and are always treated as
        # categorical by the colouring, whatever the cluster count.
        values = np.where(
            ~finite,
            "",
            np.where(
                labels < 0,
                "noise",
                np.char.add(
                    "C", np.char.zfill((labels + 1).astype(str), width)
                ),
            ),
        ).astype(object)
        col = f"cluster_{METHOD_TAGS[method]}"
        self._df[col] = values
        n_noise = int(((labels < 0) & finite).sum())

        self._refresh_column_combos()
        self._fill_table()
        src = "calibrated" if calibrated else "raw"
        if fmask is not None:
            src += f", {int(fmask.sum()):,} filtered rows"
        msg = f"{method}: {n_clusters} clusters on {src} G,S → {col}"
        if n_noise:
            msg += f" ({n_noise:,} noise points)"
        if self._color_combo.currentText() == col:
            self._on_plot()
        else:
            self._color_combo.setCurrentText(col)
            if not self._plotted_once:
                self._on_plot()
        self._status.info(msg + ".")

    # ------------------------------------------------------------------ #
    # Data tools dialog                                                    #
    # ------------------------------------------------------------------ #

    def _on_open_data_tools(self):
        if self._df is None:
            self._status.info("Load a CSV first.")
            return
        dlg = _DataToolsDialog(self, parent=self)
        dlg.exec_()

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
            "phasor_analyzer.png",
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
        if self._df is None:
            self._status.info("No data to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Table", "phasor_analyzer.csv", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            out = self._filtered_df()
            out.to_csv(path, index=False)
            note = (
                f", filtered from {len(self._df):,}"
                if len(out) != len(self._df)
                else ""
            )
            self._status.info(
                f"Table saved: {Path(path).name} ({len(out):,} rows{note})"
            )
        except Exception as e:
            self._status.error(f"Save error: {e}")


# ── Data Tools dialog ────────────────────────────────────────────────────


class _DataToolsDialog(QDialog):
    """Modal dialog for group-from-name extraction and column combining."""

    def __init__(self, panel: PhasorAnalyzerPanel, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Tools")
        self._panel = panel
        self.setMinimumWidth(420)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        # ── Filter rows ────────────────────────────────────────────────
        flt_box = QGroupBox("Filter rows")
        apply_style(flt_box, S.GROUP_PRIMARY)
        flt_box.setObjectName("plain")
        flt_lay = QVBoxLayout(flt_box)
        flt_lay.setSpacing(4)
        flt_lay.setContentsMargins(6, 4, 6, 4)

        b_row = QHBoxLayout()
        self._flt_col = QComboBox()
        self._flt_col.setMinimumWidth(80)
        self._flt_col.currentTextChanged.connect(self._on_filter_col_changed)
        b_row.addWidget(self._flt_col, 1)
        self._flt_op = QComboBox()
        self._flt_op.addItems(OPS)
        self._flt_op.currentTextChanged.connect(self._on_filter_op_changed)
        b_row.addWidget(self._flt_op)
        self._flt_value = QLineEdit()
        self._flt_value.setMinimumWidth(70)
        self._flt_value.returnPressed.connect(self._on_filter_add)
        b_row.addWidget(self._flt_value, 1)
        add_btn = QPushButton("Add ▸")
        add_btn.setToolTip(
            "Append this condition to the expression (with 'and')"
        )
        add_btn.clicked.connect(self._on_filter_add)
        b_row.addWidget(add_btn)
        flt_lay.addLayout(b_row)

        e_row = QHBoxLayout()
        self._flt_expr = QLineEdit(self._panel._filter_expr)
        self._flt_expr.setPlaceholderText(
            "expression, e.g. tau_phi_ns > 2 and grp == 'A'"
        )
        self._flt_expr.setToolTip(
            "pandas query expression — edit freely.\n"
            "Combine with and / or / not and brackets. Examples:\n"
            "  tau_phi_ns > 2 and tau_phi_ns < 4\n"
            "  grp == 'A' or grp == 'B'\n"
            "  photon_count_sum >= 1000\n"
            "  dataset_name.str.contains('ctrl')\n"
            "Column names with spaces go in `backticks`."
        )
        e_row.addWidget(self._flt_expr, 1)
        flt_apply = QPushButton("Apply")
        flt_apply.clicked.connect(self._on_filter_apply)
        e_row.addWidget(flt_apply)
        flt_clear = QPushButton("Clear")
        flt_clear.clicked.connect(self._on_filter_clear)
        e_row.addWidget(flt_clear)
        flt_lay.addLayout(e_row)

        self._flt_preview = QLabel("")
        self._flt_preview.setStyleSheet(S.HINT)
        self._flt_preview.setWordWrap(True)
        flt_lay.addWidget(self._flt_preview)
        root.addWidget(flt_box)

        self._flt_timer = QTimer(self)
        self._flt_timer.setSingleShot(True)
        self._flt_timer.setInterval(250)
        self._flt_timer.timeout.connect(self._update_filter_preview)
        self._flt_expr.textChanged.connect(lambda *_: self._flt_timer.start())

        # ── Group from name ────────────────────────────────────────────
        grp_box = QGroupBox("Group from name")
        apply_style(grp_box, S.GROUP_PRIMARY)
        grp_box.setObjectName("plain")
        grp_lay = QVBoxLayout(grp_box)
        grp_lay.setSpacing(4)
        grp_lay.setContentsMargins(6, 4, 6, 4)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Source:"))
        self._grp_source_combo = QComboBox()
        self._grp_source_combo.setMinimumWidth(80)
        row1.addWidget(self._grp_source_combo, 1)
        row1.addWidget(QLabel("Pattern:"))
        self._grp_pattern_edit = QLineEdit()
        self._grp_pattern_edit.setPlaceholderText("regex, e.g. F[0-9] or F.*")
        self._grp_pattern_edit.setToolTip(
            "Regular expression (not glob). Use .* for 'anything', "
            "not bare *.\nExamples: F[0-9], ch\\d+, .*_accu"
        )
        row1.addWidget(self._grp_pattern_edit, 1)
        grp_lay.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("New col:"))
        self._grp_name_edit = QLineEdit()
        self._grp_name_edit.setPlaceholderText("column name")
        row2.addWidget(self._grp_name_edit, 1)
        preview_btn = QPushButton("Preview")
        preview_btn.clicked.connect(self._on_group_preview)
        row2.addWidget(preview_btn)
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._on_group_apply)
        row2.addWidget(apply_btn)
        grp_lay.addLayout(row2)

        self._grp_preview_label = QLabel("")
        self._grp_preview_label.setStyleSheet(S.HINT)
        self._grp_preview_label.setWordWrap(True)
        grp_lay.addWidget(self._grp_preview_label)
        root.addWidget(grp_box)

        # ── Combine columns ────────────────────────────────────────────
        comb_box = QGroupBox("Combine columns")
        apply_style(comb_box, S.GROUP_PRIMARY)
        comb_box.setObjectName("plain")
        comb_lay = QVBoxLayout(comb_box)
        comb_lay.setContentsMargins(6, 4, 6, 4)

        comb_row = QHBoxLayout()
        self._comb_col1 = QComboBox()
        self._comb_col1.setMinimumWidth(60)
        comb_row.addWidget(self._comb_col1, 1)
        comb_row.addWidget(QLabel("+"))
        self._comb_col2 = QComboBox()
        self._comb_col2.setMinimumWidth(60)
        comb_row.addWidget(self._comb_col2, 1)
        comb_row.addWidget(QLabel("→"))
        self._comb_name_edit = QLineEdit()
        self._comb_name_edit.setPlaceholderText("new column")
        comb_row.addWidget(self._comb_name_edit, 1)
        comb_apply = QPushButton("Apply")
        comb_apply.clicked.connect(self._on_combine_apply)
        comb_row.addWidget(comb_apply)
        comb_lay.addLayout(comb_row)
        root.addWidget(comb_box)

        # ── Close ──────────────────────────────────────────────────────
        close_row = QHBoxLayout()
        close_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_row.addWidget(close_btn)
        root.addLayout(close_row)

        self._populate_combos()

    def _populate_combos(self):
        df = self._panel._df
        if df is None:
            return
        cols = list(df.columns)
        str_cols = [c for c in cols if df[c].dtype == object]
        cur = self._flt_col.currentText()
        self._flt_col.blockSignals(True)
        self._flt_col.clear()
        self._flt_col.addItems(cols)
        self._flt_col.setCurrentIndex(max(0, self._flt_col.findText(cur)))
        self._flt_col.blockSignals(False)
        self._on_filter_col_changed()
        self._update_filter_preview()
        self._grp_source_combo.clear()
        self._grp_source_combo.addItems(str_cols if str_cols else cols)
        self._comb_col1.clear()
        self._comb_col2.clear()
        self._comb_col1.addItems(cols)
        self._comb_col2.addItems(cols)

    def _filter_col_numeric(self) -> bool:
        df = self._panel._df
        col = self._flt_col.currentText()
        return (
            df is not None
            and col in df.columns
            and pd.api.types.is_numeric_dtype(df[col])
        )

    def _on_filter_col_changed(self, *_):
        """Hint the value field from the column: range or known values."""
        df = self._panel._df
        col = self._flt_col.currentText()
        self._flt_value.setCompleter(None)
        if df is None or col not in df.columns:
            return
        series = df[col]
        if self._filter_col_numeric():
            vals = pd.to_numeric(series, errors="coerce")
            if vals.notna().any():
                self._flt_value.setPlaceholderText(
                    f"{vals.min():.4g} … {vals.max():.4g}"
                )
            if self._flt_op.currentText() in ("contains", "not contains"):
                self._flt_op.setCurrentText(">")
        else:
            uniq = [str(v) for v in series.dropna().unique()[:200]]
            self._flt_value.setPlaceholderText(
                ", ".join(uniq[:3]) + (" …" if len(uniq) > 3 else "")
            )
            completer = QCompleter(uniq, self._flt_value)
            completer.setCaseSensitivity(Qt.CaseInsensitive)
            completer.setFilterMode(Qt.MatchContains)
            self._flt_value.setCompleter(completer)
            if self._flt_op.currentText() in (">", ">=", "<", "<="):
                self._flt_op.setCurrentText("==")

    def _on_filter_op_changed(self, op: str):
        self._flt_value.setEnabled(op not in UNARY_OPS)

    def _on_filter_add(self):
        col = self._flt_col.currentText()
        if not col:
            return
        op = self._flt_op.currentText()
        value = self._flt_value.text().strip()
        if op not in UNARY_OPS and not value:
            self._flt_preview.setText("Enter a value.")
            return
        try:
            clause = build_clause(col, op, value, self._filter_col_numeric())
        except ValueError as e:
            self._flt_preview.setText(str(e))
            return
        expr = self._flt_expr.text().strip()
        self._flt_expr.setText(f"{expr} and {clause}" if expr else clause)
        self._flt_value.clear()

    def _update_filter_preview(self):
        df = self._panel._df
        expr = self._flt_expr.text().strip()
        if df is None or not expr:
            self._flt_preview.setStyleSheet(S.HINT)
            n = 0 if df is None else len(df)
            self._flt_preview.setText(f"No filter — all {n:,} rows.")
            return
        try:
            mask = evaluate_filter(df, expr)
        except ValueError as e:
            self._flt_preview.setStyleSheet("color: #e06060;")
            self._flt_preview.setText(f"Error: {e}")
            return
        self._flt_preview.setStyleSheet(S.HINT)
        self._flt_preview.setText(
            f"Matches {int(mask.sum()):,} / {len(mask):,} rows."
        )

    def _on_filter_apply(self):
        if self._panel.set_filter(self._flt_expr.text()):
            self._update_filter_preview()

    def _on_filter_clear(self):
        self._flt_expr.clear()
        self._panel.set_filter("")

    def _extract_groups(self) -> pd.Series | None:
        df = self._panel._df
        if df is None:
            return None
        source_col = self._grp_source_combo.currentText()
        if source_col not in df.columns:
            return None
        pattern = self._grp_pattern_edit.text().strip()
        if not pattern:
            self._grp_preview_label.setText("Enter a regex pattern.")
            return None
        try:
            regex = re.compile(pattern)
        except re.error as e:
            self._grp_preview_label.setText(f"Invalid regex: {e}")
            return None

        def _extract(val):
            m = regex.search(str(val))
            return m.group(0) if m else ""

        return df[source_col].map(_extract)

    def _on_group_preview(self):
        groups = self._extract_groups()
        if groups is None:
            return
        source_col = self._grp_source_combo.currentText()
        df = self._panel._df
        source_vals = df[source_col].head(8)
        preview = groups.head(8)
        lines = []
        for src, grp in zip(source_vals, preview, strict=True):
            short_src = str(src)[:40]
            lines.append(f"{short_src} → {grp!r}")
        matched = (groups != "").sum()
        lines.append(f"({matched}/{len(groups)} matched)")
        self._grp_preview_label.setText("\n".join(lines))

    def _on_group_apply(self):
        groups = self._extract_groups()
        if groups is None:
            return
        col_name = self._grp_name_edit.text().strip()
        if not col_name:
            self._grp_preview_label.setText("Enter a column name.")
            return
        self._panel._df[col_name] = groups
        self._panel._refresh_column_combos()
        self._panel._fill_table()
        self._populate_combos()
        matched = (groups != "").sum()
        self._panel._status.info(
            f"Added column '{col_name}' — "
            f"{matched}/{len(groups)} rows matched."
        )
        self._grp_preview_label.setText("")

    def _on_combine_apply(self):
        df = self._panel._df
        if df is None:
            return
        col1 = self._comb_col1.currentText()
        col2 = self._comb_col2.currentText()
        name = self._comb_name_edit.text().strip()
        if not name:
            self._panel._status.error("Enter a name for the combined column.")
            return
        if col1 not in df.columns or col2 not in df.columns:
            self._panel._status.error("Select valid columns.")
            return
        df[name] = df[col1].astype(str) + "_" + df[col2].astype(str)
        self._panel._refresh_column_combos()
        self._panel._fill_table()
        self._populate_combos()
        self._panel._status.info(f"Added column '{name}' = {col1} + {col2}.")


# ── Module-level helpers ──────────────────────────────────────────────────


def _draw_lifetime_ticks(ax, sync_hz: float):
    center = np.array([0.5, 0.0])
    tick_len = 0.018
    for tau_ns, g_t, s_t in lifetime_ticks(sync_hz):
        p = np.array([g_t, s_t])
        v = p - center
        nrm = np.linalg.norm(v)
        if nrm < 1e-9:
            continue
        v_u = v / nrm
        p1, p2 = p - tick_len / 2 * v_u, p + tick_len / 2 * v_u
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            "w-",
            lw=0.8,
            alpha=0.6,
            zorder=4,
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
