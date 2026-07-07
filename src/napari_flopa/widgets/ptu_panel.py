import traceback
from pathlib import Path

# ── optional TOML support ────────────────────────────────────────────────
try:
    import tomli_w  # pip install tomli-w
except ImportError:
    tomli_w = None

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
from qtpy.QtCore import Qt, QThreadPool, Signal, Slot
from qtpy.QtGui import QFont
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from napari_flopa.demo import load_demo
from napari_flopa.io.loader import (
    _format_marker_suggestions,
    analyze_marker_distribution,
    format_ptu_header,
    get_markers,
    read_ptu_file,
)
from napari_flopa.io.ptuio.reconstructor import ScanConfig
from napari_flopa.io.ptuio.utils import estimate_bidirectional_shift

# estimate_bidirectional_shift internally uses tttrkit's ImageReconstructor,
# which isinstance-checks against tttrkit's ScanConfig — import it for that path only.
try:
    from ptuio.reconstructor import ScanConfig as _TtkScanConfig
except ImportError:
    _TtkScanConfig = ScanConfig
from napari_flopa.processing.logger import ProgressLogger
from napari_flopa.processing.reconstruction import reconstruct_ptu_to_dataset
from napari_flopa.state import AppState
from napari_flopa.widgets.style import SS, apply_style
from napari_flopa.widgets.utils.threading import Worker


class PtuPanel(QWidget):
    """
    Tab 1: Load PTU file, inspect header, configure scan parameters,
    run reconstruction.
    """

    reconstruction_finished = Signal(object)  # xr.Dataset

    def __init__(self, state: AppState, viewer, parent=None):
        super().__init__(parent)
        self.state = state
        self.viewer = viewer
        self.threadpool = QThreadPool()

        self.ptu_data = None
        self.ptu_filepath = None
        self.shift_plot_data = None
        self._log_buffer: list[str] = []

        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(2, 2, 2, 2)

        # --- File loading ---
        file_group = QGroupBox("Load Data")
        apply_style(file_group, SS.GROUP_A)
        file_layout = QVBoxLayout(file_group)
        self.file_label = QLabel("No file selected.")
        btn_row = QHBoxLayout()
        self.select_ptu_btn = QPushButton("Read PTU file...")
        self.select_ptu_btn.clicked.connect(self._select_ptu_file)
        self.load_demo_btn = QPushButton("Load Demo file...")
        self.load_demo_btn.setToolTip(
            "Load the bundled demo PTU file with preset scan parameters"
        )
        self.load_demo_btn.clicked.connect(self._on_load_demo)
        btn_row.addWidget(self.select_ptu_btn)
        btn_row.addWidget(self.load_demo_btn)
        file_layout.addWidget(self.file_label)
        file_layout.addLayout(btn_row)
        main_layout.addWidget(file_group)

        # --- Header info (summary only, full header via button) ---
        self.header_group = QGroupBox("Header Info")
        apply_style(self.header_group, SS.GROUP_A)
        header_layout = QVBoxLayout(self.header_group)
        self.header_info = QTextEdit()
        self.header_info.setReadOnly(True)
        self.header_info.setFont(QFont("Courier", 6))
        self.header_info.setFixedHeight(100)
        header_layout.addWidget(self.header_info)

        marker_row = QHBoxLayout()
        self.markers_btn = QPushButton("Analyze Markers")
        self.markers_btn.setToolTip(
            "Read marker events to suggest scan dimensions"
        )
        self.markers_btn.clicked.connect(self._show_markers)
        self.full_header_btn = QPushButton("Full Header...")
        self.full_header_btn.setToolTip("Show complete raw header tags")
        self.full_header_btn.clicked.connect(self._show_full_header)
        marker_row.addWidget(self.markers_btn)
        marker_row.addWidget(self.full_header_btn)
        header_layout.addLayout(marker_row)
        main_layout.addWidget(self.header_group)
        self.header_group.setVisible(False)

        # --- Scan configuration ---
        self.config_group = self._build_config_group()
        main_layout.addWidget(self.config_group)
        self.config_group.setVisible(False)

        # --- Reconstruction ---
        self.recon_group = QGroupBox("Reconstruction")
        apply_style(self.recon_group, SS.GROUP_A)
        recon_layout = QVBoxLayout(self.recon_group)

        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output:"))
        self.out_combo = QComboBox()
        self.out_combo.addItems(
            [
                "Intensity only",
                "Intensity + Mean Lifetime",
                "All (+ Phasor & Decay)",
            ]
        )
        self.out_combo.setCurrentIndex(2)
        output_row.addWidget(self.out_combo)
        output_row.addStretch()
        self.reconstruct_btn = QPushButton("Reconstruct Image")
        self.reconstruct_btn.clicked.connect(self._run_reconstruction)
        output_row.addWidget(self.reconstruct_btn)
        recon_layout.addLayout(output_row)

        main_layout.addWidget(self.recon_group)
        self.recon_group.setVisible(False)

        # --- Info / log ---
        self.info_group = QGroupBox("Log")
        apply_style(self.info_group, SS.GROUP_A)
        info_layout = QVBoxLayout(self.info_group)
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(120)
        apply_style(self.log_text, SS.LOG)
        # Override font size to match header_info (8pt > 9px from SS.LOG)
        self.log_text.setStyleSheet(
            self.log_text.styleSheet() + "\nQPlainTextEdit { font-family: Courier; font-size: 8pt; }"
        )
        info_layout.addWidget(self.log_text)
        main_layout.addWidget(self.info_group)

        main_layout.addStretch()

    def _build_config_group(self) -> QGroupBox:
        group = QGroupBox("Scan Configuration")
        apply_style(group, SS.GROUP_A)
        main_layout = QVBoxLayout(group)

        grid = QGridLayout()
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        # Left column
        left = QGroupBox()
        left.setFlat(True)
        lf = QFormLayout(left)
        def _spin_row(spinbox):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(spinbox)
            row.addStretch()
            return row

        self.lines_spin = QSpinBox()
        self.lines_spin.setRange(1, 10000)
        self.lines_spin.setValue(512)
        self.lines_spin.setMinimumWidth(40)
        lf.addRow("Lines:", _spin_row(self.lines_spin))

        self.pixels_spin = QSpinBox()
        self.pixels_spin.setRange(1, 10000)
        self.pixels_spin.setValue(512)
        self.pixels_spin.setMinimumWidth(40)
        lf.addRow("Pixels:", _spin_row(self.pixels_spin))

        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(1, 1000)
        self.frames_spin.setValue(1)
        self.frames_spin.setMinimumWidth(40)
        lf.addRow("Frames:", _spin_row(self.frames_spin))

        self.tcspc_bins_spin = QSpinBox()
        self.tcspc_bins_spin.setRange(1, 65536)
        self.tcspc_bins_spin.setValue(4096)
        self.tcspc_bins_spin.setMinimumWidth(40)
        lf.addRow("TCSPC Bins:", _spin_row(self.tcspc_bins_spin))

        self.max_detector_spin = QSpinBox()
        self.max_detector_spin.setRange(1, 128)
        self.max_detector_spin.setValue(2)
        self.max_detector_spin.setMinimumWidth(40)
        lf.addRow("Max Det.:", _spin_row(self.max_detector_spin))

        grid.addWidget(left, 0, 0)

        # Right column — sequences + accumulations
        right = QGroupBox()
        right.setFlat(True)
        rf = QFormLayout(right)
        rf.setVerticalSpacing(4)

        self.frep_spin = QDoubleSpinBox()
        self.frep_spin.setRange(0.001, 500.0)
        self.frep_spin.setValue(40.0)
        self.frep_spin.setDecimals(3)
        self.frep_spin.setMinimumWidth(40)
        self.frep_spin.setToolTip(
            "Laser/excitation repetition rate — auto-filled from file header.\n"
            "Used for phasor calibration and lifetime calculation."
        )
        self.frep_spin.valueChanged.connect(lambda v: self.state.set_frep(v))
        rf.addRow("Rep. rate (MHz):", _spin_row(self.frep_spin))

        self.sequences_spin = QSpinBox()
        self.sequences_spin.setRange(1, 16)
        self.sequences_spin.setValue(1)
        self.sequences_spin.setMinimumWidth(40)
        self.sequences_spin.valueChanged.connect(
            self._update_accumulation_widgets
        )
        rf.addRow("N Sequences:", _spin_row(self.sequences_spin))

        self.accu_scroll = QScrollArea()
        self.accu_scroll.setWidgetResizable(True)
        self.accu_scroll.setMinimumHeight(60)
        self.accu_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.accu_container = QWidget()
        self.accu_container_layout = QVBoxLayout(self.accu_container)
        self.accu_container_layout.setContentsMargins(2, 2, 2, 2)
        self.accu_container_layout.setSpacing(2)
        self.accu_scroll.setWidget(self.accu_container)
        self.accu_spinboxes = []
        accu_label = QLabel("Accumulations")
        accu_label.setToolTip("Number of line accumulations per sequence")
        rf.addRow(accu_label, self.accu_scroll)

        grid.addWidget(right, 0, 1)
        main_layout.addLayout(grid)

        # Bidirectional
        self.bidir_group = QGroupBox("Bidirectional Scan")
        self.bidir_group.setObjectName("plain")
        apply_style(self.bidir_group, SS.GROUP_A)
        self.bidir_group.setCheckable(True)
        self.bidir_group.setChecked(False)
        bidir_layout = QHBoxLayout(self.bidir_group)

        bidir_layout.addWidget(QLabel("Phase Shift:"))
        self.bidir_phase_spin = QDoubleSpinBox()
        self.bidir_phase_spin.setRange(-0.2, 0.2)
        self.bidir_phase_spin.setSingleStep(0.0001)
        self.bidir_phase_spin.setDecimals(5)
        self.bidir_phase_spin.setValue(0.0)
        self.bidir_phase_spin.setMinimumWidth(80)
        bidir_layout.addWidget(self.bidir_phase_spin)

        bidir_layout.addSpacing(10)
        self.estimate_btn = QPushButton("Estimate")
        self.estimate_btn.clicked.connect(self._on_estimate_shift)
        bidir_layout.addWidget(self.estimate_btn)

        self.plot_shift_btn = QPushButton("Plot")
        self.plot_shift_btn.setEnabled(False)
        self.plot_shift_btn.setToolTip("Plot shift correlation curve")
        self.plot_shift_btn.clicked.connect(self._on_plot_shift)
        bidir_layout.addWidget(self.plot_shift_btn)
        bidir_layout.addStretch()

        self.save_config_btn = QPushButton("Save Config")
        self.save_config_btn.setToolTip("Save scan parameters to a TOML file")
        self.save_config_btn.setStyleSheet("text-align: center;")
        self.save_config_btn.clicked.connect(self._on_save_config_toml)

        bidir_row = QHBoxLayout()
        bidir_row.setContentsMargins(0, 0, 0, 0)
        bidir_row.addWidget(self.bidir_group)
        bidir_row.addWidget(self.save_config_btn)
        main_layout.addLayout(bidir_row)

        self._update_accumulation_widgets()
        return group

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _log_start(self):
        """Begin accumulating a new log block."""
        self._log_buffer = []

    def _log_line(self, text: str):
        """Add a line to the current log block buffer."""
        self._log_buffer.append(text)

    def _log_commit(self):
        """Prepend the buffered block as one unit to the Info box."""
        if not self._log_buffer:
            return
        block = "\n".join(self._log_buffer)
        current = self.log_text.toPlainText()
        new_content = block if not current else f"{block}\n{'-' * 40}\n{current}"
        self.log_text.setPlainText(new_content)
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.Start)
        self.log_text.setTextCursor(cursor)
        self._log_buffer = []

    # keep a single-call shortcut for one-liner messages
    def _log(self, text: str):
        self._log_start()
        self._log_line(text)
        self._log_commit()

    # ------------------------------------------------------------------ #
    # Slots                                                                #
    # ------------------------------------------------------------------ #

    def _select_ptu_file(self):
        filepath_str, _ = QFileDialog.getOpenFileName(
            self, "Select PTU File", "", "PicoQuant Files (*.ptu)"
        ) # type: ignore
        if not filepath_str:
            return
        self.ptu_filepath = Path(filepath_str)
        self.file_label.setText(f"PTU: {self.ptu_filepath.name}")
        try:
            self.ptu_data = read_ptu_file(str(self.ptu_filepath), header=False)
            tags = self.ptu_data["header"]
            constants = self.ptu_data["constants"]

            # Summary (no full header) — full header via button
            summary = format_ptu_header(tags, constants, full_header=False)
            self.header_info.setPlainText(summary)

            # Auto-fill from header
            px_x = tags.get("ImgHdr_PixX")
            px_y = tags.get("ImgHdr_PixY")
            n_frames = tags.get("ImgHdr_NumberOfFrames")
            if isinstance(px_x, (int, float)):
                self.pixels_spin.setValue(int(px_x))
            if isinstance(px_y, (int, float)):
                self.lines_spin.setValue(int(px_y))
            if isinstance(n_frames, (int, float)) and n_frames > 0:
                self.frames_spin.setValue(int(n_frames))
            self.tcspc_bins_spin.setValue(constants.get("tcspc_bins", 4096))
            rep = constants.get("repetition_rate") or constants.get(
                "sync_rate_hz"
            )
            if rep:
                self.frep_spin.setValue(float(rep) / 1e6)

            self.header_group.setVisible(True)
            self.config_group.setVisible(True)
            self.recon_group.setVisible(True)
            self.log_text.clear()
            self.shift_plot_data = None
            self.plot_shift_btn.setEnabled(False)

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to read PTU header:\n{e}"
            )

    def _on_load_demo(self):
        try:
            params, ptu_path = load_demo()
        except Exception as e:
            QMessageBox.critical(self, "Demo Error", str(e))
            return

        try:
            self.ptu_filepath = ptu_path
            self.file_label.setText(f"PTU: {ptu_path.name}")
            self.ptu_data = read_ptu_file(str(ptu_path), header=False)
            tags = self.ptu_data["header"]
            constants = self.ptu_data["constants"]
            summary = format_ptu_header(tags, constants, full_header=False)
            self.header_info.setPlainText(summary)

            # Apply demo params
            self.lines_spin.setValue(params["lines"])
            self.pixels_spin.setValue(params["pixels"])
            self.frames_spin.setValue(params["frames"])
            self.max_detector_spin.setValue(params["max_detector"])
            self.sequences_spin.setValue(params["sequences"])
            self._update_accumulation_widgets()
            for i, acc in enumerate(params["accumulations"]):
                if i < len(self.accu_spinboxes):
                    self.accu_spinboxes[i].setValue(acc)
            self.bidir_group.setChecked(params.get("bidirectional", False))
            self.bidir_phase_spin.setValue(
                params.get("bidirectional_phase_shift", 0.0)
            )
            self.tcspc_bins_spin.setValue(constants.get("tcspc_bins", 4096))
            rep = constants.get("repetition_rate") or constants.get(
                "sync_rate_hz"
            )
            if rep:
                self.frep_spin.setValue(float(rep) / 1e6)

            self.header_group.setVisible(True)
            self.config_group.setVisible(True)
            self.recon_group.setVisible(True)
            self.log_text.clear()
            self.shift_plot_data = None
            self.plot_shift_btn.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(
                self, "Demo Load Error", f"Failed to load demo PTU:\n{e}"
            )

    def _show_full_header(self):
        if not self.ptu_data:
            return
        tags = self.ptu_data["header"]
        constants = self.ptu_data["constants"]
        full_text = format_ptu_header(tags, constants, full_header=True)
        dlg = QDialog(self)
        dlg.setWindowTitle("Full PTU Header")
        dlg.resize(600, 500)
        layout = QVBoxLayout(dlg)
        te = QTextEdit()
        te.setReadOnly(True)
        te.setFont(QFont("Courier", 8))
        te.setPlainText(full_text)
        layout.addWidget(te)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)
        dlg.exec_()

    def _show_markers(self):
        if not self.ptu_data:
            self._log("Analyze Markers: no file loaded.")
            return
        self._log_start()
        self._log_line("Analyzing markers...")
        QApplication.processEvents()
        try:
            dist = get_markers(self.ptu_data["reader"], chunk_limit=20)
            if "error" in dist:
                self._log_line(dist["error"])
                self._log_commit()
                return
            analysis = analyze_marker_distribution(dist)
            text = _format_marker_suggestions(analysis)
            self._log_line(text)
            self._log_commit()

            # Auto-apply if only one suggestion
            suggestions = analysis.get("suggestions", [])
            if len(suggestions) == 1:
                lines, accum = suggestions[0]
                self.lines_spin.setValue(lines)
                self.sequences_spin.setValue(1)
                self._update_accumulation_widgets()
                if self.accu_spinboxes:
                    self.accu_spinboxes[0].setValue(accum)
        except Exception as e:
            self._log_line(f"Markers error: {e}")
            self._log_commit()

    def _update_accumulation_widgets(self):
        while self.accu_container_layout.count():
            item = self.accu_container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.accu_spinboxes.clear()

        for i in range(self.sequences_spin.value()):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(QLabel(f"S{i+1}:"))
            spin = QSpinBox()
            spin.setRange(1, 1000)
            spin.setValue(1)
            row_layout.addWidget(spin)
            row_layout.addStretch()
            self.accu_container_layout.addWidget(row)
            self.accu_spinboxes.append(spin)

        self.accu_container_layout.addStretch()

    def _on_estimate_shift(self):
        if not self.ptu_data:
            QMessageBox.warning(
                self, "No File", "Please load a PTU file first."
            )
            return
        self.shift_plot_data = None
        self.plot_shift_btn.setEnabled(False)
        self.estimate_btn.setText("Estimating...")
        self._log_start()
        self._log_line("Estimating bidirectional shift...")
        QApplication.processEvents()

        try:
            accumulations = tuple(s.value() for s in self.accu_spinboxes) or (
                1,
            )
            # Use tttrkit's ScanConfig so estimate_bidirectional_shift's isinstance check passes
            config = _TtkScanConfig(
                lines=self.lines_spin.value(),
                pixels=self.pixels_spin.value(),
                bidirectional=True,
                bidirectional_phase_shift=self.bidir_phase_spin.value(),
                line_accumulations=accumulations,
                max_detector=self.max_detector_spin.value(),
                frame_start_marker_channel=(4,),
                line_start_marker_channel=(1,),
                line_stop_marker_channel=(2,),
            )
            best_shift, plot_data = estimate_bidirectional_shift(
                reader=self.ptu_data["reader"],
                config=config,
                wrap=self.ptu_data["constants"]["wrap"],
                verbose=False,
            )
            if best_shift is not None:
                self.bidir_phase_spin.setValue(best_shift)
                self._log_line(f"Best shift: {best_shift:.5f}")
                self.shift_plot_data = plot_data
                self.plot_shift_btn.setEnabled(True)
            else:
                self._log_line("Estimation failed — check parameters.")
        except Exception as e:
            self._log_line(f"Error: {e}\n{traceback.format_exc()}")
        finally:
            self._log_commit()
            self.estimate_btn.setText("Estimate")

    def _on_plot_shift(self):
        if self.shift_plot_data is None:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Bidirectional Shift Estimation")
        dlg.resize(500, 350)
        layout = QVBoxLayout(dlg)
        canvas = FigureCanvas(Figure(figsize=(5, 3.5)))
        layout.addWidget(canvas)
        ax = canvas.figure.subplots()
        shifts, scores, fit = self.shift_plot_data
        ax.plot(shifts, scores, "o-", label="correlation")
        ax.plot(shifts, fit, "--", label="Gaussian fit")
        best = shifts[scores.argmax()]
        ax.axvline(
            self.bidir_phase_spin.value(),
            color="r",
            linestyle=":",
            label=f"best={self.bidir_phase_spin.value():.5f}",
        )
        ax.set_xlabel("Phase shift")
        ax.set_ylabel("Score")
        ax.legend()
        canvas.draw()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)
        dlg.exec_()

    def _get_outputs(self) -> list:
        idx = self.out_combo.currentIndex()
        if idx == 0:
            return ["photon_count"]
        if idx == 1:
            return ["photon_count", "mean_arrival_time"]
        return None  # all

    def _on_save_config_toml(self):
        if tomli_w is None:
            QMessageBox.warning(
                self,
                "Missing dependency",
                "tomli-w not available — pip install tomli-w",
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save scan config", "scan_config.toml", "TOML (*.toml)"
        )
        if not path:
            return
        accumulations = [s.value() for s in self.accu_spinboxes] or [1]
        cfg = {
            "scan": dict(
                frames=self.frames_spin.value(),
                lines=self.lines_spin.value(),
                pixels=self.pixels_spin.value(),
                sequences=self.sequences_spin.value(),
                accum_per_seq=" ".join(str(a) for a in accumulations),
                max_detector=self.max_detector_spin.value(),
                bidirectional=self.bidir_group.isChecked(),
                bidirectional_phase_shift=self.bidir_phase_spin.value(),
            ),
            "calibration": dict(
                f_rep_mhz=float(self.frep_spin.value()),
                factor="1+0j",
            ),
        }
        try:
            with open(path, "wb") as f:
                tomli_w.dump(cfg, f)
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))

    def _build_scan_config(self) -> ScanConfig:
        accumulations = tuple(s.value() for s in self.accu_spinboxes) or (1,)
        return ScanConfig(
            lines=self.lines_spin.value(),
            pixels=self.pixels_spin.value(),
            frames=self.frames_spin.value(),
            line_accumulations=accumulations,
            bidirectional=self.bidir_group.isChecked(),
            bidirectional_phase_shift=self.bidir_phase_spin.value(),
            max_detector=self.max_detector_spin.value(),
            frame_start_marker_channel=(4,),
            line_start_marker_channel=(1,),
            line_stop_marker_channel=(2,),
        )

    def _run_reconstruction(self):
        if not self.ptu_data:
            QMessageBox.warning(
                self, "No Data", "Please load a PTU file first."
            )
            return
        self.log_text.clear()
        self.reconstruct_btn.setEnabled(False)
        self._log("Starting reconstruction...")
        QApplication.processEvents()

        scan_config = self._build_scan_config()
        outputs = self._get_outputs()
        tcspc_override = self.tcspc_bins_spin.value()
        ptu_data = self.ptu_data
        logger = ProgressLogger(mode="collect")

        def _run():
            return reconstruct_ptu_to_dataset(
                ptu_data=ptu_data,
                scan_config=scan_config,
                outputs=outputs,
                tcspc_channels_override=tcspc_override,
                logger=logger,
            )

        worker = Worker(_run)
        worker.signals.result.connect(
            lambda ds: self._on_reconstruction_result(ds, scan_config)
        )
        worker.signals.error.connect(self._on_reconstruction_error)
        worker.signals.finished.connect(
            lambda: self.reconstruct_btn.setEnabled(True)
        )
        self.threadpool.start(worker)

    @Slot(object)
    def _on_reconstruction_result(self, ds, scan_config):
        constants = self.ptu_data["constants"].copy() if self.ptu_data else {}
        constants["tcspc_bins"] = self.tcspc_bins_spin.value()
        ds.attrs["instrument_params"] = constants
        if scan_config is not None:
            ds.attrs["scan_config"] = scan_config.to_dict()
        if self.ptu_filepath:
            ds.attrs["source_filename"] = self.ptu_filepath.name

        self._log(f"Done. Dataset: {dict(ds.sizes)}")
        self.state.set_dataset(ds, constants)
        self.reconstruction_finished.emit(ds)

    @Slot(tuple)
    def _on_reconstruction_error(self, error_tuple):
        exctype, value, tb = error_tuple
        self._log(f"ERROR: {value}\n{tb}")
        QMessageBox.critical(self, "Reconstruction Error", str(value))
