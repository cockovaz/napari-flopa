import json
import traceback
from pathlib import Path

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

from napari_flopa.io.loader import (
    _format_marker_suggestions,
    analyze_marker_distribution,
    format_ptu_header,
    get_markers,
    load_h5_dataset,
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

        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI construction                                                      #
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(2, 2, 2, 2)

        # --- File loading ---
        file_group = QGroupBox("Load Data")
        file_layout = QVBoxLayout(file_group)
        self.file_label = QLabel("No file selected.")
        btn_row = QHBoxLayout()
        self.select_ptu_btn = QPushButton("Read PTU File...")
        self.select_ptu_btn.clicked.connect(self._select_ptu_file)
        self.select_h5_btn = QPushButton("Load H5...")
        self.select_h5_btn.setToolTip(
            "Load a previously exported FLOPA HDF5 dataset."
        )
        self.select_h5_btn.clicked.connect(self._on_load_h5)
        self.load_demo_btn = QPushButton("Load Demo")
        self.load_demo_btn.setToolTip(
            "Load the bundled demo PTU file with preset scan parameters"
        )
        self.load_demo_btn.clicked.connect(self._on_load_demo)
        btn_row.addWidget(self.select_ptu_btn)
        btn_row.addWidget(self.select_h5_btn)
        btn_row.addWidget(self.load_demo_btn)
        file_layout.addWidget(self.file_label)
        file_layout.addLayout(btn_row)
        main_layout.addWidget(file_group)

        # --- Header info (summary only, full header via button) ---
        self.header_group = QGroupBox("Header Info")
        header_layout = QVBoxLayout(self.header_group)
        self.header_info = QTextEdit()
        self.header_info.setReadOnly(True)
        self.header_info.setFont(QFont("Courier", 8))
        self.header_info.setFixedHeight(130)
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
        self.markers_output = QPlainTextEdit()
        self.markers_output.setReadOnly(True)
        self.markers_output.setFont(QFont("Courier", 8))
        marker_row.addWidget(self.markers_btn)
        marker_row.addWidget(self.full_header_btn)
        header_layout.addLayout(marker_row)

        self.markers_output.setFixedHeight(60)
        header_layout.addWidget(self.markers_output)
        main_layout.addWidget(self.header_group)
        self.header_group.setVisible(False)

        # --- Scan configuration ---
        self.config_group = self._build_config_group()
        main_layout.addWidget(self.config_group)
        self.config_group.setVisible(False)

        # --- Reconstruction ---
        self.recon_group = QGroupBox("Reconstruction")
        recon_layout = QVBoxLayout(self.recon_group)

        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output:"))
        self.out_combo = QComboBox()
        self.out_combo.addItems(
            [
                "Intensity only",
                "Intensity + Mean Lifetime",
                "All (+ Phasor & TCSPC)",
            ]
        )
        self.out_combo.setCurrentIndex(2)
        output_row.addWidget(self.out_combo)
        output_row.addStretch()
        recon_layout.addLayout(output_row)

        self.reconstruct_btn = QPushButton("Reconstruct Image")
        self.reconstruct_btn.clicked.connect(self._run_reconstruction)
        recon_layout.addWidget(self.reconstruct_btn)

        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 8))
        self.log_text.setFixedHeight(100)
        recon_layout.addWidget(self.log_text)

        main_layout.addWidget(self.recon_group)
        self.recon_group.setVisible(False)

        main_layout.addStretch()

    def _build_config_group(self) -> QGroupBox:
        group = QGroupBox("Scan Configuration")
        main_layout = QVBoxLayout(group)

        grid = QGridLayout()
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        # Left column
        left = QGroupBox()
        left.setFlat(True)
        lf = QFormLayout(left)
        self.lines_spin = QSpinBox()
        self.lines_spin.setRange(1, 10000)
        self.lines_spin.setValue(512)
        lf.addRow("Lines:", self.lines_spin)

        self.pixels_spin = QSpinBox()
        self.pixels_spin.setRange(1, 10000)
        self.pixels_spin.setValue(512)
        lf.addRow("Pixels:", self.pixels_spin)

        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(1, 1000)
        self.frames_spin.setValue(1)
        lf.addRow("Frames:", self.frames_spin)

        self.tcspc_bins_spin = QSpinBox()
        self.tcspc_bins_spin.setRange(1, 65536)
        self.tcspc_bins_spin.setValue(4096)
        lf.addRow("TCSPC Bins:", self.tcspc_bins_spin)

        self.max_detector_spin = QSpinBox()
        self.max_detector_spin.setRange(1, 128)
        self.max_detector_spin.setValue(2)
        lf.addRow("Max Det.:", self.max_detector_spin)

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
        self.frep_spin.setSuffix(" MHz")
        self.frep_spin.setToolTip(
            "Laser/excitation repetition rate — auto-filled from file header.\n"
            "Used for phasor calibration and lifetime calculation."
        )
        self.frep_spin.valueChanged.connect(lambda v: self.state.set_frep(v))
        rf.addRow("f_rep:", self.frep_spin)

        self.sequences_spin = QSpinBox()
        self.sequences_spin.setRange(1, 16)
        self.sequences_spin.setValue(1)
        self.sequences_spin.valueChanged.connect(
            self._update_accumulation_widgets
        )
        rf.addRow("N Sequences:", self.sequences_spin)

        self.accu_scroll = QScrollArea()
        self.accu_scroll.setWidgetResizable(True)
        self.accu_scroll.setFixedHeight(80)
        self.accu_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.accu_container = QWidget()
        self.accu_container_layout = QVBoxLayout(self.accu_container)
        self.accu_container_layout.setContentsMargins(2, 2, 2, 2)
        self.accu_container_layout.setSpacing(2)
        self.accu_scroll.setWidget(self.accu_container)
        self.accu_spinboxes = []
        accu_label = QLabel("Accu:")
        accu_label.setToolTip("Number of line accumulations per sequence")
        rf.addRow(accu_label, self.accu_scroll)

        grid.addWidget(right, 0, 1)
        main_layout.addLayout(grid)

        # Bidirectional
        self.bidir_group = QGroupBox("Bidirectional Scan")
        self.bidir_group.setCheckable(True)
        self.bidir_group.setChecked(False)
        bidir_layout = QHBoxLayout(self.bidir_group)

        bidir_layout.addWidget(QLabel("Phase Shift:"))
        self.bidir_phase_spin = QDoubleSpinBox()
        self.bidir_phase_spin.setRange(-0.2, 0.2)
        self.bidir_phase_spin.setSingleStep(0.0001)
        self.bidir_phase_spin.setDecimals(5)
        self.bidir_phase_spin.setValue(0.0)
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

        main_layout.addWidget(self.bidir_group)

        self._update_accumulation_widgets()
        return group

    # ------------------------------------------------------------------ #
    # Slots                                                                #
    # ------------------------------------------------------------------ #

    def _select_ptu_file(self):
        filepath_str, _ = QFileDialog.getOpenFileName(
            self, "Select PTU File", "", "PicoQuant Files (*.ptu)"
        )
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
            self.markers_output.setPlainText("")
            self.shift_plot_data = None
            self.plot_shift_btn.setEnabled(False)

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to read PTU header:\n{e}"
            )

    def _on_load_h5(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load FLOPA HDF5 Dataset", "", "HDF5 Files (*.h5)"
        )
        if not filepath:
            return
        try:
            ds = load_h5_dataset(Path(filepath))
            ds.attrs["source_filename"] = Path(filepath).name
            self._on_reconstruction_result(ds, scan_config=None)
        except Exception as e:
            QMessageBox.critical(
                self, "H5 Load Error", f"Failed to load HDF5:\n{e}"
            )

    def _on_load_demo(self):
        pkg_data = Path(__file__).parent.parent / "data"
        params_path = pkg_data / "demo_params.json"
        if not params_path.exists():
            QMessageBox.critical(
                self,
                "Demo Error",
                f"demo_params.json not found:\n{params_path}",
            )
            return
        with open(params_path) as f:
            params = json.load(f)

        # Locate PTU — first try alongside the params file, then the repo data/ dir
        ptu_name = params["ptu_filename"]
        candidates = [
            pkg_data / ptu_name,
            Path(__file__).parent.parent.parent.parent.parent
            / "data"
            / ptu_name,
        ]
        ptu_path = next((p for p in candidates if p.exists()), None)
        if ptu_path is None:
            QMessageBox.critical(
                self,
                "Demo Error",
                f"Demo PTU file not found:\n{ptu_name}\n\n"
                "Place it in the repo data/ folder or alongside demo_params.json.",
            )
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
            self.markers_output.setPlainText("")
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
            self.markers_output.setPlainText("No file loaded.")
            return
        self.markers_output.setPlainText("Analyzing...")
        QApplication.processEvents()
        try:
            dist = get_markers(self.ptu_data["reader"], chunk_limit=20)
            if "error" in dist:
                self.markers_output.setPlainText(dist["error"])
                return
            analysis = analyze_marker_distribution(dist)
            text = _format_marker_suggestions(analysis)
            self.markers_output.setPlainText(text)

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
            self.markers_output.setPlainText(f"Error: {e}")

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
        self.log_text.appendPlainText("Estimating bidirectional shift...")
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
                self.log_text.appendPlainText(f"Best shift: {best_shift:.5f}")
                self.shift_plot_data = plot_data
                self.plot_shift_btn.setEnabled(True)
            else:
                self.log_text.appendPlainText(
                    "Estimation failed — check parameters."
                )
        except Exception as e:
            self.log_text.appendPlainText(
                f"Error: {e}\n{traceback.format_exc()}"
            )
        finally:
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
        self.log_text.appendPlainText("Starting reconstruction...")

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

        self.log_text.appendPlainText(f"Done. Dataset: {dict(ds.sizes)}")
        self.state.set_dataset(ds, constants)
        self.reconstruction_finished.emit(ds)

    @Slot(tuple)
    def _on_reconstruction_error(self, error_tuple):
        exctype, value, tb = error_tuple
        self.log_text.appendPlainText(f"ERROR: {value}\n{tb}")
        QMessageBox.critical(self, "Reconstruction Error", str(value))
