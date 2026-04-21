# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run a single test
pytest tests/test_sample_data.py::test_placeholder

# Lint
ruff check src/

# Format
black src/

# Run via tox (full matrix)
tox
```

There is no standalone "run" command — the plugin is loaded inside napari. To test interactively:
```bash
napari
# Then Plugins → FLIM Analysis
```

## Architecture

The plugin is a single napari dock widget (`FlimWidget`) registered via `napari.yaml` (npe2 manifest). It uses a **shared reactive state** pattern where all panels communicate through a central `AppState` object rather than directly.

This project follows a strict separation between **Core Logic** and **UI/Plugin**.

### Boundaries
- **Core (Library):** `napari_flopa/io/`, `napari_flopa/processing/`, `napari_flopa/models/`.
    - **NO IMPORTS** of `napari`, `qtpy`, `PyQt6`, or `magicgui`.
    - Must be fully functional via Python script or IPython.
- **UI (Plugin):** `napari_flopa/widgets/`, `napari_flopa/napari.yaml`.
    - Thin wrappers. Logic should be limited to: "Get values from UI" -> "Call Core Function" -> "Update Display".
- **External/Stable:** `napari_flopa/io/ptuio/`.
    - Treat as a read-only third-party library. Do not modify. Use wrappers in `io/loader.py` to adapt its output.

### Data flow

```
.ptu file
  → io/ptuio/  (parse header + stream TTTR records)
  → io/loader.py  (assemble instrument constants)
  → processing/reconstruction.py  (ImageReconstructor, chunk-by-chunk)
  → xr.Dataset  (stored in AppState)
  → widgets observe via AppState signals
```

The `xr.Dataset` holds up to 5 variables: `photon_count`, `mean_arrival_time`, `phasor_g`, `phasor_s`, `tcspc_histogram`. All have dimensions `(frame, sequence, line, pixel)` or `(frame, sequence, line, pixel, tcspc_bin)` for the histogram. Detector axis is handled separately.

### Key files

**`state.py` — `AppState(QObject)`**
Singleton-per-session shared state. All panels hold a reference to the same instance.
- `dataset_changed` signal → fired when new data is loaded
- `view_changed` signal → fired when the user changes F/S/D selection
- `frep_mhz` and `calib_factor` — written by PtuPanel/PhasorPanel, read by others
- `set_dataset()` auto-extracts `frep_mhz` from `ds.attrs['constants']`

**`widgets/main_widget.py` — `FlimWidget`**
Tab container (Process PTU, Phasor, Decay, Batch). Creates `AppState` and passes it to all child panels. Adds `FlimViewPanel` as a napari bottom dock on first reconstruction.

**`widgets/ptu_panel.py` — `PtuPanel`**
File loading and reconstruction. Runs reconstruction in a background `Worker` (QRunnable). Writes to `state.dataset` and `state.frep_mhz` on completion.

**`widgets/flim_view_panel.py` — `FlimViewPanel`**
Interactive bottom dock. Manages napari image layers for intensity and lifetime display. Owns the `HistogramSlider` for contrast/mask control. Exports FLIM RGB composites.

**`widgets/phasor_panel.py` — `PhasorPanel`**
Phasor scatter plot with calibration, ROI lasso selection, per-object/per-pixel modes. Stores `_calib_factor` locally and also pushes to `state.calib_factor`. ROI selection uses matplotlib blitting for performance (animated line, background captured once per stroke).

**`widgets/batch_panel.py` — `BatchPanel`**
Batch-processes a folder of `.ptu` files. Scan config and calibration are saved/loaded as TOML (`tomllib`/`tomli` read, `tomli-w` write).

**`widgets/histogram_slider.py` — `HistogramSlider`**
Custom dual-range slider. `update_data(arr)` sets the cyan slider to `[p2, p98]` by default — this is the reference "auto contrast" behavior.

**`io/ptuio/reconstructor.py` — `ScanConfig` + `ImageReconstructor`**
`ScanConfig` is the single source of truth for scan geometry. Pass it as a dataclass; `to_dict()`/`from_dict()` for serialization. `ImageReconstructor` is stateful and processes chunk-by-chunk via `.update(corrected_events)`.

**`utils/threading.py` — `Worker`**
Standard pattern for background tasks: wrap any callable, connect `.signals.result` / `.signals.error` / `.signals.progress`, submit to `QThreadPool.globalInstance()`.

### Conventions

- **Qt bindings:** always import from `qtpy` (not PyQt6/PySide2 directly). For validators with regex: `QRegularExpressionValidator(QRegularExpression(r'...'))` — PyQt6 requires the wrapper object.
- **Stale indicators:** Red `●` = "Settings or view have changed since the last plot."; Green `●` = "Plot matches current settings." — tooltip must change at the same time as color.
- **Dataset attrs:** `ds.attrs['constants']` holds instrument params (repetition_rate in Hz, tcspc_resolution_ns, etc.); `ds.attrs['scan_config']` holds the `ScanConfig` dict; `ds.attrs['source_filename']` holds the original file name.
