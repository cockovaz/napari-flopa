import xarray as xr
from qtpy.QtCore import QObject, Signal


class FlopaState(QObject):
    """
    Central shared state passed to all widgets.

    Widgets connect to signals to react to changes.
    Widgets call setters to update state.

    Signals:
        dataset_changed — new xr.Dataset loaded (after reconstruction)
        calib_factor_changed — phasor calibration factor set by any panel
        calib_enabled_changed — Phasor tab's Calibration box switched on/off
        file_loaded — a PTU was read in the File tab (before reconstruction)
    """

    dataset_changed = Signal()
    calib_factor_changed = Signal(object)  # complex
    calib_enabled_changed = Signal(bool)
    file_loaded = Signal()  # File tab read a PTU — file_config() now has data

    def __init__(self):
        super().__init__()

        self.dataset: xr.Dataset | None = None
        self.frep_mhz: float = 40.0
        self.calib_factor: complex = 1.0 + 0j
        self.calib_enabled: bool = False
        self.file_config_provider = None
        # Pull, like file_config_provider: returns the File tab's live
        # {"ptu_path": Path | None, "chunk_size": int}. The path and the chunk
        # size are not part of the saved scan config, but other tabs still need
        # to read the same file the same way.
        self.file_runtime_provider = None

    # Setters

    def set_dataset(self, ds: xr.Dataset, constants: dict):
        self.dataset = ds
        # Auto-update frep from the header constants
        hz = constants.get("repetition_rate")
        if hz:
            self.frep_mhz = float(hz) / 1e6
        self.dataset_changed.emit()

    def set_calib_factor(self, factor: complex):
        self.calib_factor = complex(factor)
        self.calib_factor_changed.emit(self.calib_factor)

    def set_calib_enabled(self, on: bool):
        self.calib_enabled = bool(on)
        self.calib_enabled_changed.emit(self.calib_enabled)

    # Accessors                                                            #

    def has_data(self) -> bool:
        return self.dataset is not None

    def file_config(self) -> dict | None:
        """The File tab's current scan config, or None if no PTU is loaded."""
        if self.file_config_provider is None:
            return None
        return self.file_config_provider()

    def file_runtime(self) -> dict:
        """The File tab's current PTU path and chunk size.

        Always a dict; ``ptu_path`` is None until a file has been read.
        """
        if self.file_runtime_provider is None:
            return {"ptu_path": None, "chunk_size": 0}
        return self.file_runtime_provider()

    def notify_file_loaded(self):
        """Called by PtuPanel once a PTU header has been read."""
        self.file_loaded.emit()
