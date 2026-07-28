import xarray as xr
from qtpy.QtCore import QObject, Signal


class FlopaState(QObject):
    """
    Central shared state passed to all widgets.

    Widgets connect to signals to react to changes.
    Widgets call setters to update state.

    Signals:
        dataset_changed — new xr.Dataset loaded (after reconstruction)
    """

    dataset_changed = Signal()

    def __init__(self):
        super().__init__()

        # --- Dataset ---
        self.dataset: xr.Dataset | None = None

        # --- Instrument / calibration (shared across panels) ---
        self.frep_mhz: float = 40.0
        self.calib_factor: complex = 1.0 + 0j

    # ------------------------------------------------------------------ #
    # Setters — emit signals so widgets update automatically              #
    # ------------------------------------------------------------------ #

    def set_dataset(self, ds: xr.Dataset, constants: dict):
        self.dataset = ds
        # Auto-update frep from constants if available
        hz = constants.get("repetition_rate") or constants.get("sync_rate_hz")
        if hz:
            self.frep_mhz = float(hz) / 1e6
        self.dataset_changed.emit()

    def set_frep(self, mhz: float):
        self.frep_mhz = float(mhz)

    def set_calib_factor(self, factor: complex):
        self.calib_factor = complex(factor)

    # ------------------------------------------------------------------ #
    # Accessors                                                            #
    # ------------------------------------------------------------------ #

    def has_data(self) -> bool:
        return self.dataset is not None
