import numpy as np
import xarray as xr
from qtpy.QtCore import QObject, Signal


class FlopaState(QObject):
    """
    Central shared state passed to all widgets.

    Widgets connect to signals to react to changes.
    Widgets call setters to update state — never write attributes directly.

    Signals:
        dataset_changed   — new xr.Dataset loaded (after reconstruction)
        view_changed      — frame/sequence/channel selection changed
        smoothing_changed — smoothing kernel size changed
    """

    dataset_changed = Signal()
    view_changed = Signal()
    smoothing_changed = Signal()

    def __init__(self):
        super().__init__()

        # --- Dataset ---
        self.dataset: xr.Dataset | None = None
        self.constants: dict = {}

        # --- View selection ---
        self.frame: int = 0
        self.sequence: int = 0
        self.channel: int = 0

        # --- Smoothing ---
        self.smooth: bool = False
        self.smooth_kernel: int = 3

        # --- Instrument / calibration (shared across panels) ---
        self.frep_mhz: float = 40.0
        self.calib_factor: complex = 1.0 + 0j

        # --- Cache: smoothed arrays (keyed by kernel size) ---
        # Avoids recomputing when toggling smooth on/off
        self._smooth_cache: dict = {}

    # ------------------------------------------------------------------ #
    # Setters — emit signals so widgets update automatically              #
    # ------------------------------------------------------------------ #

    def set_dataset(self, ds: xr.Dataset, constants: dict):
        self.dataset = ds
        self.constants = constants
        self._smooth_cache.clear()
        self.frame = 0
        self.sequence = 0
        self.channel = 0
        # Auto-update frep from constants if available
        hz = constants.get("repetition_rate") or constants.get("sync_rate_hz")
        if hz:
            self.frep_mhz = float(hz) / 1e6
        self.dataset_changed.emit()

    def set_frep(self, mhz: float):
        self.frep_mhz = float(mhz)

    def set_calib_factor(self, factor: complex):
        self.calib_factor = complex(factor)

    def set_view(
        self, frame: int = None, sequence: int = None, channel: int = None
    ):
        changed = False
        if frame is not None and frame != self.frame:
            self.frame = frame
            changed = True
        if sequence is not None and sequence != self.sequence:
            self.sequence = sequence
            changed = True
        if channel is not None and channel != self.channel:
            self.channel = channel
            changed = True
        if changed:
            self.view_changed.emit()

    def set_smoothing(self, enabled: bool, kernel: int = None):
        changed = enabled != self.smooth
        self.smooth = enabled
        if kernel is not None and kernel != self.smooth_kernel:
            self.smooth_kernel = kernel
            self._smooth_cache.clear()
            changed = True
        if changed:
            self.smoothing_changed.emit()

    # ------------------------------------------------------------------ #
    # Accessors — return current 2D slice ready for display               #
    # ------------------------------------------------------------------ #

    def has_data(self) -> bool:
        return self.dataset is not None

    def get_photon_count(self) -> np.ndarray | None:
        """Returns (line, pixel) uint32 array for current selection."""
        if not self.has_data() or "photon_count" not in self.dataset:
            return None
        arr = (
            self.dataset["photon_count"]
            .isel(
                frame=self.frame,
                sequence=self.sequence,
                channel=self.channel,
            )
            .values
        )
        if self.smooth:
            arr = self._get_smoothed("photon_count", arr)
        return arr

    def get_mean_arrival_time(self) -> np.ndarray | None:
        """Returns (line, pixel) float32 array for current selection."""
        if not self.has_data() or "mean_arrival_time" not in self.dataset:
            return None
        return (
            self.dataset["mean_arrival_time"]
            .isel(
                frame=self.frame,
                sequence=self.sequence,
                channel=self.channel,
            )
            .values
        )

    def get_phasor(self) -> tuple | None:
        """Returns (phasor_g, phasor_s) tuple of (line, pixel) arrays."""
        if not self.has_data():
            return None
        if "phasor_g" not in self.dataset or "phasor_s" not in self.dataset:
            return None
        sel = dict(
            frame=self.frame, sequence=self.sequence, channel=self.channel
        )
        g = self.dataset["phasor_g"].isel(**sel).values
        s = self.dataset["phasor_s"].isel(**sel).values
        return g, s

    def get_tcspc_histogram(self) -> np.ndarray | None:
        """Returns (tcspc_channel,) decay array for current frame/channel."""
        if not self.has_data() or "tcspc_histogram" not in self.dataset:
            return None
        ds = self.dataset["tcspc_histogram"]
        sel = {}
        if "frame" in ds.sizes:
            sel["frame"] = self.frame
        if "channel" in ds.sizes:
            sel["channel"] = self.channel
        return ds.isel(**sel).values

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get_smoothed(self, key: str, arr: np.ndarray) -> np.ndarray:
        cache_key = (
            key,
            self.frame,
            self.sequence,
            self.channel,
            self.smooth_kernel,
        )
        if cache_key not in self._smooth_cache:
            from scipy.signal import convolve2d

            kernel = np.ones(
                (self.smooth_kernel, self.smooth_kernel), dtype=np.float32
            )
            self._smooth_cache[cache_key] = convolve2d(
                arr.astype(np.float32), kernel, mode="same"
            ).astype(arr.dtype)
        return self._smooth_cache[cache_key]
