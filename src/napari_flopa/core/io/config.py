"""
Scan settings: one description of how a .ptu is reconstructed.

``ScanSettings`` owns every value that describes a reconstruction — the ones
tttrkit's ``ScanConfig`` takes, plus the plugin's own (TCSPC bins, calibration
factor, source filename) and where each value came from. Everything else is
derived from it:

    widgets ─► ScanSettings ─┬─► to_scan_config()  → tttrkit ScanConfig
                             ├─► to_json_dict()    → the .json config file
                             └─► ds.attrs["scan_config"]

That keeps tttrkit's vocabulary in exactly one method. Its own
``ScanConfig.to_dict()`` is deliberately not used for serialisation: it renames
the marker keys and drops ``max_detector`` and both marker delays.

JSON schema (every key optional when reading)::

    {
      "ptu_filename": "scan.ptu",       # recorded for reference only
      "scan": {
        "frames", "lines", "pixels", "sequences",
        "accumulations": [1, 3, ...],   # one int per sequence
        "max_detector", "tcspc_bins",
        "bidirectional", "harmonic_scan", "laser_duty",
        "line_start_marker_delay", "line_stop_marker_delay",
        "line_duration",                # omitted when tttrkit should infer it
        "frame_start_marker", "line_start_marker", "line_stop_marker"
      },
      "calibration": {"factor": "1+0j"},
      "provenance": {"frames": "metadata", ...}   # omitted when empty
    }
"""

import json
from dataclasses import dataclass, field, fields

#: Marker delays and the line duration are stored in seconds — tttrkit takes
#: them as sync counts. The UI enters delays in µs and durations in ms.
US = 1e-6
MS = 1e-3

DEFAULT_LASER_DUTY = 0.6


@dataclass
class ScanSettings:
    """Everything needed to reconstruct a file, in the plugin's own vocabulary."""

    # ── geometry ────────────────────────────────────────────────────────
    frames: int = 1
    lines: int = 512
    pixels: int = 512
    accumulations: tuple[int, ...] = (1,)
    max_detector: int = 4

    # ── scan modes ──────────────────────────────────────────────────────
    bidirectional: bool = False
    harmonic_scan: bool = False
    laser_duty: float = DEFAULT_LASER_DUTY
    line_start_marker_delay: float = 0.0
    line_stop_marker_delay: float = 0.0
    #: None lets tttrkit measure it from the markers of the first chunk.
    line_duration: float | None = None

    # ── marker bits (one of 1, 2, 4, 8) ─────────────────────────────────
    frame_start_marker: int = 4
    line_start_marker: int = 1
    line_stop_marker: int = 2

    # ── plugin extras, not part of tttrkit's ScanConfig ─────────────────
    tcspc_bins: int | None = None
    calib_factor: str = "1+0j"
    ptu_filename: str | None = None

    #: field name → 'metadata' | 'default' | 'user' | 'estimated'
    sources: dict[str, str] = field(default_factory=dict)

    @property
    def sequences(self) -> int:
        """Sequence count — always ``len(accumulations)``, never stored twice."""
        return len(self.accumulations)

    @property
    def effective_marker_delays(self) -> tuple[float, float]:
        """Line start/stop delays as applied — independent of scan mode."""
        return self.line_start_marker_delay, self.line_stop_marker_delay

    def to_scan_config(self, sync_rate: float):
        """Build the tttrkit ``ScanConfig`` — the only place its names appear.

        *sync_rate* is the laser repetition rate in Hz: tttrkit takes the
        marker delays as sync counts, so the stored seconds are scaled here.
        """
        # Imported lazily so save/load stay usable without tttrkit installed.
        from tttrkit.ptuio.reconstructor import ScanConfig

        start_delay, stop_delay = self.effective_marker_delays
        return ScanConfig(
            frames=self.frames,
            lines=self.lines,
            pixels=self.pixels,
            line_accumulations=tuple(self.accumulations) or (1,),
            max_detector=self.max_detector,
            bidirectional=self.bidirectional,
            harmonic_scan=self.harmonic_scan,
            laser_duty=self.laser_duty,
            line_start_marker_delay=seconds_to_nsync(start_delay, sync_rate),
            line_stop_marker_delay=seconds_to_nsync(stop_delay, sync_rate),
            line_duration=(
                None
                if self.line_duration is None
                else seconds_to_nsync(self.line_duration, sync_rate)
            ),
            frame_start_marker_channel=self.frame_start_marker,
            line_start_marker_channel=self.line_start_marker,
            line_stop_marker_channel=self.line_stop_marker,
        )

    def to_json_dict(self) -> dict:
        """Serialisable form — the .json file and ``ds.attrs['scan_config']``."""
        start_delay, stop_delay = self.effective_marker_delays
        cfg: dict = {
            "scan": {
                "frames": int(self.frames),
                "lines": int(self.lines),
                "pixels": int(self.pixels),
                "sequences": self.sequences,
                "accumulations": [int(a) for a in self.accumulations],
                "max_detector": int(self.max_detector),
                "bidirectional": bool(self.bidirectional),
                "harmonic_scan": bool(self.harmonic_scan),
                "laser_duty": float(self.laser_duty),
                "line_start_marker_delay": float(start_delay),
                "line_stop_marker_delay": float(stop_delay),
                "frame_start_marker": int(self.frame_start_marker),
                "line_start_marker": int(self.line_start_marker),
                "line_stop_marker": int(self.line_stop_marker),
            },
            "calibration": {"factor": str(self.calib_factor)},
        }
        if self.tcspc_bins is not None:
            cfg["scan"]["tcspc_bins"] = int(self.tcspc_bins)
        if self.line_duration is not None:
            cfg["scan"]["line_duration"] = float(self.line_duration)
        if self.ptu_filename:
            cfg["ptu_filename"] = str(self.ptu_filename)
        if self.sources:
            cfg["provenance"] = dict(self.sources)
        return cfg

    @classmethod
    def from_json_dict(cls, cfg: dict) -> "ScanSettings":
        """Read a config dict, filling anything absent with the defaults.

        Tolerant on purpose: configs written by older versions are missing
        whole keys, and the Batch tab used to write ``accum_per_seq`` as a
        comma-separated string instead of a list.
        """
        scan = cfg.get("scan", {})
        out = cls()

        accum = scan.get("accumulations", scan.get("accum_per_seq"))
        if isinstance(accum, str):  # legacy "1,3" form
            accum = [int(p) for p in accum.split(",") if p.strip()]
        if accum:
            out.accumulations = tuple(int(a) for a in accum)

        simple = {
            "frames": int,
            "lines": int,
            "pixels": int,
            "max_detector": int,
            "bidirectional": bool,
            "harmonic_scan": bool,
            "laser_duty": float,
            "line_start_marker_delay": float,
            "line_stop_marker_delay": float,
            "line_duration": float,
            "frame_start_marker": int,
            "line_start_marker": int,
            "line_stop_marker": int,
            "tcspc_bins": int,
        }
        for name, cast in simple.items():
            if scan.get(name) is not None:
                setattr(out, name, cast(scan[name]))

        factor = (cfg.get("calibration") or {}).get("factor")
        if factor:
            out.calib_factor = str(factor)
        if cfg.get("ptu_filename"):
            out.ptu_filename = str(cfg["ptu_filename"])
        out.sources = dict(cfg.get("provenance") or {})
        return out

    def replace(self, **changes) -> "ScanSettings":
        """Copy with *changes* applied — rejects unknown field names."""
        known = {f.name for f in fields(self)}
        unknown = set(changes) - known
        if unknown:
            raise TypeError(
                f"Unknown ScanSettings field(s): {sorted(unknown)}"
            )
        return type(self)(**{**self.__dict__, **changes})


def seconds_to_nsync(delay_s: float, sync_rate: float) -> int:
    """Marker delay in seconds → the sync counts tttrkit's ScanConfig takes."""
    return int(round(float(delay_s) * float(sync_rate)))


def nsync_to_seconds(delay_nsync: float, sync_rate: float) -> float:
    """Inverse of :func:`seconds_to_nsync`."""
    return float(delay_nsync) / float(sync_rate)


def scan_config_from_dict(cfg: dict, sync_rate: float):
    """Convenience: config dict → tttrkit ``ScanConfig``."""
    return ScanSettings.from_json_dict(cfg).to_scan_config(sync_rate)


def save_config(path, cfg: dict) -> None:
    """Write *cfg* to *path* as indented JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def load_config(path) -> dict:
    """Read and return the JSON config at *path*."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
