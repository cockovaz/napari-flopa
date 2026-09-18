import numpy as np
import xarray as xr
from tttrkit.ptuio.reconstructor import SegmentReconstructor
from tttrkit.ptuio.utils import (
    _read_probe_chunk,
    estimate_bidirectional_prealign,
    estimate_bidirectional_shift,
)

#: Raw TTTR records read per probe. Not photons — markers and overflow
#: records are counted too.
DEFAULT_CHUNK_RECORDS = 500_000


def _sync_rate_and_wrap(ptu_data: dict) -> tuple[float, int]:
    constants = ptu_data["constants"]
    return float(constants["repetition_rate"]), int(constants["wrap"])


def prealign(
    ptu_data: dict,
    config,
    *,
    chunk_records: int = DEFAULT_CHUNK_RECORDS,
    skip_chunks: int = 0,
) -> xr.Dataset:
    """Coarse forward/backward alignment of a bidirectional scan.

    The marker delays on *config* are ignored: tttrkit probes with its own
    symmetric margin window and reports the offset it measures against that.

    Returns the tttrkit dataset: ``forward`` / ``backward`` /
    ``backward_aligned`` over ``pixel``, the scalar ``pixel_shift`` and
    ``time_shift`` (seconds), and a ``time_axis`` coordinate (seconds, its
    own dimension) covering the probed window.
    """
    sync_rate, wrap = _sync_rate_and_wrap(ptu_data)
    try:
        return estimate_bidirectional_prealign(
            reader=ptu_data["reader"],
            cfg=config,
            laser_sync_rate=sync_rate,
            wrap=wrap,
            chunk_length=int(chunk_records),
            skip_chunks=int(skip_chunks),
            verbose=False,
        )
    except ValueError as e:
        # tttrkit filters invalid line pairs out of the durations but not out
        # of the intervals, so its result can fail to assemble.
        raise RuntimeError(
            f"Pre-alignment failed on this region: {e}\n"
            "Try a larger chunk size or skipping more chunks."
        ) from e


def optimize_shift(
    ptu_data: dict,
    config,
    *,
    max_shift_s: float = 5e-6,
    steps: int = 11,
    chunk_records: int = DEFAULT_CHUNK_RECORDS,
    skip_chunks: int = 0,
) -> xr.Dataset:
    """Refine the marker delays of *config* by correlating forward vs backward.

    ``best_shift`` in the result is **relative** to the delays already on
    *config* — add it to both of them rather than assigning it.
    """
    sync_rate, wrap = _sync_rate_and_wrap(ptu_data)
    return estimate_bidirectional_shift(
        reader=ptu_data["reader"],
        config=config,
        laser_sync_rate=sync_rate,
        wrap=wrap,
        max_shift=float(max_shift_s),
        steps=int(steps),
        chunk_length=int(chunk_records),
        skip_chunks=int(skip_chunks),
        verbose=False,
    )


def fit_failed(result: xr.Dataset) -> bool:
    """True when the Gaussian fit did not converge and ``fit`` is all NaN."""
    return bool(np.isnan(np.asarray(result["fit"].values)).all())


def segment_profiles(
    ptu_data: dict,
    config,
    *,
    chunk_records: int = DEFAULT_CHUNK_RECORDS,
    skip_chunks: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward and backward line profiles at the delays set on *config*."""
    sync_rate, wrap = _sync_rate_and_wrap(ptu_data)
    chunk, parity = _read_probe_chunk(
        ptu_data["reader"],
        config,
        wrap,
        int(chunk_records),
        int(skip_chunks),
        False,
    )
    result = SegmentReconstructor(config, sync_rate).reconstruct(chunk)

    # The reconstructor flips every odd line, which is only right when the
    # chunk starts on a forward line; dropping `parity` lines makes it so.
    lines = result.sizes["line"]
    if lines - parity < 2:
        raise RuntimeError(
            "Not enough complete lines in this region to compare directions.\n"
            "Try a larger chunk size."
        )

    photon_count = (
        result["photon_count"].isel(line=slice(parity, lines)).values
    )
    forward = photon_count[0::2].sum(axis=0).astype(float)
    backward = photon_count[1::2].sum(axis=0).astype(float)
    return forward, backward
