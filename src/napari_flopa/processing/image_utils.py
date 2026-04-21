import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy.signal import convolve2d


def smooth_weighted(
    array: NDArray[np.floating],
    count: NDArray[np.integer],
    size: int = 3,
) -> tuple[NDArray[np.float32], NDArray[np.uint32]]:
    """
    Apply photon-weighted box smoothing to a real-valued 2-D scalar field.

    Smooths `array` using a uniform square kernel of side `size`, weighting
    each pixel by `count`.  Invalid entries (NaN, Inf, or zero/negative count)
    are excluded from both numerator and denominator so they do not bleed into
    neighbouring pixels.  Output pixels with no valid kernel contribution are
    set to NaN.

    Intended for scalar fields such as mean_arrival_time.  Do NOT use this
    for phasor data — use ptuio.utils.smooth_phasor instead, which operates
    on the complex g+i·s representation directly and handles both components
    in a single pass.
    """
    array = np.asarray(array)
    count = np.asarray(count)
    if array.ndim != 2 or count.ndim != 2:
        raise ValueError("array and count must both be 2D arrays")
    assert (
        array.shape == count.shape
    ), "array and count must have the same shape"
    if not (isinstance(size, int) and size > 0):
        raise ValueError("size must be a positive integer")
    kernel = np.ones((size, size), dtype=np.float32)
    valid = np.isfinite(array) & (count > 0)
    num = convolve2d(
        np.where(valid, array * count, 0).astype(np.float32),
        kernel,
        mode="same",
    )
    den = convolve2d(
        np.where(valid, count, 0).astype(np.float32), kernel, mode="same"
    )
    out = np.full_like(array, np.nan, dtype=np.float32)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out, np.asarray(den, dtype=np.uint32)


def smooth_count(
    count: NDArray[np.integer], size: int = 3
) -> NDArray[np.uint32]:
    """
    Box-sum smoothing of a photon-count array.

    Complements smooth_weighted for cases where
    only the accumulated count (not the weighted scalar) is needed — e.g.
    updating the intensity layer after smoothing.
    """
    count = np.asarray(count)
    if count.ndim != 2:
        raise ValueError("count must be a 2D array")
    kernel = np.ones((size, size), dtype=np.float32)
    return np.asarray(convolve2d(count, kernel, mode="same"), dtype=np.uint32)


def aggregate_dataset(ds: xr.Dataset, dims) -> xr.Dataset:
    if isinstance(dims, str):
        dims = [dims]
    if not dims:
        return ds
    missing = [d for d in dims if d not in ds.sizes]
    if missing:
        raise ValueError(f"Dims not in dataset: {missing}")

    out = {}
    photon_sum = None
    if "photon_count" in ds:
        photon_sum = ds["photon_count"].sum(dim=dims, keepdims=True)
        out["photon_count"] = photon_sum.astype("uint64")

    for var in ["mean_arrival_time", "phasor_g", "phasor_s"]:
        if var in ds and photon_sum is not None:
            valid = ds[var].notnull()
            num = (ds[var].where(valid, 0) * ds["photon_count"]).sum(
                dim=dims, keepdims=True
            )
            den = (
                ds["photon_count"].where(valid, 0).sum(dim=dims, keepdims=True)
            )
            out[var] = xr.where(photon_sum > 0, num / den, np.nan).astype(
                "float32"
            )

    if "tcspc_histogram" in ds:
        out["tcspc_histogram"] = (
            ds["tcspc_histogram"].sum(dim=dims, keepdims=True).astype("uint64")
        )

    if not out:
        return ds

    out_ds = xr.Dataset(out)
    coord_ds = ds.coords.to_dataset()
    indexers = {d: slice(0, 1) for d in dims if d in coord_ds.sizes}
    coord_ds = coord_ds.isel(indexers)

    # Ensure every reduced dimension has a dimension coordinate, even when
    # the original dataset had no explicit coordinate for that dim.  Without
    # this, the merged result can have size-1 dims with no coordinate, which
    # breaks downstream isel / sel calls that expect a coordinate to exist.
    for d in dims:
        if d in ds.dims and d not in coord_ds:
            if d in ds.coords:
                coord_ds[d] = ds[d].isel({d: slice(0, 1)})
            else:
                coord_ds[d] = xr.DataArray(np.arange(1), dims=(d,))

    return xr.merge([out_ds, coord_ds], compat="override")
