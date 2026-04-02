import numpy as np
import xarray as xr
from matplotlib import cm
from numpy.typing import NDArray
from scipy.signal import convolve2d


def create_FLIM_image(
    mean_photon_arrival_time: np.ndarray,
    intensity: np.ndarray,
    colormap=cm.rainbow,
    lt_min=None,
    lt_max=None,
    int_min=None,
    int_max=None,
) -> np.ndarray:
    if mean_photon_arrival_time.shape != intensity.shape:
        raise ValueError(
            "Lifetime and intensity arrays must have the same shape"
        )
    if lt_min is None or lt_max is None:
        lt_min, lt_max = np.nanmin(mean_photon_arrival_time), np.nanmax(
            mean_photon_arrival_time
        )
    if lt_max == lt_min:
        raise ValueError(f"lt_max and lt_min must differ — got {lt_min}")
    if int_min is None or int_max is None:
        int_min, int_max = np.nanmin(intensity), np.nanmax(intensity)
    if int_max == int_min:
        raise ValueError("int_max and int_min must differ")
    LT_norm = np.clip(
        (mean_photon_arrival_time - lt_min) / (lt_max - lt_min), 0, 1
    )
    LT_rgb = colormap(LT_norm)[..., :3]
    int_norm = np.clip((intensity - int_min) / (int_max - int_min), 0, 1)
    return LT_rgb * int_norm[..., np.newaxis]


def smooth_weighted(
    array: NDArray[np.floating],
    count: NDArray[np.integer],
    size: int = 3,
) -> tuple:
    array = np.asarray(array)
    count = np.asarray(count)
    if array.ndim != 2 or count.ndim != 2:
        raise ValueError("array and count must both be 2D arrays")
    assert array.shape == count.shape
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
    return xr.merge([out_ds, coord_ds], compat="override")
