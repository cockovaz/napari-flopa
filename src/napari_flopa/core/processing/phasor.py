import numpy as np
import xarray as xr
from numpy.typing import ArrayLike


def monoexp_phasor(
    tau_s: ArrayLike, f_rep_hz: float
) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
    """Phasor coordinates of a monoexponential decay.

    The universal semicircle, with ω = 2πf::

        g = 1 / (1 + (ωτ)²)
        s = ωτ / (1 + (ωτ)²)

    Lifetime ticks, the calibration factor and the component fractions all
    go through this function. *f_rep_hz* must be the rate the phasor data was
    reconstructed with (``repetition_rate`` from the file header), not a
    user-entered value, or the reference points do not describe the data.

    Args:
        tau_s: Lifetime in **seconds**; scalar or array.
        f_rep_hz: Excitation repetition rate in Hz.

    Returns:
        ``(g, s)`` — floats for a scalar *tau_s*, arrays otherwise. τ = 0 maps to
        (1, 0) and τ → ∞ to (0, 0), the two ends of the semicircle.
    """
    omega_tau = 2 * np.pi * f_rep_hz * np.asarray(tau_s, dtype=float)
    denom = 1.0 + omega_tau**2
    g = 1.0 / denom
    s = omega_tau / denom
    if np.isscalar(tau_s) or np.ndim(tau_s) == 0:
        return float(g), float(s)
    return g, s


def _tick_priority(tau_ns: float) -> int:
    """Lower = kept first: 1 and 0.5 ns, then multiples of 10, 5, 2, 1."""
    if tau_ns in (0.5, 1.0):
        return 0
    for rank, step in enumerate((10, 5, 2), start=1):
        if tau_ns % step == 0:
            return rank
    return 4


def lifetime_ticks(
    f_rep_hz: float,
    tau_max_ns: int | None = None,
    min_spacing: float = 0.05,
) -> list[tuple[float, float, float]]:
    """Reference lifetimes to mark on the universal semicircle.

    Candidates are 0.5 ns and every whole ns up to *tau_max_ns*. They are
    accepted in order of roundness (see :func:`_tick_priority`) only if their
    phasor point is at least *min_spacing* (phasor units) from every tick
    already accepted, so low repetition rates — where long lifetimes crowd
    towards the origin — automatically keep only the round values.

    Args:
        f_rep_hz: Excitation repetition rate in Hz.
        tau_max_ns: Largest lifetime to consider, in ns. Defaults to half the
            laser period.
        min_spacing: Minimum distance between two tick points in (g, s).

    Returns:
        ``[(tau_ns, g, s), …]`` sorted by lifetime.
    """
    if f_rep_hz <= 0:
        return []
    if tau_max_ns is None:
        tau_max_ns = int(np.ceil(1e9 / f_rep_hz / 2))
    tau_max_ns = max(1, int(tau_max_ns))

    candidates = [0.5] + [float(t) for t in range(1, tau_max_ns + 1)]
    chosen: list[tuple[float, float, float]] = []
    for tau_ns in sorted(candidates, key=lambda t: (_tick_priority(t), t)):
        g, s = monoexp_phasor(tau_ns * 1e-9, f_rep_hz)
        forced = _tick_priority(tau_ns) == 0
        if forced or all(
            np.hypot(g - cg, s - cs) >= min_spacing for _, cg, cs in chosen
        ):
            chosen.append((tau_ns, g, s))
    return sorted(chosen)


def apply_calibration(
    g: ArrayLike, s: ArrayLike, factor: complex
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrated phasor coordinates: ``(g + i·s) · factor``.

    *factor* is the complex value from :func:`calibration_factor` (rotation by
    the IRF phase plus modulation scaling). NaNs stay NaN.
    """
    c = (np.asarray(g, dtype=float) + 1j * np.asarray(s, dtype=float)) * factor
    return c.real, c.imag


def calibration_factor(
    tau_ns: float, measured: complex, f_rep_hz: float
) -> complex:
    """Factor that maps a *measured* reference phasor onto its true position.

    Applied as ``phasor * factor`` (see the Phasor panel), it corrects the
    instrument response using a dye of known lifetime *tau_ns*.

    Raises:
        ValueError: If *measured* is (numerically) zero, leaving the factor
            undefined.
    """
    if abs(measured) < 1e-12:
        raise ValueError(
            "Measured phasor is zero — cannot compute calibration factor."
        )
    g_th, s_th = monoexp_phasor(tau_ns * 1e-9, f_rep_hz)
    return complex(g_th, s_th) / measured


def irf_delay_ns(
    factor: complex, omega_per_channel: float, res_ns: float
) -> float:
    """IRF delay encoded in the phase of a calibration *factor*, in ns.

    The phasor is built as ``exp(+iωt)``, so a delay Δ rotates it by +ωΔ and
    the factor undoes that rotation: Δ = −arg(c)/ω. *omega_per_channel* must
    be the one the phasor was reconstructed with (``ds["omega"]``).
    """
    return float(-np.angle(factor) / omega_per_channel * res_ns)


def correct_lifetime(
    lt_ns: ArrayLike,
    delay_ns: float,
    period_ns: float,
    counts: ArrayLike | None = None,
) -> np.ndarray:
    """Subtract *delay_ns* from a lifetime image, wrapping into [0, period).

    Values that would go negative wrap by one laser period, the circular
    shift the Decay tab applies to curves. Pixels with zero *counts* stay 0.
    """
    out = np.mod(np.asarray(lt_ns, dtype=np.float32) - delay_ns, period_ns)
    if counts is not None:
        out[np.asarray(counts) == 0] = 0
    return out


def lifetime_correction(
    ds: xr.Dataset, factor: complex
) -> tuple[float, float] | None:
    """``(delay_ns, period_ns)`` for correcting *ds*'s lifetime, or None.

    None when there is nothing to correct: identity factor, or a dataset
    without the reconstruction ``omega`` or repetition rate.
    """
    if factor == 1 + 0j or "omega" not in ds:
        return None
    omega = float(ds["omega"].values)
    ip = ds.attrs.get("instrument_params", {})
    rep = float(ip.get("repetition_rate", 0) or 0)
    if omega == 0 or rep <= 0:
        return None
    res_ns = float(ip.get("tcspc_resolution_ns", 1.0))
    if ip.get("resolution_unit", "ns") == "ns":
        period = 1e9 / rep
    else:
        # No resolution in the header: lifetimes are in channels, and the
        # reconstruction omega is the only period estimate in those units.
        period = 2 * np.pi / omega * res_ns
    return irf_delay_ns(factor, omega, res_ns), period


def phasor_lifetimes(
    g: ArrayLike, s: ArrayLike, f_rep_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """Phase and modulation lifetimes of phasor points, in ns.

    τφ = s / (g·ω) and τm = √(1/M² − 1) / ω with M² = g² + s². Both equal
    the true lifetime for a monoexponential decay on the semicircle. Points
    with g ≤ 0 or M outside (0, 1) have no defined lifetime and are NaN.
    """
    g_arr = np.asarray(g, dtype=float)
    s_arr = np.asarray(s, dtype=float)
    omega = 2 * np.pi * f_rep_hz
    m2 = g_arr**2 + s_arr**2
    with np.errstate(divide="ignore", invalid="ignore"):
        tau_phi = s_arr / (g_arr * omega) * 1e9
        tau_mod = np.sqrt(1.0 / m2 - 1.0) / omega * 1e9
    tau_phi[~(np.isfinite(tau_phi) & (g_arr > 0))] = np.nan
    tau_mod[~(np.isfinite(tau_mod) & (m2 > 0) & (m2 < 1))] = np.nan
    return tau_phi, tau_mod


def component_fractions_2(
    g: ArrayLike, s: ArrayLike, tau1_ns: float, tau2_ns: float, f_rep_hz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fractional contributions of two monoexponential components.

    Each data point (g, s) is projected onto the line between the two
    single-exponential phasor positions.  Returns ``(f1, f2, residual)``
    arrays with ``f1 + f2 == 1``.  *residual* is the perpendicular distance
    from each point to the line through the two components.  Points that
    fall outside the segment (any fraction < 0 or > 1) are set to NaN in
    all three outputs.
    """
    g1, s1 = monoexp_phasor(tau1_ns * 1e-9, f_rep_hz)
    g2, s2 = monoexp_phasor(tau2_ns * 1e-9, f_rep_hz)
    dg, ds = g1 - g2, s1 - s2
    denom = dg**2 + ds**2
    g_arr = np.asarray(g, dtype=float)
    s_arr = np.asarray(s, dtype=float)
    if denom < 1e-20:
        nan = np.full_like(g_arr, np.nan)
        return nan, nan.copy(), nan.copy()
    length = np.sqrt(denom)
    f1 = ((g_arr - g2) * dg + (s_arr - s2) * ds) / denom
    f2 = 1.0 - f1
    residual = np.abs((g_arr - g2) * ds - (s_arr - s2) * dg) / length
    outside = (f1 < 0) | (f1 > 1)
    f1[outside] = np.nan
    f2[outside] = np.nan
    residual[outside] = np.nan
    return f1, f2, residual


def component_fractions_3(
    g: ArrayLike,
    s: ArrayLike,
    tau1_ns: float,
    tau2_ns: float,
    tau3_ns: float,
    f_rep_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Barycentric fractions for three monoexponential components.

    Solves ``P = f1·P1 + f2·P2 + f3·P3`` with ``f1+f2+f3 = 1`` via the 2×2
    system in (g, s).  Returns ``(f1, f2, f3)`` arrays.  Points outside the
    triangle (any fraction < 0) are set to NaN.
    """
    g1, s1 = monoexp_phasor(tau1_ns * 1e-9, f_rep_hz)
    g2, s2 = monoexp_phasor(tau2_ns * 1e-9, f_rep_hz)
    g3, s3 = monoexp_phasor(tau3_ns * 1e-9, f_rep_hz)
    a11, a12 = g1 - g3, g2 - g3
    a21, a22 = s1 - s3, s2 - s3
    det = a11 * a22 - a12 * a21
    g_arr = np.asarray(g, dtype=float)
    s_arr = np.asarray(s, dtype=float)
    if abs(det) < 1e-20:
        nan = np.full_like(g_arr, np.nan)
        return nan, nan.copy(), nan.copy()
    rg = g_arr - g3
    rs = s_arr - s3
    f1 = (a22 * rg - a12 * rs) / det
    f2 = (-a21 * rg + a11 * rs) / det
    f3 = 1.0 - f1 - f2
    outside = (f1 < 0) | (f2 < 0) | (f3 < 0)
    f1[outside] = np.nan
    f2[outside] = np.nan
    f3[outside] = np.nan
    return f1, f2, f3
