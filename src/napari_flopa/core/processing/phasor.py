"""
Monoexponential phasor maths.

The phasor of a single-exponential decay of lifetime τ at excitation angular
frequency ω = 2πf lies on the universal semicircle::

    g = 1 / (1 + (ωτ)²)
    s = ωτ / (1 + (ωτ)²)

Everything that needs that relation — the lifetime tick marks drawn on the
phasor plot and the calibration factor derived from a reference dye — goes
through :func:`monoexp_phasor`, so the formula exists once.

The frequency must be the one the phasor data itself was reconstructed with,
i.e. ``repetition_rate`` from the file header (see ``core/io/loader.py``), not a
user-entered value; otherwise the reference points do not describe the data.
"""

import numpy as np


def monoexp_phasor(tau_s, f_rep_hz: float):
    """Phasor coordinates of a monoexponential decay.

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


def lifetime_ticks(
    f_rep_hz: float, tau_max_ns: int | None = None
) -> list[tuple[int, float, float]]:
    """Reference lifetimes to mark on the universal semicircle.

    Args:
        f_rep_hz: Excitation repetition rate in Hz.
        tau_max_ns: Largest lifetime to include, in ns. Defaults to half the
            laser period — beyond that the points crowd into the origin and
            stop being readable (13 ns at 40 MHz).

    Returns:
        ``[(tau_ns, g, s), …]`` for τ = 1 … *tau_max_ns* ns.
    """
    if f_rep_hz <= 0:
        return []
    if tau_max_ns is None:
        period_ns = 1e9 / f_rep_hz
        tau_max_ns = int(np.ceil(period_ns / 2))
    tau_max_ns = max(1, int(tau_max_ns))

    ticks = []
    for tau_ns in range(1, tau_max_ns + 1):
        g, s = monoexp_phasor(tau_ns * 1e-9, f_rep_hz)
        ticks.append((tau_ns, g, s))
    return ticks


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


def correct_lifetime(lt_ns, delay_ns: float, period_ns: float, counts=None):
    """Subtract *delay_ns* from a lifetime image, wrapping into [0, period).

    Values that would go negative wrap by one laser period, the circular
    shift the Decay tab applies to curves. Pixels with zero *counts* stay 0.
    """
    out = np.mod(np.asarray(lt_ns, dtype=np.float32) - delay_ns, period_ns)
    if counts is not None:
        out[np.asarray(counts) == 0] = 0
    return out


def lifetime_correction(ds, factor: complex) -> tuple[float, float] | None:
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


def phasor_lifetimes(g, s, f_rep_hz: float):
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
    g, s, tau1_ns: float, tau2_ns: float, f_rep_hz: float
):
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
    g, s, tau1_ns: float, tau2_ns: float, tau3_ns: float, f_rep_hz: float
):
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
