import numpy as np

#: Phasor plot extent shared with the Analyzer axes, so KDE grids cover it.
PLOT_XLIM = (-0.05, 1.05)
PLOT_YLIM = (-0.05, 0.80)


def _finite(g, s):
    g = np.asarray(g, dtype=float)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(g) & np.isfinite(s)
    return g[ok], s[ok]


def covariance_ellipse(g, s, n_sigma: float = 2.0):
    """``(cx, cy, width, height, angle_deg)`` of the n-σ covariance ellipse.

    Width and height are full axis lengths (2·n·√eigenvalue), the form
    ``matplotlib.patches.Ellipse`` takes. None with fewer than 3 points or a
    degenerate covariance.
    """
    g, s = _finite(g, s)
    if g.size < 3:
        return None
    cov = np.cov(g, s)
    if not np.all(np.isfinite(cov)):
        return None
    vals, vecs = np.linalg.eigh(cov)
    if vals[-1] <= 0:
        return None
    vals = np.clip(vals, 0, None)
    major = vecs[:, -1]
    angle = float(np.degrees(np.arctan2(major[1], major[0])))
    width = 2 * n_sigma * float(np.sqrt(vals[-1]))
    height = 2 * n_sigma * float(np.sqrt(vals[0]))
    return float(g.mean()), float(s.mean()), width, height, angle


def kde_grid(g, s, n: int = 200):
    """Gaussian density of the points on an *n*×*n* grid over the plot extent.

    A binned KDE: a fine 2-D histogram smoothed with a Gaussian of Scott's-
    rule bandwidth per axis. It matches ``scipy.stats.gaussian_kde`` closely
    for contouring, scales to any point count, and avoids scipy.linalg.
    Returns ``(X, Y, Z)`` or None with too few points or zero spread.
    """
    from scipy.ndimage import gaussian_filter

    g, s = _finite(g, s)
    if g.size < 5:
        return None
    sd_g, sd_s = float(np.std(g)), float(np.std(s))
    if sd_g <= 0 or sd_s <= 0:
        return None
    factor = g.size ** (-1.0 / 6.0)  # Scott's rule for 2-D
    xe = np.linspace(*PLOT_XLIM, n + 1)
    ye = np.linspace(*PLOT_YLIM, n + 1)
    hist, _, _ = np.histogram2d(g, s, bins=[xe, ye])
    sigma_bins = (
        sd_g * factor / (xe[1] - xe[0]),
        sd_s * factor / (ye[1] - ye[0]),
    )
    Z = gaussian_filter(hist, sigma=sigma_bins, mode="constant").T
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    X, Y = np.meshgrid(xc, yc)
    return X, Y, Z


def contour_levels(Z, fractions=(0.25, 0.5, 0.75)) -> list[float]:
    """Density thresholds whose regions enclose *fractions* of the mass.

    Highest-density levels, so a 50 % contour means the same thing for every
    group regardless of its spread. Returned ascending, as contour() wants.
    """
    z = np.sort(np.asarray(Z, dtype=float).ravel())[::-1]
    total = z.sum()
    if total <= 0:
        return []
    cum = np.cumsum(z) / total
    levels = {
        float(z[min(np.searchsorted(cum, f), z.size - 1)]) for f in fractions
    }
    return sorted(lv for lv in levels if lv > 0)
