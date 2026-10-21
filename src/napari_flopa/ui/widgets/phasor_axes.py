import numpy as np

from napari_flopa.core.processing.phasor import lifetime_ticks
from napari_flopa.ui.style import MPL

#: Shared extent of every phasor plot (G, S).
XLIM = (-0.05, 1.05)
YLIM = (-0.05, 0.80)


def draw_semicircle(ax, color: str = "white", alpha: float = 0.5) -> None:
    """Universal semicircle: centre (0.5, 0), radius 0.5."""
    theta = np.linspace(0, np.pi, 300)
    ax.plot(
        0.5 + 0.5 * np.cos(theta),
        0.5 * np.sin(theta),
        color=color,
        linewidth=0.8,
        alpha=alpha,
        zorder=5,
    )


def draw_lifetime_ticks(ax, f_rep_hz: float) -> None:
    """Monoexponential lifetime marks on the semicircle, ticked outward.

    Which lifetimes are marked is decided in core (:func:`lifetime_ticks`).
    """
    center = np.array([0.5, 0.0])
    tick_len = 0.018
    for tau_ns, g_t, s_t in lifetime_ticks(f_rep_hz):
        p = np.array([g_t, s_t])
        v = p - center
        nrm = np.linalg.norm(v)
        if nrm < 1e-9:
            continue
        v_u = v / nrm
        p1, p2 = p - tick_len / 2 * v_u, p + tick_len / 2 * v_u
        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]], "w-", lw=0.8, alpha=0.6, zorder=4
        )
        lp = p + tick_len * 1.7 * v_u
        ax.text(
            lp[0],
            lp[1],
            f"{tau_ns:g}",
            fontsize=6,
            ha="center",
            va="center",
            color="white",
            alpha=0.7,
            zorder=4,
        )


def count_outside(g, s) -> int:
    """Finite points that fall outside the fixed phasor-plot extent."""
    g = np.asarray(g, dtype=float).ravel()
    s = np.asarray(s, dtype=float).ravel()
    ok = np.isfinite(g) & np.isfinite(s)
    inside = (g >= XLIM[0]) & (g <= XLIM[1]) & (s >= YLIM[0]) & (s <= YLIM[1])
    return int((ok & ~inside).sum())


def finalise_phasor_axes(ax) -> None:
    """Limits, equal aspect and the dark-theme styling of a phasor plot."""
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("G", color=MPL.TICK)
    ax.set_ylabel("S", color=MPL.TICK)
    ax.tick_params(colors=MPL.TICK, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(MPL.SPINE)
