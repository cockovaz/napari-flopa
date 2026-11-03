import numpy as np

from napari_flopa.core.processing.phasor import lifetime_ticks
from napari_flopa.ui.style import MPL

#: Shared extent of every phasor plot (G, S).
XLIM = (-0.05, 1.05)
YLIM = (-0.05, 0.80)


def draw_semicircle(ax, color: str = "white", alpha: float = 0.5) -> None:
    """Universal semicircle (centre (0.5, 0), radius 0.5) and its base line."""
    theta = np.linspace(0, np.pi, 300)
    ax.plot(
        0.5 + 0.5 * np.cos(theta),
        0.5 * np.sin(theta),
        color=color,
        linewidth=0.8,
        alpha=alpha,
        zorder=5,
    )
    ax.plot([0, 1], [0, 0], color=color, linewidth=0.5, alpha=alpha, zorder=5)


def draw_full_circle(ax, color: str = "white", alpha: float = 0.35) -> None:
    """Unit circle (centre (0, 0), radius 1), dashed.

    The full range of the cosine / sine transform: every phasor lies on or
    inside it.
    """
    theta = np.linspace(0, 2 * np.pi, 600)
    ax.plot(
        np.cos(theta),
        np.sin(theta),
        color=color,
        linewidth=0.6,
        linestyle=(0, (4, 4)),
        alpha=alpha,
        zorder=5,
    )


def draw_grid(ax) -> None:
    """Dashed grid behind the data, as on the Decay plot."""
    ax.grid(True, color=MPL.GRID, linewidth=0.6, linestyle="--", alpha=1.0)
    ax.set_axisbelow(True)


def draw_lifetime_ticks(ax, f_rep_hz: float) -> None:
    """Monoexponential lifetime marks on the semicircle, ticked outward.

    Which lifetimes are marked is decided in core (:func:`lifetime_ticks`).
    Tick and label are sized in screen points, so they keep their size when
    the plot is zoomed, and both are hidden once their point leaves the view.
    """
    center = np.array([0.5, 0.0])
    for tau_ns, g_t, s_t in lifetime_ticks(f_rep_hz):
        v = np.array([g_t, s_t]) - center
        nrm = np.linalg.norm(v)
        if nrm < 1e-9:
            continue
        v_u = v / nrm
        # A 2-point asterisk marker is a line, vertical at angle 0.
        angle = np.degrees(np.arctan2(v_u[1], v_u[0])) - 90
        ax.plot(
            g_t,
            s_t,
            marker=(2, 2, angle),
            markersize=6,
            markeredgewidth=0.8,
            color="white",
            alpha=0.6,
            zorder=4,
        )
        ax.annotate(
            f"{tau_ns:g}",
            (g_t, s_t),
            xytext=tuple(9 * v_u),
            textcoords="offset points",
            fontsize=6,
            ha="center",
            va="center",
            color="white",
            alpha=0.7,
            zorder=4,
            annotation_clip=True,
        )


def count_negative(g, s) -> tuple[int, int]:
    """Finite points with G < 0 and with S < 0, counted separately."""
    g = np.asarray(g, dtype=float).ravel()
    s = np.asarray(s, dtype=float).ravel()
    ok = np.isfinite(g) & np.isfinite(s)
    return int((ok & (g < 0)).sum()), int((ok & (s < 0)).sum())


def negative_note(g, s) -> str:
    """Status-line note on points below zero, or '' when there are none."""
    n_g, n_s = count_negative(g, s)
    parts = [
        f"{n:,} with {axis} < 0" for n, axis in ((n_g, "G"), (n_s, "S")) if n
    ]
    return f" {' and '.join(parts)}." if parts else ""


def finalise_phasor_axes(ax, full_circle: bool = False) -> None:
    """Limits, equal aspect and the dark-theme styling of a phasor plot.

    With *full_circle* the view spans -1 to 1 on both axes (unit circle).
    """
    if full_circle:
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
    else:
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("G", color=MPL.TICK)
    ax.set_ylabel("S", color=MPL.TICK)
    ax.tick_params(colors=MPL.TICK, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor(MPL.SPINE)
