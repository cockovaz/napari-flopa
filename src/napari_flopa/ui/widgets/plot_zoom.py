from collections.abc import Callable

import numpy as np

ZOOM_HINT = "Wheel: zoom, right-drag: pan, double-click: reset"
ZOOM_HINT_Y = (
    "Wheel: zoom y, right button held + wheel: zoom x, "
    "right-drag: pan, double-click: reset"
)

_STEP = 0.8  # limits shrink to 80 % per wheel step in


class PlotZoom:
    """Wheel zoom, right/middle-drag pan and double-click reset for a plot.

    x and y zoom by the same factor, so a round semicircle stays round. The
    panel keeps drawing as before and calls :meth:`after_draw` at the end
    of every draw. The user's zoom is re-applied there, so it survives
    replots until a double-click or :meth:`forget`. Left clicks are left
    alone (the phasor ROI lasso uses them).

    Args:
        canvas: The matplotlib FigureCanvas.
        get_ax: Returns the panel's current main axes (figures are rebuilt
            on every plot, and colorbar axes must not react).
        on_view_changed: Called with the axes after every view change.
        wheel_y_only: The wheel zooms only y, and only x while the right
            button is held (for curves, where the two axes are unrelated).
    """

    def __init__(
        self,
        canvas,
        get_ax: Callable,
        *,
        on_view_changed: Callable | None = None,
        wheel_y_only: bool = False,
    ):
        self._canvas = canvas
        self._get_ax = get_ax
        self._on_view_changed = on_view_changed
        self._wheel_y_only = wheel_y_only
        self._view: tuple | None = None
        self._home: tuple | None = None
        self._pan: tuple | None = None
        canvas.mpl_connect("scroll_event", self._on_scroll)
        canvas.mpl_connect("button_press_event", self._on_press)
        canvas.mpl_connect("motion_notify_event", self._on_motion)
        canvas.mpl_connect("button_release_event", self._on_release)

    @property
    def zoomed(self) -> bool:
        return self._view is not None

    def after_draw(self) -> None:
        """Record the fresh limits as home and re-apply the user's zoom."""
        ax = self._get_ax()
        if ax is None:
            return
        self._home = (ax.get_xlim(), ax.get_ylim())
        if self._view is not None:
            xlim, ylim = self._view
            if ax.get_yscale() == "log" and min(ylim) <= 0:
                self._view = None
            else:
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
        self._changed(ax, draw=False)

    def forget(self) -> None:
        """Drop the user's zoom, so the next draw shows the full view."""
        self._view = None

    def reset(self) -> None:
        """Back to the full view of the current plot."""
        ax = self._get_ax()
        self._view = None
        if ax is None or self._home is None:
            return
        ax.set_xlim(*self._home[0])
        ax.set_ylim(*self._home[1])
        self._changed(ax)

    def _own(self, event):
        ax = self._get_ax()
        return ax if ax is not None and event.inaxes is ax else None

    def _changed(self, ax, draw: bool = True) -> None:
        if self._on_view_changed is not None:
            self._on_view_changed(ax)
        if draw:
            self._canvas.draw_idle()

    @staticmethod
    def _to(axis_scale: str, values):
        v = np.asarray(values, dtype=float)
        return np.log10(v) if axis_scale == "log" else v

    @staticmethod
    def _from(axis_scale: str, values):
        v = np.asarray(values, dtype=float)
        return 10**v if axis_scale == "log" else v

    def _on_scroll(self, event):
        ax = self._own(event)
        if ax is None or event.xdata is None:
            return
        factor = _STEP if event.button == "up" else 1 / _STEP
        zoom_x = zoom_y = True
        if self._wheel_y_only:
            zoom_x = self._pan is not None  # right button held
            zoom_y = not zoom_x
        new = []
        for lim, centre, scale, on in (
            (ax.get_xlim(), event.xdata, ax.get_xscale(), zoom_x),
            (ax.get_ylim(), event.ydata, ax.get_yscale(), zoom_y),
        ):
            if not on:
                new.append(lim)
                continue
            lo, hi = self._to(scale, lim)
            c = float(self._to(scale, centre))
            new.append(
                tuple(
                    self._from(
                        scale, (c + (lo - c) * factor, c + (hi - c) * factor)
                    )
                )
            )
        ax.set_xlim(*new[0])
        ax.set_ylim(*new[1])
        self._view = (ax.get_xlim(), ax.get_ylim())
        if self._pan is not None:
            # Restart a held pan from here, or it would undo this zoom.
            self._pan = (event.x, event.y, *self._view, *self._pan[4:])
        self._changed(ax)

    def _on_press(self, event):
        ax = self._own(event)
        if ax is None:
            return
        if event.button == 1 and event.dblclick:
            self.reset()
        elif event.button in (2, 3):
            self._pan = (
                event.x,
                event.y,
                ax.get_xlim(),
                ax.get_ylim(),
                ax.bbox.width,
                ax.bbox.height,
            )

    def _on_motion(self, event):
        if self._pan is None:
            return
        ax = self._get_ax()
        if ax is None or event.x is None:
            return
        x0, y0, xlim, ylim, w, h = self._pan
        new = []
        for lim, d_px, size, scale in (
            (xlim, event.x - x0, w, ax.get_xscale()),
            (ylim, event.y - y0, h, ax.get_yscale()),
        ):
            lo, hi = self._to(scale, lim)
            shift = d_px / size * (hi - lo)
            new.append(tuple(self._from(scale, (lo - shift, hi - shift))))
        ax.set_xlim(*new[0])
        ax.set_ylim(*new[1])
        self._view = (ax.get_xlim(), ax.get_ylim())
        self._changed(ax)

    def _on_release(self, event):
        self._pan = None
