import contextlib
from collections.abc import Callable

import numpy as np
from matplotlib.path import Path as MplPath


def napari_label_color(layer, label_id: int) -> tuple:
    """Return (R, G, B, A) float 0-1 for label_id from a napari Labels layer."""
    try:
        rgba = layer.get_color(label_id)
        return (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))
    except Exception:
        return (0.5, 0.5, 0.5, 0.8)


class PhasorRoiSelector:
    """Lasso ROI on the phasor plot, painted into a napari Labels layer.

    Each closed lasso gets the next label id: the points inside it are
    recoloured on the plot, and their pixels (per pixel) or objects (per
    object) are painted into the "Phasor ROI Labels" layer.

    *get_context* is called on every lasso release and returns the panel
    state the painting depends on: ``per_object`` (bool), ``object_labels``
    (object id per plotted point, or None), ``mask_layer_name`` (str) and
    ``dataset`` (xr.Dataset or None).
    """

    def __init__(self, viewer, canvas, get_context: Callable[[], dict]):
        self.viewer = viewer
        self._canvas = canvas
        self._get_context = get_context
        self.active = False
        self._ax = None
        self._label_id = 0
        self._lasso_verts = []
        self._line = None
        self._bg = None  # blitting background
        self._labels_layer = None
        self._label_data = None
        self._point_labels = None
        self._mpl_cids = []
        self._g = None
        self._s = None
        self._scatter = None
        self._pixel_yx = None

    def set_points(self, ax, g, s, pixel_yx=None) -> None:
        """Points of a fresh plot; *pixel_yx* = (ys, xs) per point, per pixel."""
        self._ax = ax
        self._g = np.asarray(g, dtype=np.float32).ravel()
        self._s = np.asarray(s, dtype=np.float32).ravel()
        self._pixel_yx = pixel_yx
        self._scatter = None

    def start(self) -> None:
        """Connect the matplotlib mouse events."""
        self.active = True
        canvas = self._canvas
        self._mpl_cids = [
            canvas.mpl_connect("button_press_event", self._on_press),
            canvas.mpl_connect("motion_notify_event", self._on_motion),
            canvas.mpl_connect("button_release_event", self._on_release),
        ]

    def stop(self) -> None:
        """Disconnect the mouse events and remove the lasso line."""
        for cid in self._mpl_cids:
            self._canvas.mpl_disconnect(cid)
        self._mpl_cids = []
        self._bg = None
        if self._line is not None:
            with contextlib.suppress(Exception):
                self._line.remove()
            self._line = None
        self._lasso_verts = []
        self._canvas.draw_idle()

    def finish(self) -> None:
        """End the session; the next one starts a new Labels layer."""
        self.stop()
        self.active = False
        self._label_id = 0
        self._point_labels = None
        self._labels_layer = None
        self._label_data = None

    def _on_press(self, event):
        if self._ax is None or event.inaxes != self._ax or event.button != 1:
            return
        self._lasso_verts = [(event.xdata, event.ydata)]
        if self._line is not None:
            with contextlib.suppress(Exception):
                self._line.remove()
        (self._line,) = self._ax.plot(
            [event.xdata],
            [event.ydata],
            color="yellow",
            linewidth=1.0,
            alpha=0.8,
            zorder=20,
            animated=True,
        )
        # Capture static background for blitting (excludes animated artists)
        self._canvas.draw()
        self._bg = self._canvas.copy_from_bbox(self._ax.bbox)

    def _on_motion(self, event):
        if not self._lasso_verts or event.inaxes != self._ax:
            return
        self._lasso_verts.append((event.xdata, event.ydata))
        xs = [v[0] for v in self._lasso_verts]
        ys = [v[1] for v in self._lasso_verts]
        self._line.set_data(xs, ys)
        self._canvas.restore_region(self._bg)
        self._ax.draw_artist(self._line)
        self._canvas.blit(self._ax.bbox)

    def _on_release(self, event):
        if event.button != 1:
            return
        if len(self._lasso_verts) < 3:
            self._lasso_verts = []
            self._bg = None
            return
        verts = self._lasso_verts + [self._lasso_verts[0]]
        self._lasso_verts = []
        self._bg = None
        # Non-animated again, so the closed lasso stays in the normal render
        self._line.set_animated(False)
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        self._line.set_data(xs, ys)

        if self._g is None or len(self._g) == 0:
            return

        inside = MplPath(verts).contains_points(
            np.column_stack([self._g, self._s])
        )
        if not inside.any():
            return

        self._label_id += 1
        lbl = self._label_id
        if self._point_labels is None:
            self._point_labels = np.zeros(len(self._g), dtype=np.int32)
        # Later lassos overwrite earlier assignments
        self._point_labels[inside] = lbl

        context = self._get_context()
        # Labels layer first, so its colours exist for the scatter
        self._update_labels_layer(inside, lbl, context)
        self._update_scatter_colors(context["per_object"])

    def _update_scatter_colors(self, per_obj: bool):
        """Recolour points: assigned → napari label colour, others → grey."""
        if self._point_labels is None or self._labels_layer is None:
            return
        if self._scatter is not None:
            with contextlib.suppress(Exception):
                self._scatter.remove()
        n = len(self._g)
        colors = np.zeros((n, 4), dtype=np.float32)
        colors[:, :3] = 0.35
        colors[:, 3] = 0.5
        for lbl_id in range(1, self._label_id + 1):
            mask = self._point_labels == lbl_id
            if mask.any():
                colors[mask] = napari_label_color(self._labels_layer, lbl_id)
                colors[mask, 3] = 0.85
        self._scatter = self._ax.scatter(
            self._g,
            self._s,
            c=colors,
            s=50 if per_obj else 3,
            edgecolors="white" if per_obj else "none",
            linewidths=0.5 if per_obj else 0,
            zorder=15,
        )
        self._canvas.draw_idle()

    def _update_labels_layer(self, inside_mask: np.ndarray, lbl: int, context):
        """Paint the pixels behind the selected points into the Labels layer.

        Per object: the selected points are object centroids, so every pixel
        of those objects in the mask layer is painted (the whole image when
        there is no mask). Per pixel: the stored pixel coordinates are painted.
        """
        ds = context["dataset"]
        if ds is None:
            return

        shape_2d = None
        for var in ("photon_count", "phasor_g", "mean_arrival_time"):
            if var in ds:
                vals = ds[var].values
                if vals.ndim >= 2:
                    shape_2d = vals.shape[-2:]
                break
        if shape_2d is None:
            return

        if (
            self._labels_layer is None
            or self._labels_layer not in self.viewer.layers
        ):
            self._label_data = np.zeros(shape_2d, dtype=np.int32)
            try:
                self._labels_layer = self.viewer.add_labels(
                    self._label_data.copy(), name="Phasor ROI Labels"
                )
            except Exception:
                return

        if context["per_object"]:
            obj_labels = context["object_labels"]
            if obj_labels is None:
                return
            selected_obj_ids = np.asarray(obj_labels).ravel()[inside_mask]
            mask_name = context["mask_layer_name"]
            if mask_name == "None" or mask_name not in self.viewer.layers:
                self._label_data[:] = lbl
            else:
                ldata = np.asarray(self.viewer.layers[mask_name].data)
                if ldata.ndim > 2:
                    ldata = ldata.reshape(
                        -1, ldata.shape[-2], ldata.shape[-1]
                    )[0]
                if ldata.shape != shape_2d:
                    return
                label_2d = ldata.astype(np.int32)
                for oid in selected_obj_ids:
                    if np.isfinite(oid):
                        self._label_data[label_2d == int(oid)] = lbl
        elif self._pixel_yx is not None:
            ys, xs = self._pixel_yx
            in_ys = ys[inside_mask]
            in_xs = xs[inside_mask]
            valid = (
                (in_ys >= 0)
                & (in_ys < shape_2d[0])
                & (in_xs >= 0)
                & (in_xs < shape_2d[1])
            )
            self._label_data[in_ys[valid], in_xs[valid]] = lbl

        self._labels_layer.data = self._label_data.copy()
