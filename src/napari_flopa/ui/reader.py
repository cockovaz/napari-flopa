import json
from pathlib import Path

import napari
from napari.utils.notifications import show_warning

#: What napari expects back when a read succeeded but adds no layers. Neither
#: a .ptu nor a config has anything to show — the File tab is filled in and
#: this is returned instead of layer data.
_NO_LAYERS = [(None,)]

#: A scan config is a few KB; refuse to parse something far larger just to
#: find out whether it is one.
_MAX_CONFIG_BYTES = 1_000_000


def _as_paths(path) -> list[Path]:
    """napari passes one path, or a list when files are dropped as a stack."""
    return [Path(p) for p in (path if isinstance(path, list) else [path])]


def _is_scan_config(path: Path) -> bool:
    """True for a flopa scan config, so other JSON is left alone."""
    try:
        if path.stat().st_size > _MAX_CONFIG_BYTES:
            return False
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    # The one key _apply_config reads.
    return isinstance(cfg, dict) and isinstance(cfg.get("scan"), dict)


def napari_get_reader(path):
    """Claim .ptu files, and .json files that are flopa scan configs."""
    paths = _as_paths(path)
    if not paths:
        return None
    suffix = paths[0].suffix.lower()
    if suffix == ".ptu":
        return read_ptu
    if suffix == ".json" and _is_scan_config(paths[0]):
        return read_config
    return None


def _target_widget(paths: list[Path]):
    """The live FLIM widget, or None when there is no viewer to put it in."""
    if len(paths) > 1:
        show_warning(
            f"{len(paths) - 1} more file(s) ignored — the File tab takes one "
            "at a time. Use the Batch tab for several files."
        )
    viewer = napari.current_viewer()
    if viewer is None:
        show_warning("No napari viewer available to load into.")
        return None
    # Returns the running widget when the plugin is already open, and opens
    # it first when it is not.
    _dock, widget = viewer.window.add_plugin_dock_widget("napari-flopa")
    return widget


def read_ptu(path, stack: bool = False):
    """Load the dropped .ptu into the File tab, adding no layers."""
    paths = _as_paths(path)
    widget = _target_widget(paths)
    if widget is not None:
        widget.load_ptu(paths[0])
    return _NO_LAYERS


def read_config(path, stack: bool = False):
    """Apply the dropped scan config to the File tab, adding no layers."""
    paths = _as_paths(path)
    widget = _target_widget(paths)
    if widget is not None:
        widget.load_config(paths[0])
    return _NO_LAYERS
