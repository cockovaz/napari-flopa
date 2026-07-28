"""Locate and load the bundled demo dataset.

Resolution order for the demo params file (first match wins):

1. ``$NAPARI_FLOPA_DEMO`` — absolute path to a params ``.json``. For pointing at
   an arbitrary local file.
2. ``demo_data/demo_params.local.json`` — a git-ignored local override.
3. ``demo_data/demo_params.json`` — the small demo shipped with the package.

The ``ptu_filename`` in the params file is resolved relative to that file's own
directory (unless it is absolute), so a local override can reference e.g.
``"../data/big_scan.ptu"``.
"""

import json
import os
from pathlib import Path

# demo_data lives at the package root; this module is one level down (core/).
_DEMO_DIR = Path(__file__).parent.parent / "demo_data"


def demo_params_path() -> Path | None:
    """Return the demo params ``.json`` to use, or ``None`` if none exists."""
    env = os.environ.get("NAPARI_FLOPA_DEMO")
    if env and Path(env).exists():
        return Path(env)
    for name in ("demo_params.local.json", "demo_params.json"):
        candidate = _DEMO_DIR / name
        if candidate.exists():
            return candidate  # first one that exist
    return None


def load_demo() -> tuple[dict, Path]:
    """Return ``(params, ptu_path)`` for the demo dataset.

    Raises:
        FileNotFoundError: if no params file or the referenced ``.ptu`` is found.
    """
    params_path = demo_params_path()
    if params_path is None:
        raise FileNotFoundError(
            f"No demo params found. Expected {_DEMO_DIR / 'demo_params.json'}."
        )
    params = json.loads(params_path.read_text())
    ptu_path = Path(params["ptu_filename"])
    if not ptu_path.is_absolute():
        ptu_path = (params_path.parent / ptu_path).resolve()
    if not ptu_path.exists():
        raise FileNotFoundError(f"Demo PTU not found: {ptu_path}")
    return params, ptu_path
