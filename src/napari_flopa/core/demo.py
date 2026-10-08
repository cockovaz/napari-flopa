"""Locate and load the bundled demo dataset.

Resolution order for the demo params file (first match wins):

1. ``$NAPARI_FLOPA_DEMO`` — absolute path to a params ``.json``. For pointing at
   an arbitrary local file.
2. ``data/ptu_demo_local/demo_params.local.json`` — a git-ignored override in
   the repo-root ``data`` folder, where large unshipped scans are kept. Only
   present in a dev checkout.
3. ``demo_data/demo_params.local.json`` — the same override inside the package.
4. ``demo_data/demo_params.json`` — the small demo shipped with the package.

The ``ptu_filename`` in the params file is resolved relative to that file's own
directory (unless it is absolute), so an override normally just names its
sibling ``.ptu``.
"""

import json
import os
from pathlib import Path

# demo_data lives at the package root; this module is one level down (core/).
_DEMO_DIR = Path(__file__).parent.parent / "demo_data"

#: Where a dev checkout keeps demo data too large to ship. Outside the package,
#: so this resolves to nothing once flopa is installed from a wheel.
_LOCAL_DEMO_DIR = Path(__file__).parents[3] / "data" / "ptu_demo_local"


def demo_params_path() -> Path | None:
    """Return the demo params ``.json`` to use, or ``None`` if none exists."""
    env = os.environ.get("NAPARI_FLOPA_DEMO")
    if env and Path(env).exists():
        return Path(env)
    candidates = (
        _LOCAL_DEMO_DIR / "demo_params.local.json",
        _DEMO_DIR / "demo_params.local.json",
        _DEMO_DIR / "demo_params.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
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
