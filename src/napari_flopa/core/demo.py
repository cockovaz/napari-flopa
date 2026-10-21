import json
import os
from pathlib import Path

_DEMO_DIR = Path(__file__).parent.parent / "demo_data"
_LOCAL_DEMO_DIR = Path(__file__).parents[3] / "data" / "ptu_demo_local"


def demo_params_path() -> Path | None:
    """Return the demo params ``.json`` to use, or ``None`` if none exists.

    First match wins:

    1. ``$NAPARI_FLOPA_DEMO`` — path to any params ``.json``; only when the
       variable is set before napari / pytest starts.
    2. ``data/ptu_demo_local/demo_params.local.json`` — git-ignored override
       in the repo-root ``data`` folder (for development purpose).
    3. ``demo_data/demo_params.json`` — the small demo shipped with the
       package.
    """
    env = os.environ.get("NAPARI_FLOPA_DEMO")
    if env and Path(env).exists():
        return Path(env)
    candidates = (
        _LOCAL_DEMO_DIR / "demo_params.local.json",
        _DEMO_DIR / "demo_params.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_demo() -> tuple[dict, Path]:
    """Return ``(params, ptu_path)`` for the demo dataset.

    A relative ``ptu_filename`` is resolved against the params file.

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
