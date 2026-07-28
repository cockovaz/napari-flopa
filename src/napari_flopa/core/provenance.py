"""Track where a parameter value came from.

Each parameter (instrument constant or scan setting) is tagged with one of:

- ``METADATA``  — read from the .ptu file header.
- ``DEFAULT``   — a fallback value (header did not provide it).
- ``USER``      — explicitly set/overridden by the user.
- ``ESTIMATED`` — computed (e.g. tcspc_bins), not read from the file

Letters (M/D/U/E) are defined here so the printed summary and the UI agree.
Colours are a UI concern and live in ``ui/style.py``.
"""

METADATA = "metadata"
DEFAULT = "default"
USER = "user"
ESTIMATED = "estimated"

LETTER = {METADATA: "M", DEFAULT: "D", USER: "U", ESTIMATED: "E"}


def letter(source: str) -> str:
    """One-letter code (M/D/U/E) for a provenance source; '?' if unknown."""
    return LETTER.get(source, "?")
