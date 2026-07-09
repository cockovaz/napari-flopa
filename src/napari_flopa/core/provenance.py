"""Track where a parameter value came from.

Each parameter (instrument constant or scan setting) is tagged with one of:

- ``METADATA`` — read from the .ptu file header.
- ``DEFAULT``  — a fallback value (header did not provide it).
- ``USER``     — explicitly set/overridden by the user.

Letters (M/D/U) are defined here so the printed summary and the UI agree.
Colours are a UI concern and live in ``ui/style.py``.
"""

METADATA = "metadata"
DEFAULT = "default"
USER = "user"

LETTER = {METADATA: "M", DEFAULT: "D", USER: "U"}


def letter(source: str) -> str:
    """One-letter code (M/D/U) for a provenance source; '?' if unknown."""
    return LETTER.get(source, "?")
