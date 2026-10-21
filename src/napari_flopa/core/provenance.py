#: Where a parameter value (instrument constant or scan setting) came from.
METADATA = "metadata"  # read from the .ptu file header
DEFAULT = "default"  # fallback, the header did not provide it
USER = "user"  # set or overridden by the user
ESTIMATED = "estimated"  # computed (e.g. tcspc_bins), not read from the file

LETTER = {METADATA: "M", DEFAULT: "D", USER: "U", ESTIMATED: "E"}


def letter(source: str) -> str:
    """One-letter code (M/D/U/E) for a provenance source; '?' if unknown.

    Defined here so the printed summary and the UI agree; the colours live in
    ``ui/style.py``.
    """
    return LETTER.get(source, "?")
