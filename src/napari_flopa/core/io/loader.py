import numpy as np
from ptuio.decoder import T3OverflowCorrector
from ptuio.marker import get_marker_distribution, marker_events
from ptuio.reader import TTTRReader
from ptuio.utils import estimate_tcspc_bins

from napari_flopa.core.logger import ProgressLogger


def format_ptu_header(
    header_tags: dict, constants: dict, full_header: bool = False
) -> str:
    """
    Generates a formatted string summary of PTU header and constants.

    Args:
        header_tags: The dictionary of header tags from the PTU file.
        constants: The dictionary of calculated constants.
        full_header: If True, appends the entire raw header dump.

    Returns:
        A formatted, multi-line string with the summary.
    """
    lines = []

    measurement_sub_mode = header_tags.get("Measurement_SubMode")
    if measurement_sub_mode is not None and measurement_sub_mode < 1:
        lines.append("* WARNING: Not an image. Configure scanning settings.")

    lines += [
        "--- Key Parameters ---",
        f"Repetition Rate:   {constants['repetition_rate']:.2e} Hz",
        f"TCSPC Resolution:  {constants['tcspc_resolution_ns']:.2e}",
        f"Resolution Unit:   {constants['resolution_unit']}",
        f"TCSPC Bins:        {constants['tcspc_bins']}",
        f"Wrap Around:       {constants['wrap']}",
        f"Omega:             {constants['omega']:.4e} rad/s",
        "",
        "--- Image Header ---",
        f"Pixels X:          {header_tags.get('ImgHdr_PixX', 'N/A')}",
        f"Pixels Y:          {header_tags.get('ImgHdr_PixY', 'N/A')}",
        f"Frame Count:       {header_tags.get('ImgHdr_NumberOfFrames', 'N/A')}",
    ]

    if full_header:
        lines += ["", "--- Full Header ---"]
        for key, value in header_tags.items():
            lines.append(f"{key}: {value}")

    return "\n".join(lines)


def read_ptu_file(
    path, header: bool = True, logger: ProgressLogger = None
) -> dict:
    """
    Reads a PTU file and creates a standardized dictionary of instrument constants.

    Args:
        path: Path to the .ptu file.
        header: If True, includes the full raw header in the log output.
        logger: Optional ProgressLogger. Defaults to print mode.

    Returns:
        dict with keys: 'reader', 'header', 'constants'.
    """
    if logger is None:
        logger = ProgressLogger(mode="print")

    logger.log(f"Reading PTU file: {path}")
    reader = TTTRReader(path)
    header_tags = reader.header.tags

    repetition_rate = header_tags.get("TTResult_SyncRate", 40e6)
    tcspc_resolution = header_tags.get("MeasDesc_Resolution", 1 / 1e9)
    tcspc_resolution_ns = tcspc_resolution * 1e9
    resolution_unit = "ch" if tcspc_resolution_ns == 1.0 else "ns"
    tcspc_bins = estimate_tcspc_bins(header_tags, buffer=0)
    wrap = header_tags.get("TTResultFormat_WrapAround", 1024)
    omega = 2 * np.pi * repetition_rate * tcspc_resolution

    constants = {
        "repetition_rate": repetition_rate,
        "tcspc_resolution": tcspc_resolution,
        "tcspc_resolution_ns": tcspc_resolution_ns,
        "resolution_unit": resolution_unit,
        "tcspc_bins": tcspc_bins,
        "wrap": wrap,
        "omega": omega,
    }

    summary_text = format_ptu_header(
        header_tags, constants, full_header=header
    )
    logger.log(summary_text)

    return {
        "reader": reader,
        "header": header_tags,
        "constants": constants,
    }


def get_markers(reader: TTTRReader, chunk_limit: int = 0) -> dict:
    """
    Reads chunks from a PTU file and extracts the distribution of markers.

    Args:
        reader: An initialized TTTRReader for the PTU file.
        chunk_limit: Number of 1M-record chunks to read (0 = all).

    Returns:
        A dict mapping marker channel numbers to their counts,
        or {"error": "..."} if no markers found.
    """
    all_markers = []
    wrap = reader.header.tags.get("TTResultFormat_WrapAround", 1024)
    corrector = T3OverflowCorrector(wraparound=wrap)

    for i, chunk in enumerate(reader.iter_chunks(chunk_size=1_000_000)):
        if chunk_limit > 0 and i >= chunk_limit:
            break
        corrected_chunk = corrector.correct(chunk)
        all_markers.append(marker_events(corrected_chunk))

    all_markers_flat = np.concatenate(all_markers)
    if all_markers_flat.size == 0:
        return {"error": "No markers found."}

    return get_marker_distribution(all_markers_flat)


def analyze_marker_distribution(
    distribution: dict,
    verbose: bool = False,
    line_start_marker: int = 1,
    frame_start_marker: int = 4,
    max_accumulations: int = 64,
) -> dict:
    """
    Analyzes a marker distribution to suggest scan parameters.

    Args:
        distribution: Output from get_markers().
        verbose: If True, prints a formatted summary.
        line_start_marker: Marker channel for line starts.
        frame_start_marker: Marker channel for frame starts.
        max_accumulations: Max accumulations to consider in suggestions.

    Returns:
        dict with structured analysis results and suggested (lines, accumulations) pairs.
    """
    num_line_starts = distribution.get(line_start_marker, 0)
    num_frame_starts = distribution.get(frame_start_marker, 0)

    frames_guess = max(1, num_frame_starts)
    total_lines_per_frame = num_line_starts // frames_guess

    suggestion_pairs = []
    for i in range(1, max_accumulations + 1):
        if total_lines_per_frame % i == 0:
            lines = total_lines_per_frame // i
            if 64 <= lines <= 4096:
                suggestion_pairs.append((lines, i))

    analysis_results = {
        "num_line_starts": num_line_starts,
        "num_frame_starts": num_frame_starts,
        "frames_guess": frames_guess,
        "total_lines_per_frame": total_lines_per_frame,
        "suggestions": suggestion_pairs,
    }

    if verbose:
        print("--- Marker Analysis Suggestions ---")
        print(_format_marker_suggestions(analysis_results))

    return analysis_results


def _format_marker_suggestions(analysis_results: dict) -> str:
    lines = [
        f"Frame Starts: {analysis_results['num_frame_starts']} | Line Starts: {analysis_results['num_line_starts']}",
        f"For {analysis_results['frames_guess']} frame(s) ~ {analysis_results['total_lines_per_frame']} line scans per frame.",
        "",
        "Possible combinations: Lines x Accumulations",
    ]
    suggestions = analysis_results.get("suggestions", [])
    if not suggestions:
        lines.append(
            "  - Could not find common factors. Please check header or lab notes."
        )
    else:
        for lines_val, acc_val in suggestions:
            lines.append(f"  - {lines_val} x {acc_val}")
    return "\n".join(lines)
