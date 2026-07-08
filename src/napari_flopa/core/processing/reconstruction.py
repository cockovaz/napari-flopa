import xarray as xr
from ptuio.decoder import T3OverflowCorrector
from ptuio.reconstructor import ImageReconstructor, ScanConfig

from napari_flopa.core.logger import ProgressLogger


def reconstruct_ptu_to_dataset(
    ptu_data: dict,
    scan_config: ScanConfig,
    outputs: list = None,
    tcspc_channels_override: int = None,
    logger: ProgressLogger = None,
) -> xr.Dataset:
    """
    Full pipeline: reconstruct from already-loaded PTU data → xarray Dataset.

    Args:
        ptu_data: Dict returned by read_ptu_file() with keys 'reader', 'header', 'constants'.
        scan_config: ScanConfig with scan parameters.
        outputs: List of outputs to compute, e.g. ['photon_count', 'mean_arrival_time'].
                 If None, all outputs are computed.
        tcspc_channels_override: Override the tcspc_bins from constants if provided.
        logger: Optional ProgressLogger.

    Returns:
        xr.Dataset with dimensions (frame, sequence, line, pixel, channel).

    Raises:
        RuntimeError: If no photons were assigned (likely wrong scan parameters).
    """
    if logger is None:
        logger = ProgressLogger(mode="print")

    reader = ptu_data["reader"]
    constants = ptu_data["constants"]

    tcspc_channels = tcspc_channels_override or constants["tcspc_bins"]

    if outputs is None or outputs == ["all"]:
        requested_outputs = None  # ImageReconstructor default = all
    elif outputs == ["mean_arrival_time"]:
        requested_outputs = ["photon_count", "mean_arrival_time"]
    else:
        requested_outputs = outputs

    recon = ImageReconstructor(
        config=scan_config,
        outputs=requested_outputs,
        omega=constants["omega"],
        tcspc_channels=tcspc_channels,
    )

    corrector = T3OverflowCorrector(wraparound=constants["wrap"])

    for chunk_num, chunk in enumerate(
        reader.iter_chunks(chunk_size=1_000_000), start=1
    ):
        corrected = corrector.correct(chunk)
        recon.update(corrected)
        logger.log(f"Processed chunk {chunk_num}...")
        if recon._finished:
            break

    if not recon.active_detectors:
        raise RuntimeError(
            "No photons were assigned to any image segment.\n"
            "Check scan parameters: marker channels, lines, pixels, frames.\n"
            "Use 'Analyze Markers' to inspect the file's marker distribution."
        )

    logger.log("Finalizing dataset...")
    ds = recon.finalize()
    return ds
