"""Tests exercising the bundled demo dataset end-to-end.

These skip automatically when no demo data is available (e.g. on a checkout
where the shipped demo is absent), so they are safe on CI. Locally, a
``demo_params.local.json`` override lets you run them against a larger file.
"""

import pytest

from napari_flopa import demo

requires_demo = pytest.mark.skipif(
    demo.demo_params_path() is None,
    reason="no demo data available (see src/napari_flopa/demo.py)",
)


def _scan_config(params):
    from napari_flopa.io.ptuio.reconstructor import ScanConfig

    return ScanConfig(
        lines=params["lines"],
        pixels=params["pixels"],
        frames=params["frames"],
        line_accumulations=tuple(params["accumulations"]),
        max_detector=params["max_detector"],
        bidirectional=params.get("bidirectional", False),
        bidirectional_phase_shift=params.get("bidirectional_phase_shift", 0.0),
        frame_start_marker_channel=(4,),
        line_start_marker_channel=(1,),
        line_stop_marker_channel=(2,),
    )


@requires_demo
def test_demo_reconstructs():
    """Core pipeline: read + reconstruct the demo .ptu into a dataset."""
    from napari_flopa.io.loader import read_ptu_file
    from napari_flopa.processing.reconstruction import (
        reconstruct_ptu_to_dataset,
    )

    params, ptu = demo.load_demo()
    data = read_ptu_file(str(ptu), header=False)
    ds = reconstruct_ptu_to_dataset(
        data, _scan_config(params), outputs=["photon_count"]
    )

    assert "photon_count" in ds
    assert ds["photon_count"].sizes["pixel"] == params["pixels"]
    assert ds["photon_count"].sizes["line"] == params["lines"]


@requires_demo
def test_load_demo_button(make_napari_viewer):
    """Widget smoke test: the Load Demo button populates state without error."""
    from napari_flopa.widgets.main_widget import FlimWidget

    viewer = make_napari_viewer()
    widget = FlimWidget(viewer)
    widget._ptu_panel._on_load_demo()

    assert widget._ptu_panel.ptu_data is not None
