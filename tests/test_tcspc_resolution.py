import numpy as np
import pytest
import xarray as xr

from napari_flopa.core import demo
from napari_flopa.core.processing.phasor import (
    irf_delay_from_calibration,
    monoexp_phasor,
)

requires_demo = pytest.mark.skipif(
    demo.demo_params_path() is None,
    reason="no demo data available (see src/napari_flopa/core/demo.py)",
)

F_REP = 40e6


def _tiny_ds(res_s: float, header_res_s: float | None = None) -> xr.Dataset:
    """0-d dataset carrying what the reconstruction stores about omega."""
    header_res_s = res_s if header_res_s is None else header_res_s
    ds = xr.Dataset(
        {
            "omega": ((), 2 * np.pi * F_REP * res_s),
            "tcspc_resolution": ((), res_s),
        }
    )
    ds.attrs["instrument_params"] = {
        "repetition_rate": F_REP,
        "tcspc_resolution_ns": header_res_s * 1e9,
        "resolution_unit": "ns",
    }
    return ds


def _monoexp_dtimes(
    tau_s: float, res_s: float, n: int = 200_000, seed: int = 0
) -> np.ndarray:
    """TCSPC channels of photons from a decay repeating every laser period."""
    rng = np.random.default_rng(seed)
    t = np.mod(rng.exponential(tau_s, n), 1 / F_REP)
    return np.floor(t / res_s)


@pytest.mark.parametrize("res", [1e-12, 4e-12, 5e-12, 16e-12, 25e-12])
def test_reconstruction_passes_header_resolution(monkeypatch, res):
    """The header resolution reaches tttrkit (regression for the 5 ps bug)."""
    from napari_flopa.core.io.config import ScanSettings
    from napari_flopa.core.processing import reconstruction

    class _Stop(Exception):
        pass

    seen = {}

    def _spy(**kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(reconstruction, "ImageReconstructor", _spy)
    ptu_data = {
        "reader": None,
        "constants": {
            "repetition_rate": F_REP,
            "tcspc_resolution": res,
            "tcspc_bins": 1600,
            "wrap": 1024,
        },
    }
    with pytest.raises(_Stop):
        reconstruction.reconstruct_ptu_to_dataset(
            ptu_data, ScanSettings().to_scan_config(F_REP)
        )
    assert seen["tcspc_resolution"] == res
    assert seen["laser_sync_rate"] == F_REP


@pytest.mark.parametrize("res", [1e-12, 4e-12, 16e-12])
def test_omega_puts_monoexp_on_semicircle(res):
    """With omega from the true resolution, a decay lands on its τ point."""
    tau = 2.5e-9
    dtime = _monoexp_dtimes(tau, res)
    phasor = np.mean(np.exp(1j * 2 * np.pi * F_REP * res * dtime))
    g, s = monoexp_phasor(tau, F_REP)
    assert phasor.real == pytest.approx(g, abs=1e-2)
    assert phasor.imag == pytest.approx(s, abs=1e-2)


def test_assumed_5ps_misplaces_16ps_data():
    """What the bug did: a 16 ps file with a 5 ps omega is off the τ point."""
    tau, res = 2.5e-9, 16e-12
    dtime = _monoexp_dtimes(tau, res)
    phasor = np.mean(np.exp(1j * 2 * np.pi * F_REP * 5e-12 * dtime))
    g, s = monoexp_phasor(tau, F_REP)
    assert abs(phasor - complex(g, s)) > 0.1


@pytest.mark.parametrize("res", [4e-12, 5e-12, 16e-12])
def test_irf_delay_independent_of_resolution(res):
    """The IRF delay from a factor is in ns whatever the channel width."""
    delay_ns = 1.2
    factor = np.exp(-1j * 2 * np.pi * F_REP * delay_ns * 1e-9)
    out = irf_delay_from_calibration(_tiny_ds(res), complex(factor))
    assert out is not None
    got_delay, period = out
    assert got_delay == pytest.approx(delay_ns, rel=1e-9)
    assert period == pytest.approx(1e9 / F_REP)


def _reconstruct_demo(outputs=None, resolution=None):
    from napari_flopa.core.io.config import ScanSettings
    from napari_flopa.core.io.loader import read_ptu_file
    from napari_flopa.core.processing.reconstruction import (
        reconstruct_ptu_to_dataset,
    )

    params, ptu = demo.load_demo()
    data = read_ptu_file(str(ptu), header=False)
    constants = data["constants"]
    if resolution is not None:
        constants["tcspc_resolution"] = resolution
    settings = ScanSettings.from_json_dict(params)
    ds = reconstruct_ptu_to_dataset(
        data,
        settings.to_scan_config(constants["repetition_rate"]),
        outputs=outputs,
        tcspc_channels_override=settings.tcspc_bins,
    )
    return ds, constants


@requires_demo
def test_demo_dataset_omega_matches_header():
    """The dataset records the header resolution and its omega."""
    ds, c = _reconstruct_demo()
    assert float(ds["tcspc_resolution"]) == pytest.approx(
        c["tcspc_resolution"]
    )
    assert float(ds["omega"]) == pytest.approx(c["omega"])
    assert c["omega"] == pytest.approx(
        2 * np.pi * c["repetition_rate"] * c["tcspc_resolution"]
    )


@requires_demo
def test_demo_omega_scales_with_resolution():
    """End to end: doubling the resolution doubles the phasor omega."""
    outputs = ["photon_count", "phasor"]
    ds, c = _reconstruct_demo(outputs)
    ds2, _ = _reconstruct_demo(outputs, resolution=2 * c["tcspc_resolution"])
    assert float(ds2["omega"]) == pytest.approx(2 * float(ds["omega"]))


def test_phasor_cal_info_warns_on_mismatch(qtbot):
    """The Calibration box shows the dataset values and flags a stale omega."""
    from types import SimpleNamespace

    from napari_flopa.ui.panels.phasor_panel import PhasorPanel
    from napari_flopa.ui.state import FlopaState

    class _Signal:
        def connect(self, _fn):
            pass

    class _Layers(list):
        events = SimpleNamespace(inserted=_Signal(), removed=_Signal())

    panel = PhasorPanel(FlopaState(), SimpleNamespace(layers=_Layers()))
    qtbot.addWidget(panel)

    panel._update_cal_info(_tiny_ds(5e-12))
    text = panel._cal_info.text()
    assert "40.000 MHz" in text
    assert "5.0 ps" in text
    assert "Reconstruct again" not in text

    panel._update_cal_info(_tiny_ds(5e-12, header_res_s=16e-12))
    assert "Reconstruct again" in panel._cal_info.text()
