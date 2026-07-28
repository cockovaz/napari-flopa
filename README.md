# napari-flopa

[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue)](https://napari.org/stable/plugins/index.html)

<!-- Enable these badges once the package is published to PyPI:
[![PyPI](https://img.shields.io/pypi/v/napari-flopa.svg?color=green)](https://pypi.org/project/napari-flopa)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-flopa.svg?color=green)](https://python.org)
-->


> **Work in progress** — the plugin is functional but under active development. Expect breaking changes between versions.

A [napari] plugin for opening, processing and analysing FLIM (Fluorescence Lifetime Imaging Microscopy) data from `.ptu` files.

## Features

- **Process PTU** — reconstruct `.ptu` files into xarray datasets (photon count, mean arrival time, phasor, TCSPC histogram); supports multi-frame, multi-sequence and multi-detector data
- **FLIM View** — interactive display with histogram contrast sliders for intensity and lifetime; FLIM RGB composite; export to TIFF/PNG
- **Phasor** — phasor plot with calibration, smoothing, per-object or per-pixel scatter, monoexponential lifetime semi-circle overlay
- **Decay** — TCSPC decay plot with aggregation, normalisation and log scale
- **Batch** — **NOW DISABLED** process a folder of `.ptu` files with a shared scan config and export images, phasor tables and decay tables; config saved/loaded as json

## Installation

> ⚠️ Not yet released on PyPI. Install from source:

```
git clone https://github.com/cockovaz/napari-flopa
cd napari-flopa
pip install -e ".[all]"
```

<!-- Once released on PyPI, installation will be:

    pip install napari-flopa           # plugin only
    pip install "napari-flopa[all]"    # with napari + Qt included
-->


## License

Distributed under the terms of the [MIT] license.
`napari-flopa` is free and open source software.

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[MIT]: http://opensource.org/licenses/MIT
[file an issue]: https://github.com/cockovaz/napari-flopa/issues
[pip]: https://pypi.org/project/pip/
