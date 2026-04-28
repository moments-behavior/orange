# <img src="icon.png" alt="" height="48"> orange capture

A high-performance, GPU-accelerated multi-camera capture, streaming and recording application for Emergent GigE Vision cameras, built in C++.

![gui](images/gui.png)

## Overview

`orange` is built for high-throughput, time-synchronized multi-camera recording. Encoding is GPU-accelerated and scales with the number of GPUs in the host. PTP keeps cameras aligned to sub-frame precision, and a multi-host architecture (one GUI host coordinating any number of headless `cam_server` nodes over ENet) lets a recording rig scale beyond what a single machine can drive — both in camera count and aggregate pixel rate. Optional TensorRT-based YOLO detection runs on the live streams when a model is provided.

## Documentation

Full documentation — installation, system requirements, configuration, network mode, real-time detection, PTP — lives at the [moments-behavior docs site](https://moments-behavior.github.io/docs/orange/).

[Video demo](https://youtu.be/ahceluqBYj8)

## Quick build

Linux-only. Requires an NVIDIA GPU with NVENC. See the docs for full system requirements and the dependency install walkthrough.

```bash
git clone --recursive https://github.com/moments-behavior/orange.git
cd orange
./build.sh    # builds release/orange, release/cam_server, release/yolo_offline
./run.sh      # sudo release/orange
```

## Authors

**Orange** is developed by Jinyao Yan, with contributions from Diptodip Deb, Wilson Chen, Ratan Othayoth, Jeremy Delahanty, and Rob Johnson.

Contact [Jinyao Yan](mailto:yanj11@janelia.hhmi.org) with questions about the software.

## Citation

If you use **Orange**, please cite the software:

```bibtex
@software{moments_behavior_orange_2026,
  author       = {Yan, Jinyao and
                  Deb, Diptodip and
                  Chen, Wilson and
                  Othayoth, Ratan and
                  Delahanty, Jeremy and
                  Johnson, Rob},
  title        = {moments-behavior/orange: v2.1.0},
  month        = apr,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {v2.1.0},
  doi          = {10.5281/zenodo.19688150},
  url          = {https://doi.org/10.5281/zenodo.19688150},
}
```

## Contribute

Please open an issue for bug fixes or feature requests. If you wish to make changes to the source code, fork the repo and open a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request).
