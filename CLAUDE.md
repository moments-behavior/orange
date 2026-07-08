# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`orange` is a high-performance multi-camera capture, streaming, and recording application for Emergent Vision GigE cameras. Linux-only, requires an NVIDIA GPU with NVENC. Encoding is GPU-accelerated (CUDA + NVENC), cameras are time-synchronized via IEEE-1588 PTP, and a multi-host architecture lets one GUI host coordinate any number of headless `cam_server` nodes over ENet. Optional TensorRT YOLOv8 detection runs on live streams.

Full docs: https://moments-behavior.github.io/docs/orange/

## Build & run

```bash
./build.sh          # cmake + build all targets into release/
./run.sh            # sudo release/orange (GUI needs root for the Emergent SDK)
./server_build.sh   # build only cam_server (BUILD_ORANGE=OFF, BUILD_YOLO_OFFLINE=OFF)
./start_server.sh   # run cam_server, named from $CAM_SERVER_NAME or hostname
./install.sh        # install GUI binary + fonts + desktop entry into ~/.local
```

Manual build:

```bash
cmake -S . -B release -DCMAKE_BUILD_TYPE=Release
cmake --build release -j
```

- CMake options `BUILD_ORANGE`, `BUILD_CAM_SERVER`, `BUILD_YOLO_OFFLINE` (all ON by default) select the three executables.
- CUDA architectures default to `75;80`; override with `-DCMAKE_CUDA_ARCHITECTURES=<arch>`.
- Hard-coded dependency paths: Emergent SDK at `/opt/EVT/eSDK/`, FFmpeg at `~/nvidia/ffmpeg`, TensorRT at `~/nvidia/TensorRT`. Clone with `--recursive` (submodules under `third_party/`: imgui, implot, flatbuffers, ImGuiFileDialog, etc.).
- There is no test suite or linter.
- `src/ctrl_generated.h` is checked in, generated from `schema/ctrl.fbs` with `flatc --cpp`. Regenerate it manually when the schema changes; there is no build-step integration.

## The three executables

All three share the `src/` tree; per-target sources are selected by exclusion lists in `CMakeLists.txt`:

- **`orange`** — GUI host (`src/orange.cpp`). ImGui/GLFW/OpenGL + ImPlot. Drives locally-attached cameras directly and/or orchestrates remote `cam_server` nodes as the ENet client.
- **`cam_server`** — headless capture/record node (`src/cam_server.cpp`), compiled with `-DHEADLESS` which strips GUI/display/detection code (e.g. blocks in `video_capture.cpp` are `#ifndef HEADLESS`). Runs as ENet server, default port 34001. Usage: `cam_server <name> [port]`.
- **`yolo_offline`** — standalone offline YOLOv8/TensorRT detector for recorded videos (`src/yolo_offline.cpp`). Usage: `yolo_offline <engine> <video> <gpu_id>`.

When adding a new `.cpp` to `src/`, check the `list(REMOVE_ITEM ...)` exclusion lists in `CMakeLists.txt` — sources are globbed, so a GUI-only or detection-only file will otherwise break the `cam_server` (HEADLESS) build.

## Architecture

### Frame pipeline (per camera)

One acquisition thread per camera (`acquire_frames` in `src/video_capture.cpp`), pinned to its configured GPU. Each frame from the Emergent SDK fans out to independent worker threads:

1. **Encode/record**: `GPUVideoEncoder` (`src/gpu_video_encoder.cpp`, a `CThreadWorker`) — GPU debayer (`image_processing.h`, `kernel.cu`) → NVENC encode (`src/NvEncoder/`, `nvenc_api/`) → `FFmpegWriter` muxes to `Cam<serial>.mp4` plus `_meta.csv` (frame_id, timestamps, ptp_offset) and `_keyframe.csv`.
2. **Display** (non-HEADLESS): `COpenGLDisplay` via CUDA-GL interop.
3. **Detection** (non-HEADLESS): `FrameDetector` per camera runs a TensorRT `YOLOv8` (`src/yolov8_det.cpp`), writes 2D detections into the global `detection2d[]`; `detection3d_proc` (`src/detect3d.cpp`) waits on `mtx3d`/`cv3d` until all 3D-mode cameras have fresh detections, triangulates (`src/realtime_tool.cpp`), and optionally streams 3D targets over UDP to an external galvo motor controller (`src/galvo_sender.cpp`, packed `GCT1` packets).
4. **Still capture**: `FrameSaver` (calibration images / snapshots) via `cv::imwrite`.

Inter-thread handoff uses atomic `PictureState` state machines and condition variables rather than a single queue framework. Threading primitives live in `src/offthreadmachine.*` (`COffThreadMachine`) and `src/threadworker.*` (`CThreadWorker` = machine + locked in/out queues). Cross-thread shared state (fps counters, detection results, 3D state, galvo params, `mtx3d`/`cv3d`) is `extern` globals in `src/global.h`/`global.cpp` — the shared-memory bus between acquisition, detection, 3D, and galvo threads.

### Network protocol (host ⇄ cam_server)

- Transport: ENet, wrapped by `EnetRuntime` (`src/enet_runtime_threaded.*`) — a dedicated I/O thread exchanging with the app through `TSQueue`s (`src/enet_types.h`). `src/enet_fb_helpers.h` has `PeerRegistry` and `FBMessageSender`; `src/enet_utils.*` bundles them into `AppContext`.
- Messages: FlatBuffers, schema `schema/ctrl.fbs` (namespace `camnet::v1`). Envelope table `Server` carries a `ServerControl` command (OPENCAMERA, STARTTHREAD, STARTRECORDING, STOPRECORDING, STARTSTREAMING, TAKEPICTURE, ...), idempotency fields (`job_id`/`epoch`/`seq`), a `CommandBody` union, or a `ReplyInfo`.
- Host side (`src/host_client_imgui.cpp`) runs a phase state machine (Phase_Open → Phase_Threads → Phase_Start → Phase_Stop → Phase_Done, plus calibration sub-phases): it broadcasts a command to all server peers, collects acks, and only then advances. cam_server dispatches commands in `ctrl_action` (`src/cam_server.cpp`) and deduplicates retries via job/epoch/seq. On connect, cam_server sends a bringup reply with its name and camera count.
- Note: several `ServerControl` calibration/trigger commands (NEXTPOSE, TRIALTRIGGER, board/ball variants) are stubs in `cam_server.cpp`; the board/ball calibration flow runs through the local GUI path.

### PTP synchronization

Recording start/stop across cameras and hosts is gated on a shared PTP nanosecond timestamp: `start_ptp_sync` (`src/video_capture.cpp`) programs `PtpAcquisitionGateTimeHigh/Low` on all cameras for a common future start; camera threads barrier via atomic counters. In network mode, the host broadcasts the global `ptp_time` inside `StartArgs`/`StopArgs`. Camera-side PTP helpers are in `src/camera.cpp`.

### Configuration

- Per-camera: `config/<serial>.json` (see `config/2002496.json.example`) — resolution, frame_rate, gain, exposure, pixel_format, gpu_id, color, gpu_direct, etc. Loaded with nlohmann/json in `video_capture.cpp`; cameras without a config file fall back to model-specific defaults.
- Network: `config/network/endpoints.json` (`{ default_port, servers:[{name, host, port?}] }`), loaded by `src/server_endpoints.cpp`.
