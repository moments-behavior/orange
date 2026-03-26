# Remote Camera Focus Control & Stream Mode

## Overview

Added the ability to remotely adjust focus on any camera (dosa0, dosa1, dosa-live) from the orange GUI, with a live preview image. Also added a "Stream" mode that activates cameras and IR lights without saving video to disk.

## Usage

### Stream Mode (Focus Adjustment)

1. **Open Cameras** -> **Start Camera Threads** (as usual)
2. Click **"Start Stream (no save)"** (blue button) — PTP gate fires, IR lights turn on, cameras capture frames, but nothing is saved to disk
3. In the **"Remote Camera Focus"** section, check the box next to a camera serial
4. Drag the **Focus slider** to a new value
5. Click **"Set Focus & Preview"** — the focus is applied and a preview frame appears in the **"Remote Camera Preview"** floating window
6. Repeat: adjust slider, click preview, until the image looks sharp
7. Click **"Stop Recording"** to end the stream
8. The focus value persists on the camera hardware — the next real recording uses the corrected focus

### Recording Mode

- Click **"Start Recording"** instead — same as before, video is saved to disk
- The Remote Focus panel is **hidden** during recording to avoid accidental changes

### Test Focus (Automated Sweep)

- Click **"Test Focus"** in the WAITSTART panel — each camera runs an automated focus sweep (+-50 around current value, step 5)
- Measures sharpness via Laplacian variance at each position
- Sets the best focus automatically
- Only useful if the scene has texture (does not work on an empty table)

## Architecture

### FlatBuffers Schema Changes (`schema/fetch.fbs`)

New enum values:
- `ServerControl_SETFOCUS` (7) — set focus on a specific camera
- `ServerControl_STARTSTREAM` (8) — start PTP acquisition without video saving

New fields on `Server` table:
- `focus_value:int` — target focus value for SETFOCUS
- `camera_serial:string` — which camera to target

### Signal Flow: SETFOCUS

```
orange GUI (dosa-live)
  |-- host_broadcast_setfocus() --> ENet broadcast
  |-- also applies locally if camera is on dosa-live
  v
orange_client (dosa0/dosa1)
  |-- ENet receive: stores request in camera_control->setfocus
  v
Camera thread (in get_one_frame during recording/streaming)
  |-- Detects new setfocus.generation
  |-- Calls update_focus_value() on camera
  |-- Waits 30 frames for motor to settle
  |-- Grabs 1024x1024 center crop JPEG via cudaMemcpy2D
  |-- Stores in camera_control->setfocus.reply_jpeg
  v
orange_client main ENet loop
  |-- Detects reply_ready, sends "JPGF" packet back via ENet (reliable)
  v
orange GUI enet_thread.h
  |-- Detects "JPGF" magic header, stores JPEG in g_remote_preview_jpeg
  v
orange GUI render loop
  |-- Decodes JPEG, uploads to GL texture, displays in ImGui::Image()
```

For local cameras on dosa-live, the camera thread writes directly to `camera_control->setfocus.reply_jpeg`, and the render loop picks it up (no ENet round-trip).

### Signal Flow: STARTSTREAM

Same as STARTRECORDING except:
- orange_client sets `camera_control->record_video = false` before PTP start
- No GPU encoder is created, no video files written
- Frames still flow through `get_one_frame` so SETFOCUS preview works

### Key Files

| File | Changes |
|------|---------|
| `schema/fetch.fbs` | Added SETFOCUS, STARTSTREAM enums and focus_value/camera_serial fields |
| `src/fetch_generated.h` | Auto-generated from schema |
| `src/project.cpp/h` | `host_broadcast_setfocus()`, `host_broadcast_start_stream()` |
| `src/orange.cpp` | Remote Focus GUI panel, preview window, stream mode buttons |
| `src/orange_headless_client.cpp` | SETFOCUS/STARTSTREAM handlers, ManagerContext with ecams/camera_control pointers |
| `src/video_capture.cpp` | SETFOCUS in get_one_frame (focus apply + preview grab), focus sweep in start_ptp_sync |
| `src/video_capture.h` | `SetFocusRequest` struct in `CameraControl`, `focus_test_gen_processed` in `CameraEachSelect` |
| `src/enet_thread.h` | JPGF packet receive handler, shared preview buffer externs |
| `src/mjpeg_stream.h` | MJPEG-over-HTTP server class (per-camera streaming) |
| `CMakeLists.txt` | Added `opencv_videoio` to orange and orange_client link targets |
| `quick_build/orange_client.sh` | Added `-lopencv_videoio` |

## Build & Deploy

### dosa-live (orange GUI)

```bash
cd ~/src/lime/build_make
make -j$(nproc) orange
cp build_make/orange targets/orange
cd ~/src/lime && sudo bash run.sh
```

### dosa0 / dosa1 (orange_client)

The headless client is built with `g++` directly (not CMake) via `quick_build/orange_client.sh`.

```bash
# 1. Sync source files from dosa-live:
scp ~/src/lime/src/*.cpp ~/src/lime/src/*.h ~/src/lime/src/*.hpp vlan-dosaX:~/src/lime/src/
scp ~/src/lime/quick_build/orange_client.sh vlan-dosaX:~/src/lime/quick_build/

# 2. Build on the target machine:
ssh vlan-dosaX "cd ~/src/lime && bash quick_build/orange_client.sh"

# 3. Run:
ssh vlan-dosaX "cd ~/src/lime && sudo bash run_orange.sh"
```

### Regenerating FlatBuffers (after schema changes)

```bash
cd ~/src/lime
flatc --cpp -o src/ schema/fetch.fbs
# Then rebuild orange + orange_client and redeploy
```

## Known Issues

### Cam 2008667 Focus Motor Broken
- `EVT_CameraSetUInt32Param("Focus", X)` followed by `EVT_CameraGetUInt32Param("Focus")` always returns 0
- The motor does not respond to any focus value
- Hardware issue — needs physical inspection or replacement

### IR Lights Require PTP Gate
- Cameras are IR and operate in the dark
- IR lights only turn on when the PTP acquisition gate fires (hardware-controlled GPO strobe)
- Free-run mode (ptp_sync_off) produces black frames regardless of exposure/gain
- This is why "Start Stream" is needed — it fires the PTP gate to activate IR

### Pre-Recording Frames Are Dark
- Before PTP gate fires, all frames have brightness ~2.7 (essentially black)
- Cannot measure sharpness or brightness outside of recording/streaming
- The automated focus sweep only works during stream mode (when IR is on)

### EVT_CameraQueueFrame Before AcquisitionStart
- Calling `EVT_CameraQueueFrame` before the first `AcquisitionStart` causes a segfault
- Frame buffers are not initialized until the EVT SDK starts acquisition
- Fix: call `EVT_CameraGetFrame` directly (SDK manages initial buffer internally)

### MJPEG Stream
- Per-camera HTTP stream on port 8080+camera_id
- Only delivers frames during active recording/streaming (not during WAITSTART)
- Access: `ffplay http://vlan-dosaX:808Y` or open in browser
