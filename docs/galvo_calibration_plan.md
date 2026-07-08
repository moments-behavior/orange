# Galvo calibration with a ChArUco board — design

Plan for calibrating the GalvoCam (mirror-steered camera driven by
`wchen27/GalvoControl`) against orange's camera world frame, so that streamed
GCT1 targets are aimed accurately without hand-tuning numbers. Ease of use is
the top priority: no typed-in parameters, no files copied between machines,
one wizard in orange.

## Concept

A ChArUco board placed in the arena is seen by both camera systems at once:

- **Fixed cameras** (already calibrated; extrinsics in `calib_yaml`) give the
  board's pose in **world coordinates** via PnP.
- The **galvo camera** (an Emergent camera captured by orange) sees the board
  through the pan/tilt mirrors. At a commanded `(pan, tilt)`, detecting the
  board in the galvo view and intersecting the optical axis with the board
  plane tells us **which world point the galvo is gazing at**.

Orange steps the mirrors through a grid of angles over a control channel,
auto-collects `(pan, tilt) → world gaze point` samples at 2–3 board
placements (different depths), fits the runtime targeting model, and uploads
the parameters to GalvoControl. A closed-loop verify step then streams the
board center as a normal GCT1 target and measures how far it lands from the
galvo image center — testing the entire runtime path.

ChArUco (vs. plain ArUco) matters because the zoomed-in galvo camera usually
sees only part of the board; ChArUco corners are identifiable from partial
views. Detection uses `cv::aruco::CharucoDetector` (OpenCV ≥4.7; we build
against 4.10 — requires linking `opencv_objdetect` + `opencv_aruco`).

## Targeting model (shared math)

GalvoControl's runtime model (`targeting_solve` in its `main.cpp`), extended
with a world→galvo frame rotation it currently lacks:

```
v  = Rᵀ · (target − O)                       # world → galvo frame
az = atan2(v.x, v.z);  el = atan2(v.y, √(v.x²+v.z²))     # degrees
pan_motor  = pan_sign  · az · pan_scale  + pan_offset     # displayed motor deg
tilt_motor = tilt_sign · el · tilt_scale + tilt_offset
```

- `O` — effective pivot ("base") position, world mm (3 params)
- `R` — galvo frame orientation in world, Euler XYZ degrees,
  `R = Rz(rz)·Ry(ry)·Rx(rx)` (3 params)
- `pan/tilt scale, offset` (4 params); signs fixed at init (±1, best of the
  four combinations on a linear pre-fit). Scale ≈0.5 for mirror half-angle.

Fit: for each sample, predict `(pan, tilt)` from the measured world point and
minimize the residual **in motor-degree space** (exactly the space the runtime
inversion works in) with Levenberg–Marquardt (`cv::LMSolver`, already linked
via `opencv_calib3d`). 10 parameters; ≥30 samples over ≥2 board depths keeps
it well-conditioned. A single board plane leaves `O` poorly constrained along
the gaze direction — hence the "move the board" step.

Known approximation: a two-mirror galvo has no single pivot (the effective
viewpoint shifts slightly with angle). For keeping a target in the galvo
camera's FOV this is fine. Raw samples are persisted, so a future two-mirror
model upgrade changes only the fit, not the collection.

## Protocol: GCC1/GCS1 control channel

The existing one-way GCT1 target stream (UDP :5005, `PROTOCOL.md` in
GalvoControl) is untouched. A request/reply channel is added on **UDP :5006**.
Little-endian, packed, fixed sizes; magic+version validated, mismatches
ignored (fail-safe, same spirit as GCT1). Orange retries a request (same
`seq`) on a ~250 ms timeout, ×4; commands are idempotent so retries are safe.
Every reply carries full status, so there is no separate "get" command per
field.

### GCC1 command (orange → GalvoControl), 128 bytes

| off | size | type    | field    | meaning |
|-----|------|---------|----------|---------|
| 0   | 4    | char[4] | magic    | `'G','C','C','1'` |
| 4   | 2    | u16     | version  | 1 |
| 6   | 2    | u16     | cmd      | see below |
| 8   | 4    | u32     | seq      | echoed in the reply |
| 12  | 4    | u32     | reserved | 0 |
| 16  | 112  | f64[14] | args     | per-command |

Commands:

| cmd | name        | args |
|-----|-------------|------|
| 0   | PING        | — (status query / calib-mode keepalive) |
| 1   | SET_ANGLES  | `[0]`=pan, `[1]`=tilt — displayed motor degrees |
| 2   | CALIB_MODE  | `[0]` = 0/1 |
| 3   | SET_CALIB   | `[0..2]`=O xyz mm, `[3..5]`=R euler deg, `[6..8]`=pan sign/scale/offset, `[9..11]`=tilt sign/scale/offset |
| 4   | SAVE_CONFIG | — (persist current config to `motor_control.cfg`) |
| 5   | STOP        | — (abort the pan/tilt axes) |

### GCS1 reply (GalvoControl → orange), 96 bytes

| off | size | type    | field     | meaning |
|-----|------|---------|-----------|---------|
| 0   | 4    | char[4] | magic     | `'G','C','S','1'` |
| 4   | 2    | u16     | version   | 1 |
| 6   | 2    | u16     | cmd       | echoed |
| 8   | 4    | u32     | seq       | echoed |
| 12  | 4    | u32     | flags     | bit0 ok · bit1 in_position · bit2 calib_mode · bit3 motors_enabled · bit4 remote_allowed |
| 16  | 4    | i32     | err       | 0 ok · 1 bad args · 2 remote disabled · 3 not connected · 4 clamped |
| 20  | 4    | u32     | reserved  | 0 |
| 24  | 8    | f64     | pan_deg   | current, displayed frame |
| 32  | 8    | f64     | tilt_deg  | current, displayed frame |
| 40  | 32   | f64[4]  | limits    | pan_min, pan_max, tilt_min, tilt_max (deg) |
| 72  | 24   | f64[3]  | reserved  | 0 |

### Semantics / safety

- All remote commands are gated by an **"Allow remote control"** checkbox in
  GalvoControl (persisted; motion commands refused with err=2 when off).
- `SET_ANGLES` is always clamped to the travel limits, and moves use the
  per-axis velocity/accel clamps already in place.
- **Calib mode**: while on, GCT1 target aiming and the test circle are
  suppressed so streaming can't fight the sweep. If no GCC1 packet arrives
  for >5 s in calib mode, GalvoControl auto-exits it and holds (orange's 2 Hz
  status poll acts as the keepalive).
- `in_position` = both targeting axes idle (not MOVING) with amps enabled.

## Orange side

New files (the phase-2 ones depend on `opencv_objdetect`/`opencv_aruco` and
must go on `cam_server`'s `REMOVE_ITEM` exclusion list; the link module is
plain sockets and builds into both targets, like `galvo_sender.cpp`):

1. **`src/galvo_control_link.{h,cpp}`** — GCC1/GCS1 client. One UDP socket,
   `SO_RCVTIMEO` 250 ms, retry ×4, replies matched by `seq`. One request in
   flight at a time (mutex). A background poller thread pings at 2 Hz while
   the link is up and publishes the latest `GalvoStatus` for the GUI (also
   serves as the calib-mode keepalive). Params in `GalvoLinkParams`
   (`global.h`), mirroring `GalvoSenderParams`.
2. **`src/charuco_detect.{h,cpp}`** *(phase 2)* — board spec (JSON presets in
   `config/`), corner detection, board pose from a `CameraCalibResults`.
3. **`src/galvo_calib.{h,cpp}`** *(phase 2)* — orchestrator thread + state
   machine (follows the `CalibState`/`detection3d_proc` conventions): sweep
   planning from the queried travel limits, dwell until `in_position` then
   require the detection stable across ~3 frames (dwelling means no frame
   sync between cameras is needed), sample acceptance, LM fit, residual
   report, persistence.
4. **Frame tap** *(phase 2)* — calibration needs ~1 CPU frame per dwell, not
   full rate: reuse the `FrameSaver` copy state machine
   (`State_Copy_New_Frame`) to pull the latest frame on demand for the galvo
   cam and each fixed cam.
5. **GUI** — the "Galvo Target Streaming" tree in `orange.cpp` grows into a
   "Galvo" section: Streaming (existing) · Control Link (connect, live
   status, manual raw-angle aim, calib-mode toggle) · Calibration wizard
   (phase 2) · Verify (phase 3).

### Wizard UX (phase 2)

1. **Preflight checklist** (all green before Start enables): control link
   alive · remote allowed · galvo camera selected · ≥1 fixed camera streaming
   with calibration loaded · board preset chosen · board currently detected
   in a fixed cam and the galvo cam.
2. **Automated sweep** — a grid centered on the zeroed/home pose (0,0),
   spanning a configurable ± range (default ±20° pan, ±15° tilt, clamped to
   the travel limits). The home pose is the reference "facing the arena"
   orientation: the useful optical range around it is far narrower than the
   travel limits, which protect the mechanics — far off home the camera just
   sees the back of the pan mirror. Live coverage heatmap; angles where the
   board isn't visible are skipped automatically; the mirrors return to home
   when the sweep ends.
3. **"Move the board (2 of 3)"** → re-sweep. Two placements minimum, three
   recommended.
4. **Fit & review** — RMS in degrees and mm at working distance, outliers
   flagged. One button: **Apply & upload** (SET_CALIB + SAVE_CONFIG).
5. **Verify** *(phase 3)* — stream the board center as a GCT1 target, then
   measure how far the galvo's gaze lands from the board center (mm on the
   board plane).

No galvo-camera intrinsics step: the zoomed view never sees enough of the
board to calibrate them, and the homography gaze method below doesn't need
them.

### How a sample's world point is computed (phase 2)

The galvo camera is far too zoomed to hold the whole board — often a single
ArUco marker barely fits. So the galvo side uses **no pose and no
intrinsics**: every visible board point (ChArUco corners *plus* raw marker
corners — one marker already gives 4) feeds a local homography image → board
plane, and the board-plane point under the **image center** is the gaze
sample. The constant offset between image center and true optical axis is
identical at every sample, so the fitted pan/tilt offsets absorb it exactly;
narrow-FOV distortion is negligible. Gaze points slightly past the physical
board edge are kept (the plane is exact there; a configurable margin bounds
homography extrapolation).

Fixed cam (the one with the most detected corners): `solvePnP` → board pose
in world → transform the board-coordinates gaze point to world.

The verify step measures its error the same way, in **mm on the board
plane** (distance between the aimed gaze point and the board center).

### Files

- `config/galvo_board.json` — `{squares_x, squares_y, square_mm, marker_mm, dictionary}`
- `calib_yaml/galvo_calib.json` — raw samples + fitted model + residuals + timestamp

## GalvoControl side

All in `src/main.cpp`, matching its single-threaded style:

1. **Control listener** — nonblocking UDP socket on :5006 polled on the UI
   thread each frame (RapidCode calls stay on the UI thread, the invariant
   the app already relies on). Parse GCC1, dispatch, reply GCS1 to the
   sender's address immediately.
2. **Targeting rotation** — `TargetConfig` gains `rot[3]` (Euler XYZ deg);
   `targeting_solve` applies `Rᵀ·(target − O)`; persisted as
   `tgt_rot_x/y/z` in `motor_control.cfg`; shown in the Targeting
   calibration tree.
3. **Remote gating + calib mode** — "Allow remote control" checkbox
   (persisted); calib-mode flag suppresses GCT1/test-circle aiming and
   auto-expires after 5 s of control silence.
4. **PROTOCOL.md** — new "Control channel (GCC1/GCS1)" section.

## Phases

1. **Control channel** (this change): GCC1/GCS1 on both sides + manual
   raw-angle aim from orange — immediately useful for bench testing.
2. **Calibration**: ChArUco module, sweep/collect/fit/upload.
3. **Verify & polish**: closed-loop pointing-error readout, coverage UI.
