# Chunked recording

Goal: record for days by splitting each recording into fixed-length chunks
(30 min in production) that a separate process can move to long-term storage,
without losing frames and with every camera's chunk starting on the same
exposure.

## Why the cameras are not restarted

In `rob_minimal`, starting a recording restarts acquisition on every camera:
`stop_camera_streaming` closes the streams and frees the frame rings, then
`start_camera_streaming` reopens them, allocates the rings again (100
GPUDirect buffers per camera), creates the encoders, and runs the PTP start
barrier. That barrier programs `PtpAcquisitionGateTime` 3 s in the future and
counts down to it (`start_ptp_sync(..., 3)` in `video_capture.cpp`). Chunking
by stopping and restarting the recording would therefore leave every camera
blind for at least the 3 s gate plus the stream teardown and setup, at every
chunk boundary.

The barrier also waits for all cameras, so restarting one camera at a time is
not possible without new code. It would also need its own frame-phase-aligned
gate to stay in step with the cameras that kept running.

None of that is needed. A chunk boundary only has to change where the encoded
packets are written. The cameras, the PTP gate and the encoders keep running
for the whole recording. Only each encoder's output file rolls over.

## How it works

`src/recording_chunks.{h,cpp}`, `src/gpu_video_encoder.cpp`:

1. **Boundary by sync-frame index.** Each encoder numbers every frame
   from the shared PTP gate. The first frame is
   `round((ts_first - gate) / period)`; on this rig that is 0, since first
   frames arrive ~360 ns after the gate. Each later frame is the previous
   index plus `round((ts - ts_prev) / period)`. Then
   `chunk = index / frames_per_chunk`.
   Because the index advances by PTP time, a camera that dropped frames (or
   its first frame) still numbers the next exposure the same as the other
   cameras. On EmergentTech2, corresponding frames of the 8 cameras carry
   timestamps within ~20 ns of each other (median; ≤ 4 µs worst case over
   12k frames), far inside the half-period rounding margin. Every chunk holds
   exactly `frames_per_chunk` frames: 108 000 for 30 min at 60 fps, 1800 for
   30 s.

   *Why not a fixed PTP time per boundary?* The first implementation cut at
   `ref + k * chunk_ns`. The camera's real period, 16 666 666 ns, does not
   divide 30 min (1.8e12 ns = 108 000.0043 periods). The cut therefore slides
   ~72 µs per chunk across the frame grid and reaches a frame time after ~2.4
   days. There, a few µs of cross-camera timestamp jitter can put the same
   exposure in different chunks on different cameras.
   `tests/run_chunk_clock_test.sh` simulates 8 cameras for up to 4 days, with
   4 µs jitter, random drops, 2–60 frame bursts and lost first frames.
   Over 366 M frames the frame-count rule assigned every frame correctly. The
   time rule produced 9 cross-camera disagreements in the 3-day / 30 s-chunk
   case.
2. **Forced IDR.** The first frame of a chunk is encoded with
   `NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS`, so every file
   starts on a self-contained keyframe with its parameter sets. The encoder is
   not re-created.
3. **Routing past NVENC's output delay.** NVENC returns each frame's packet ≥ 3
   frames after the frame is submitted. Each submitted frame records its chunk
   writer in a FIFO, and each returned packet is muxed into the writer at the
   head of the FIFO. Metadata rows are written at submission into the same
   chunk, so the mp4 and the csv of a chunk always hold the same frames.
4. **Finalize off the encode path.** When the first packet of a new chunk is
   muxed, the previous chunk's writer is handed to a per-encoder finalizer
   thread. That thread runs `av_write_trailer` + close and closes the csv. The
   encoder thread only opens the new files (measured 0.2–0.4 ms).
5. **Completion marker.** When every recording camera has closed chunk *k*,
   `chunk_NNNNN/chunk_complete.json` is written (atomically, via rename). It
   holds per-camera frame counts, first/last `frame_id` and timestamps, a
   first-packet-is-keyframe flag, finalize time, the cross-camera start spread,
   and whether all frame counts are equal. A mover can copy any chunk folder
   that has this file.

`frame_id` in `_meta.csv` keeps counting across chunks (it is the per-camera
received-frame counter), so a chunked recording can be concatenated back into
one timeline.

The new `Encoder queue full` count on stop reports frames the capture thread
could not hand to the encoder. These were previously dropped silently and
showed up only as `frame_id` gaps.

## Using it

- GUI: tick **Chunk recording**, set **Chunk minutes** (≥ 0.1), press record.
- Command line (unattended):
  `sudo -E release/orange --preset <name> --chunk-seconds 1800 --record-seconds <N> [--record-dir DIR]`.
  With `--record-seconds`, orange opens every camera with that preset,
  records for N s (measured from the PTP gate), stops and exits.
  `--preset` and `--chunk-seconds` alone just preselect them in the GUI.
- Validate: `python3 tools/check_chunks.py <recording folder>`. It exits 1 on
  any frame gap, cross-camera misalignment, mp4/csv mismatch, non-keyframe
  start, or missing `chunk_complete.json`.

## Test results (EmergentTech2, 8 cameras, 60 fps, GPUDirect, HEVC p1)

| Run | Chunks | Frames / camera | Camera drops | Encoder drops | Boundary alignment |
|---|---|---|---|---|---|
| 100 s, 30 s chunks (time-based cut) | 4 | 6001 | 0 | 0 | all cameras cut on the same frame_id (1800, 3600, 5400), start spread 25 ns |
| 20 min, 30 s chunks (time-based cut) | 41 | 72002 | 0 | 0 | 40 × 1800 frames on every camera, start spread 14–29 ns, encoder queue 0 at all 320 rollovers, file open ≤ 0.5 ms |
| 10 min, 30 s chunks (sync-frame cut, current) | 21 | 36002 | 0 | 0 | 20 × 1800 frames on every camera, start spread 22–28 ns |

All three runs pass `tools/check_chunks.py`: frame continuity across every boundary (each boundary exactly one period), mp4 packets = csv rows, and each file starts on a decodable keyframe. Not yet tested: a multi-hour run with production-length (30 min) chunks.

## Still to watch

- **Page-cache writeback.** Jeremy Delahanty's fork saw multi-camera stalls
  that lined up with kernel dirty-page flushes on another machine. He fixed
  them with `sync_file_range` pacing in the writer plus
  `vm.dirty_background_bytes` / `vm.dirty_bytes`. This box has 503 GB of RAM
  and the default 10 % / 20 % dirty ratios. Chunking does not change the write
  pattern, but a multi-day run is where a writeback stall would show up.
- **Disk space.** At these settings 8 cameras write ~4.4 GB/min (~265 GB/h),
  so the 27 TB `/mnt/data` array holds ~4 days.
