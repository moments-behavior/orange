#ifndef GLOBAL_H
#define GLOBAL_H

#include <atomic>
#include <cstdint>

extern std::atomic<double> streaming_fps;
extern std::atomic<int> streaming_target_fps;
extern std::atomic<int64_t> record_start_time_ns;

// Latest PtpOffset per camera, published by each camera's capture thread (which
// already reads PtpOffset every frame). The "Start PTP Logging" worker reads
// these cached values instead of issuing its own GVCP requests — concurrent GVCP
// on the same camera collides (GVCP ACK error 0300) and crashes the EVT SDK.
constexpr int kMaxCameras = 20;
extern std::atomic<int> g_cam_ptp_offset[kMaxCameras];

// Latest average frame brightness (mono8 mean, 0-255) per camera, published by
// each camera's encoder thread via a throttled, subsampled GPU reduction (see
// gpu_video_encoder.cpp). The GUI "Camera Brightness" plot samples these to show
// light level over a recording. A negative value means "no sample yet".
extern std::atomic<float> g_cam_brightness[kMaxCameras];

bool try_start_timer();
bool try_stop_timer();

// Seconds elapsed since recording started (record_start_time_ns), or -1 if not
// currently recording. Uses the same steady_clock as try_start_timer().
double recording_elapsed_seconds();

#endif // GLOBAL_H
