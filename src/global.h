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
bool try_start_timer();
bool try_stop_timer();

#endif // GLOBAL_H
