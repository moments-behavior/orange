#ifndef GLOBAL_H
#define GLOBAL_H

#include "galvo_control_link.h"
#include "galvo_sender.h"
#include "realtime_tool.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
extern std::atomic<double> streaming_fps;
extern std::atomic<int> streaming_target_fps;
extern std::atomic<int64_t> record_start_time_ns;
extern std::mutex mtx3d;
extern std::condition_variable cv3d;
extern std::atomic<uint64_t> detector_counter;
extern std::mutex graph_capture_mutex;

// Latest PtpOffset per camera, published by each camera's capture thread (which
// already reads PtpOffset every frame). The "Start PTP Logging" worker reads
// these cached values instead of issuing its own GVCP requests — concurrent GVCP
// on the same camera collides (GVCP ACK error 0300) and crashes the EVT SDK.
constexpr int kMaxCameras = 20;
extern std::atomic<int> g_cam_ptp_offset[kMaxCameras];
bool try_start_timer();
bool try_stop_timer();

// Calibration state enumeration
enum CalibState {
    CalibIdle,
    CalibStart,
    CalibOpenCamera,
    CalibNextPose,
    CalibPoseReached,
    CalibSavePictures
};

inline const char *const *enum_names_calib_state() {
    static const char *const names[5] = {"Idle", "NextPose", "PoseReached",
                                         "SavePictures", nullptr};
    return names;
}

// Atomic calibration state (now using CalibState instead of int)
extern std::atomic<CalibState> calib_state;

// for 3d detection
extern Detection3d detection3d;
extern DetectionDataPerCam *detection2d;

// galvo target streaming (UDP sender to the Windows motor-control app)
extern GalvoSenderParams galvo_sender_params;

// galvo control link (GCC1/GCS1 request-reply channel to the same app)
extern GalvoLinkParams galvo_link_params;
#endif // GLOBAL_H
