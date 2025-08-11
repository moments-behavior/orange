#ifndef GLOBAL_H
#define GLOBAL_H

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

bool try_start_timer();
bool try_stop_timer();

// Calibration state enumeration
enum CalibState {
    CalibIdle,
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
extern PeerRegistry peers;
extern EnetRuntime net;
// for 3d detection
extern Detection3d detection3d;
extern DetectionDataPerCam *detection2d;
#endif // GLOBAL_H
