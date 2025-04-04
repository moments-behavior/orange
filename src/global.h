#ifndef GLOBAL_H
#define GLOBAL_H

#include <atomic>
#include <mutex>

extern std::atomic<double> streaming_fps;
extern std::atomic<int> streaming_target_fps;
// Calibration state enumeration
enum CalibState {
    CalibIdle,
    CalibNextPose,
    CalibPoseReached,
    CalibSavePictures
};

inline const char * const *enum_names_calib_state() {
    static const char * const names[5] = {
        "Idle",
        "NextPose",
        "PoseReached",
        "SavePictures",
        nullptr
      };
    return names;
}

// Atomic calibration state (now using CalibState instead of int)
extern std::atomic<CalibState> calib_state;

#endif // GLOBAL_H
