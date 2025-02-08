#ifndef GLOBAL_H
#define GLOBAL_H

#include <atomic>
#include <mutex>

// Calibration state enumeration
enum CalibState {
    CalibIdle,
    CalibNextPose,
    CalibPoseReached,
    CalibSavePictures
};

// Atomic calibration state (now using CalibState instead of int)
extern std::atomic<CalibState> calib_state;

#endif // GLOBAL_H
