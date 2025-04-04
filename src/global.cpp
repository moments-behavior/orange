#include "global.h"

std::atomic<double> streaming_fps = 0.0;
std::atomic<int> streaming_target_fps = 60;
std::atomic<CalibState> calib_state{CalibIdle};