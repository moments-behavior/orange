#include "global.h"

std::atomic<double> streaming_fps = 0.0;
std::atomic<CalibState> calib_state{CalibIdle};