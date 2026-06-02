#ifndef GLOBAL_H
#define GLOBAL_H

#include <atomic>
#include <cstdint>

extern std::atomic<double> streaming_fps;
extern std::atomic<int> streaming_target_fps;
extern std::atomic<int64_t> record_start_time_ns;
bool try_start_timer();
bool try_stop_timer();

#endif // GLOBAL_H
