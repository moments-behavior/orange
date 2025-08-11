#include "global.h"

std::mutex mtx3d;
std::condition_variable cv3d;
std::atomic<double> streaming_fps = 0.0;
std::atomic<int> streaming_target_fps = 60;
std::atomic<int64_t> record_start_time_ns{0};
std::atomic<CalibState> calib_state{CalibIdle};
Detection3d detection3d;
DetectionDataPerCam *detection2d;
std::atomic<uint64_t> detector_counter{0};
std::mutex graph_capture_mutex;
PeerRegistry peers;
EnetRuntime net;

bool try_start_timer() {
    int64_t expected = record_start_time_ns.load();

    // If already running, do nothing
    if (expected > 0)
        return false;

    // Try to restart if it's either 0 (not started) or -1 (stopped)
    int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count();

    while (expected <= 0) {
        if (record_start_time_ns.compare_exchange_strong(expected, now_ns)) {
            return true; // Successfully started or restarted
        }
        // CAS failed, reload and check again
        // `expected` will have been updated automatically
    }

    return false; // Another thread started it
}

bool try_stop_timer() {
    int64_t expected = record_start_time_ns.load();

    if (expected <= 0)
        return false;

    return record_start_time_ns.compare_exchange_strong(expected, -1);
}
