#pragma once

#include "emergent_camera.h"
#include "video_capture.h"
#include <vector>
#include <cstdint>

namespace evt {

struct PTPState {
    int ptp_offset{0};
    int ptp_offset_sum{0};
    int ptp_offset_prev{0};
    unsigned int ptp_time_low{0};
    unsigned int ptp_time_high{0};
    unsigned int ptp_time_plus_delta_to_start_low{0};
    unsigned int ptp_time_plus_delta_to_start_high{0};
    unsigned long long ptp_time_delta_sum{0};
    unsigned long long ptp_time_delta{0};
    unsigned long long ptp_time{0};
    unsigned long long ptp_time_prev{0};
    unsigned long long ptp_time_countdown{0};
    unsigned long long frame_ts{0};
    unsigned long long frame_ts_prev{0};
    unsigned long long frame_ts_delta{0};
    unsigned long long frame_ts_delta_sum{0};
    unsigned long long ptp_time_plus_delta_to_start{0};
    char ptp_status[100];
    unsigned long ptp_status_sz_ret{0};
    unsigned int ptp_time_plus_delta_to_start_uint{0};
};

struct PTPParams{
    unsigned long long ptp_global_time; 
    unsigned long long ptp_stop_time;
    uint64_t ptp_counter;
    uint64_t ptp_stop_counter;
    bool network_sync = false;
    bool ptp_start_reached = false;
    bool ptp_stop_reached = false;
    bool network_set_stop_ptp = false;
    bool network_set_start_ptp = false;
};

class PTPManager {
public:
    explicit PTPManager(EmergentCamera& camera);
    ~PTPManager() = default;

    // Prevent copying
    PTPManager(const PTPManager&) = delete;
    PTPManager& operator=(const PTPManager&) = delete;

    // Core PTP operations
    void enablePTP();
    void disablePTP();
    
    // Time synchronization
    uint64_t getCurrentTime() const;
    int32_t getCurrentOffset() const;
    void updateOffset();
    void waitForTimestamp(uint64_t target_time);
    void setupGateTime(uint64_t global_time);
    void waitForGateTime();
    uint64_t getCurrentPTPTime() const;
    uint64_t updatePTPGateTime();

    // Getter for internal state
    const PTPState& getState() const { return ptp_state_; }
    
    // Multi-camera synchronization
    static bool synchronizeMultipleCameras(std::vector<PTPManager*>& cameras, 
                                         uint64_t delay_ns = 3000000000ULL); // 3 second default delay

    bool isEnabled() const { return is_enabled_; }

private:
    EmergentCamera& camera_;
    bool is_enabled_{false};
    int32_t current_offset_{0};
    int32_t last_offset_{0};
    PTPState ptp_state_;
    void updateGateTimeRegisters(uint64_t time);
    
    // Maximum allowable offset between cameras for synchronization (100 microseconds)
    static constexpr int32_t MAX_SYNC_OFFSET_NS = 100000;
};

} // namespace evt