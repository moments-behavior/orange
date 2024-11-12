#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <stdexcept>

namespace evt {

// Forward declarations
class EmergentCamera;
struct CameraParams;

// External structs we need to include
struct PTPParams {
    uint64_t ptp_global_time{0}; 
    uint64_t ptp_stop_time{0};
    uint64_t ptp_counter{0};
    uint64_t ptp_stop_counter{0};
    bool network_sync{false};
    bool ptp_start_reached{false};
    bool ptp_stop_reached{false};
    bool network_set_stop_ptp{false};
    bool network_set_start_ptp{false};
};

// Exception class for PTP-related errors
class PTPException : public std::runtime_error {
public:
    explicit PTPException(const std::string& msg) : std::runtime_error(msg) {}
};

// PTP State structure
struct PTPState {
    int32_t ptp_offset{0};
    int32_t ptp_offset_sum{0};
    int32_t ptp_offset_prev{0};
    uint32_t ptp_time_low{0};
    uint32_t ptp_time_high{0};
    uint32_t ptp_time_plus_delta_to_start_low{0};
    uint32_t ptp_time_plus_delta_to_start_high{0};
    uint64_t ptp_time_delta_sum{0};
    uint64_t ptp_time_delta{0};
    uint64_t ptp_time{0};
    uint64_t ptp_time_prev{0};
    uint64_t ptp_time_countdown{0};
    uint64_t frame_ts{0};
    uint64_t frame_ts_prev{0};
    uint64_t frame_ts_delta{0};
    uint64_t frame_ts_delta_sum{0};
    uint64_t ptp_time_plus_delta_to_start{0};
    char ptp_status[100];
    uint32_t ptp_status_sz_ret{0};
    uint32_t ptp_time_plus_delta_to_start_uint{0};
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
    
    // Maximum allowable offset between cameras for synchronization (100 microseconds)
    static constexpr int32_t MAX_SYNC_OFFSET_NS = 100000;
};

// Free function declarations
void showPTPOffset(PTPState* ptp_state, EmergentCamera* camera);

void startPTPSync(PTPState* ptp_state, 
                 PTPParams* ptp_params, 
                 CameraParams* camera_params, 
                 EmergentCamera* camera, 
                 unsigned int delay_in_second);

void grabFramesAfterCountdown(PTPState* ptp_state, EmergentCamera* camera);

} // namespace evt