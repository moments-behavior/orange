#pragma once

#include "ptp_manager.h"
#include "emergent_camera.h"
#include <chrono>
#include <stdexcept>
#include <iostream>

namespace evt {

class PTPException : public std::runtime_error {
public:
    explicit PTPException(const std::string& msg) : std::runtime_error(msg) {}
};

PTPManager::PTPManager(EmergentCamera& camera) : camera_(camera) {
    if (!camera_.isOpen()) {
        throw PTPException("Camera must be opened before initializing PTP");
    }
}

void PTPManager::enablePTP() {
    try {
        camera_.enablePTPSync();
        is_enabled_ = true;
        
        // Get initial offset values to verify PTP is working
        updateOffset();
    } catch (const std::exception& e) {
        throw PTPException(std::string("Failed to enable PTP: ") + e.what());
    }
}

void PTPManager::disablePTP() {
    if (!is_enabled_) {
        return;
    }

    try {
        camera_.disablePTPSync();
        is_enabled_ = false;
    } catch (const std::exception& e) {
        throw PTPException(std::string("Failed to disable PTP: ") + e.what());
    }
}

void PTPManager::updateOffset() {
    if (!is_enabled_) {
        throw PTPException("PTP is not enabled");
    }

    int64_t offset_sum = 0;
    constexpr int SAMPLES = 5;

    // Get multiple samples to average the offset
    for (int i = 0; i < SAMPLES;) {
        int32_t new_offset = getCurrentOffset();
        
        if (new_offset != last_offset_) {
            offset_sum += new_offset;
            last_offset_ = new_offset;
            i++;
        }
    }

    current_offset_ = offset_sum / SAMPLES;
}

uint64_t PTPManager::getCurrentTime() const {
    if (!is_enabled_) {
        throw PTPException("PTP is not enabled");
    }
    return camera_.getCurrentPTPTime();
}

int32_t PTPManager::getCurrentOffset() const {
    if (!is_enabled_) {
        throw PTPException("PTP is not enabled");
    }
    return current_offset_;
}

void PTPManager::waitForTimestamp(uint64_t target_time) {
    if (!is_enabled_) {
        throw PTPException("PTP is not enabled");
    }

    uint64_t current_time;
    uint64_t last_display_time = 0;
    constexpr uint64_t DISPLAY_INTERVAL = 1000000000ULL; // 1 second in nanoseconds

    do {
        current_time = getCurrentTime();
        
        // Display countdown every second
        if (current_time > last_display_time + DISPLAY_INTERVAL) {
            uint64_t seconds_remaining = (target_time - current_time) / 1000000000ULL;
            std::cout << "PTP countdown: " << seconds_remaining << "s\n";
            last_display_time = current_time;
        }

    } while (current_time <= target_time);
}

bool PTPManager::synchronizeMultipleCameras(std::vector<PTPManager*>& cameras, uint64_t delay_ns) {
    if (cameras.empty()) {
        return false;
    }

    // Get current PTP time from first camera
    uint64_t base_time = cameras[0]->getCurrentTime();
    uint64_t target_time = base_time + delay_ns;

    // Set target time for all cameras
    for (auto* ptp : cameras) {
        if (!ptp->is_enabled_) {
            throw PTPException("All cameras must have PTP enabled for synchronization");
        }
        
        // Verify all cameras are within acceptable offset range
        ptp->updateOffset();
        if (std::abs(ptp->getCurrentOffset()) > MAX_SYNC_OFFSET_NS) {
            throw PTPException("Camera PTP offset exceeds maximum allowable range");
        }
    }

    // Wait for all cameras to reach target time
    for (auto* ptp : cameras) {
        ptp->waitForTimestamp(target_time);
    }

    return true;
}

void PTPManager::setupGateTime(uint64_t global_time) {
    ptp_state_.ptp_time_plus_delta_to_start = global_time;
    ptp_state_.ptp_time_plus_delta_to_start_uint = global_time;
    
    // Split into high/low 32-bit values
    ptp_state_.ptp_time_plus_delta_to_start_low = (unsigned int)(global_time & 0xFFFFFFFF);
    ptp_state_.ptp_time_plus_delta_to_start_high = (unsigned int)(global_time >> 32);
    
    // Set camera parameters
    camera_.updatePTPGateTime(ptp_state_.ptp_time_plus_delta_to_start_high,
                             ptp_state_.ptp_time_plus_delta_to_start_low);
    
    std::cout << "PTP Gate time(ns): " << global_time << std::endl;
}

uint64_t EmergentCamera::updatePTPGateTime(unsigned int high, unsigned int low) {
    checkError(EVT_CameraSetUInt32Param(camera_.get(), "PtpAcquisitionGateTimeHigh", high),
        "Setting PTP gate time high");
    checkError(EVT_CameraSetUInt32Param(camera_.get(), "PtpAcquisitionGateTimeLow", low),
        "Setting PTP gate time low");
    }

void EmergentCamera::getPTPStatus(char* status, size_t size, unsigned long* ret_size) {
    checkError(EVT_CameraGetEnumParam(camera_.get(), "PtpStatus", 
            status, size, ret_size), "Getting PTP status");
    }

    int32_t getPTPOffset() const {
        int32_t offset;
    checkError(EVT_CameraGetInt32Param(camera_.get(), "PtpOffset", &offset),
            "Getting PTP offset");
        return offset;
    }

void PTPManager::waitForGateTime() {
    // Implementation of grab_frames_after_countdown logic
    uint64_t current_time;
    uint64_t countdown_time = 0;
    
    do {
        current_time = getCurrentTime();
        
        if (current_time > countdown_time) {
            std::cout << (ptp_state_.ptp_time_plus_delta_to_start - current_time) / 1000000000ULL << std::endl;
            countdown_time = current_time + 1000000000ULL; // next update in 1 second
        }
    } while (current_time <= ptp_state_.ptp_time_plus_delta_to_start);
    
    std::cout << std::endl;
}

} // namespace evt