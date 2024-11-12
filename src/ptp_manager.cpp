// ptp_manager.cpp
#include "ptp_manager.h"
#include "emergent_camera.h"
#include <chrono>
#include <stdexcept>
#include <iostream>
#include <atomic>

namespace evt {

PTPException::PTPException(const std::string& msg) : std::runtime_error(msg) {}

PTPManager::PTPManager(EmergentCamera& camera) : camera_(camera) {
    if (!camera_.isOpen()) {
        throw PTPException("Camera must be opened before initializing PTP");
    }
}

void PTPManager::enablePTP() {
    try {
        // Configure PTP triggering settings
        camera_.setParameter("TriggerSource", "Software");
        camera_.setParameter("AcquisitionMode", "MultiFrame");
        camera_.setParameter("TriggerMode", "On");
        camera_.setParameter("PtpMode", "TwoStep");
        
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
        // Reset to default continuous mode settings
        camera_.setParameter("AcquisitionMode", "Continuous");
        camera_.setParameter("TriggerSelector", "AcquisitionStart");
        camera_.setParameter("TriggerMode", "Off");
        camera_.setParameter("TriggerSource", "Software");

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
        int32_t new_offset = camera_.getPTPOffset();
        
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
    
    // Split into high/low 32-bit values
    uint32_t high = static_cast<uint32_t>(global_time >> 32);
    uint32_t low = static_cast<uint32_t>(global_time & 0xFFFFFFFF);
    
    // Set camera parameters
    camera_.setParameter("PtpAcquisitionGateTimeHigh", high);
    camera_.setParameter("PtpAcquisitionGateTimeLow", low);
    
    std::cout << "PTP Gate time(ns): " << global_time << std::endl;
}

void PTPManager::waitForGateTime() {
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