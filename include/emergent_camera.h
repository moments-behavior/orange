#pragma once

#include <string>
#include <memory>
#include <stdexcept>
#include "camera.h"
#include "camera_params.h"
#include "gpu_manager.h"

namespace evt {

class CameraException : public std::runtime_error {
public:
    explicit CameraException(const std::string& msg, EVT_ERROR code = EVT_GENERAL_ERROR) 
        : std::runtime_error(msg), error_code(code) {}
    EVT_ERROR getErrorCode() const { return error_code; }
private:
    EVT_ERROR error_code;
};

class EmergentCamera {
public:
    // Constructor takes ownership of configuration
    explicit EmergentCamera(const CameraParams& params);
    ~EmergentCamera();

    // Prevent copying
    EmergentCamera(const EmergentCamera&) = delete;
    EmergentCamera& operator=(const EmergentCamera&) = delete;

    // Allow moving
    EmergentCamera(EmergentCamera&&) noexcept;
    EmergentCamera& operator=(EmergentCamera&&) noexcept;

    // Core camera operations
    void open(const GigEVisionDeviceInfo* device_info);
    void close();
    void startStream();
    void stopStream();
    
    // Frame handling
    void allocateFrameBuffers(int buffer_size);
    void releaseFrameBuffers();
    void queueFrame(Emergent::CEmergentFrame* frame);
    bool getFrame(Emergent::CEmergentFrame* frame, int timeout_ms);

    // Camera settings
    void updateExposure(int exposure_value);
    void updateGain(int gain_value);
    void updateFrameRate(int frame_rate);
    void updateResolution(int width, int height);
    void updateOffset(int x, int y);
    void updatePixelFormat(const std::string& format);
    void updateIris(int iris_value);

    // PTP synchronization
    void enablePTPSync();
    void disablePTPSync();
    uint64_t getCurrentPTPTime() const;

    // Status and monitoring
    int getSensorTemperature() const;
    bool isOpen() const { return is_open_; }
    bool isStreaming() const { return is_streaming_; }
    const CameraParams& getParams() const { return params_; }

    // Debug/testing
    // void testGPIOToggle(); // TODO: Implement this

private:
    void configureDefaults();
    // void validateResolution(int width, int height) const; // TODO: Implement this
    void checkError(EVT_ERROR err, const std::string& operation) const;
    void setFrameBufferFormat(Emergent::CEmergentFrame* frame) const;
    void printDeviceInfo(const GigEVisionDeviceInfo* device_info) const;
    static void printDeviceInfo(const GigEVisionDeviceInfo* device_info, int camera_idx);
    bool initializeGPUDirect();
    void validateGPUConfiguration() const;

// USAGE For a specific camera instance
// camera.printDeviceInfo(device_info);

// // For any camera in an array
// evt::EmergentCamera::printDeviceInfo(device_info_array, camera_idx);


    mutable CameraParams params_;  // Make params_ mutable to for caching in const functions
    std::unique_ptr<Emergent::CEmergentCamera> camera_;
    std::vector<Emergent::CEmergentFrame> frame_buffers_;
    bool is_open_ = false;
    bool is_streaming_ = false;
};

} // namespace evt