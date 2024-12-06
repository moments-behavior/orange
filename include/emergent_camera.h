#pragma once

#include <string>
#include <memory>
#include <stdexcept>
#include <unistd.h>
#include <string>
#include <algorithm>
#include <vector>
#include <numeric>
#include "camera_params.h"
#include "gpu_manager.h"

#ifndef  EMERGENT_SDK
#include <EmergentCameraAPIs.h>
#include <emergentframe.h>
#include <EvtParamAttribute.h>
#include <gigevisiondeviceinfo.h>
#include <emergentcameradef.h>
#include <emergentgigevisiondef.h>
#include <EvtParamAttribute.h>
#endif

namespace evt {

// Utility function to get human-readable error messages
std::string get_evt_error_string(EVT_ERROR error);

// Camera-specific exception class
class CameraException : public std::runtime_error {
public:
    explicit CameraException(const std::string& msg, EVT_ERROR code = EVT_GENERAL_ERROR) 
        : std::runtime_error(msg), error_code(code) {}
    EVT_ERROR getErrorCode() const { return error_code; }
private:
    EVT_ERROR error_code;
};

// Private implementation struct for Emergent camera hardware
struct CameraEmergent {
    Emergent::CEmergentCamera camera;
    Emergent::CEmergentFrame* evt_frame;
    Emergent::CEmergentFrame frame_recv;
    Emergent::CEmergentFrame frame_reorder;
};

// Helper function for error checking
inline void check_camera_errors(EVT_ERROR err, const char* camera_serial, 
                              const char* file, int line) {
    if (EVT_SUCCESS != err) {
        std::string error_string = get_evt_error_string(err);
        const char* errorStr = error_string.c_str();
        throw CameraException(
            std::string(camera_serial) + " error " + std::to_string(err) + 
            " (" + errorStr + ") at " + file + ":" + std::to_string(line)
        );
    }
}

#define CHECK_CAMERA_ERROR(err, camera_serial) \
    check_camera_errors(err, camera_serial, __FILE__, __LINE__)

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
    void updateFocus(int focus_value);

    // PTP synchronization
    void enablePTPSync();
    void disablePTPSync();
    uint64_t updatePTPGateTime(unsigned int high, unsigned int low);
    void getPTPStatus(char* status, size_t size, unsigned long* ret_size) const;
    int32_t getPTPOffset() const;
    uint64_t getCurrentPTPTime() const;

    // Status and monitoring
    int getSensorTemperature() const;
    bool isOpen() const { return is_open_; }
    bool isStreaming() const { return is_streaming_; }
    const CameraParams& getParams() const { return params_; }

    // Generic parameter setting interface
    void setParameter(const std::string& param, const std::string& value);
    void setParameter(const std::string& param, int value);
    void setParameter(const std::string& param, uint32_t value);
    void setParameter(const std::string& param, bool value);

    template<typename T>
    T getParameter(const std::string& param) const;

    // Debug/testing
    // void testGPIOToggle(); // TODO: Implement this

    // Add these new methods
    struct ParameterRange {
        unsigned int min;
        unsigned int max;
        unsigned int increment;
    };

    ParameterRange getExposureRange() const;
    ParameterRange getGainRange() const;
    ParameterRange getFrameRateRange() const;
    ParameterRange getFocusRange() const;
    ParameterRange getIrisRange() const;
    
    struct ResolutionRange {
        unsigned int width_min;
        unsigned int width_max;
        unsigned int width_inc;
        unsigned int height_min;
        unsigned int height_max;
        unsigned int height_inc;
    };
    
    ResolutionRange getResolutionRange() const;
    
    struct TemperatureRange {
        int min;
        int max;
    };
    
    TemperatureRange getTemperatureRange() const;

    // Add these new methods
    struct CameraState {
        int exposure;
        int gain;
        int frame_rate;
        int iris;
        int focus;
    };

    CameraState getCurrentState() const;
    void logCurrentState(const std::string& context) const;

    // Just keep the declarations
    void updateCameraRanges();
    void updateExposureAndFrameRate(int exposure_value, int frame_rate_value);

    // Add new method
    void updateParamsFromCurrentState(CameraParams& params) const {
        if (!is_open_) {
            throw CameraException("Cannot update params - camera not open");
        }

        auto state = getCurrentState();
        params.exposure = state.exposure;
        params.gain = state.gain;
        params.frame_rate = state.frame_rate;
        params.iris = state.iris;
        params.focus = state.focus;
    }

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