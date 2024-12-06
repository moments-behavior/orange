#include "emergent_camera.h"
#include "NvEncoder/Logger.h"
#include <iostream>
#include <sstream>

namespace evt {

namespace {
    std::string getErrorString(EVT_ERROR error) {
        std::string error_string;
        switch (error) {
            case EVT_SUCCESS:
                return "Success";
            case EVT_ERROR_DEVICE_CONNECTED_ALRD:
                return "Camera has been opened.";
            case EVT_ERROR_DEVICE_NOT_CONNECTED:
                return "Camera has not been opened.";
            case EVT_ERROR_DEVICE_LOST_CONNECTION:
                return "Camera lost connection due to disconnected, powered off, crashing etc.";
            case EVT_ERROR_GENICAM_ERROR:
                return "Generic GeniCam error from GeniCam lib.";
            case EVT_ERROR_GENICAM_NOT_MATCH:
                return "Parameter not matched.";
            case EVT_ERROR_GENICAM_OUT_OF_RANGE:
                return "Parameter out of range.";
            case EVT_ERROR_SOCK:
                return "Socket operation failed.";
            case EVT_ERROR_GVCP_ACK:
                return "GVCP ACK error.";
            case EVT_ERROR_GVSP_DATA_CORRUPT:
                return "GVSP stream data corrupted, would cause block dropped.";
            case EVT_ERROR_NIC_LIB_INIT:
                return "Fail to initialize NIC's SDK library.";
            case EVT_ERROR_OS_OBTAIN_ADAPTER:
                return "Failed to get host adapter info.";
            case EVT_ERROR_SDK:
                return "SDK error.";
            default:
                return "Unknown error: " + std::to_string(static_cast<int>(error));
        }
    }
} // anonymous namespace

std::string get_evt_error_string(EVT_ERROR error) {
    return getErrorString(error);
}

EmergentCamera::EmergentCamera(const CameraParams& params) 
    : params_(params),
      camera_(std::make_unique<Emergent::CEmergentCamera>()) {
    
    if (params_.gpu_direct) {
        camera_->gpuDirectDeviceId = params_.gpu_id;
    }
}

EmergentCamera::~EmergentCamera() {
    if (is_streaming_) {
        stopStream();
    }
    if (is_open_) {
        close();
    }
}

void EmergentCamera::open(const GigEVisionDeviceInfo* device_info) {
    try {
        if (is_open_) {
            throw CameraException("Camera already open");
        }

        // Just open the camera - don't configure anything yet
        checkError(EVT_CameraOpen(camera_.get(), device_info), "Opening camera");
        is_open_ = true;

        // Only set the absolute minimum required settings
        configureDefaults();

        // Get all parameter ranges once during initialization
        updateCameraRanges();  // store all ranges in params_

        LOG(INFO) << "Camera opened successfully";

    } catch (const std::exception& e) {
        is_open_ = false;
        throw CameraException(std::string("Failed to open camera: ") + e.what());
    }
}

void EmergentCamera::startStream() {
    if (!is_open_) {
        throw CameraException("Cannot start stream - camera not open");
    }
    if (is_streaming_) {
        throw CameraException("Stream already started");
    }

    try {
        // Get camera temperature to verify communication
        int temp = getSensorTemperature();
        std::cout << "Camera temperature before stream: " << temp << std::endl;

        // Get current packet size and other stream parameters
        unsigned int packet_size;
        checkError(EVT_CameraGetUInt32Param(camera_.get(), "GevSCPSPacketSize", &packet_size), 
                  "Getting packet size");
        std::cout << "Stream packet size: " << packet_size << " bytes" << std::endl;

        if (params_.gpu_direct) {
            std::cout << "GPU Direct enabled - initializing GPU pipeline..." << std::endl;
            if (!initializeGPUDirect()) {
                std::cerr << "GPU Direct initialization failed - falling back to standard mode" << std::endl;
                // Reset GPU Direct flag to ensure standard path is used
                params_.gpu_direct = false;
                camera_->gpuDirectDeviceId = -1;
            }
        } else {
            std::cout << "Using standard streaming mode (GPU Direct disabled)" << std::endl;
        }

        // Configure memory allocations based on mode
        if (!params_.gpu_direct) {
            // Standard mode: Ensure host memory is properly aligned
            camera_->gpuDirectDeviceId = -1;  // Explicitly disable GPU Direct
            
            // You might want to set specific buffer modes or memory alignments here
            // For example:
            checkError(EVT_CameraSetEnumParam(camera_.get(), "StreamBufferHandlingMode", 
                                            "NewestOnly"), 
                      "Setting buffer handling mode");
        }

        // Open the stream
        EVT_ERROR stream_result = EVT_CameraOpenStream(camera_.get());
        if (stream_result != EVT_SUCCESS) {
            std::stringstream ss;
            ss << "Failed to open camera stream (Error: " << evt::get_evt_error_string(stream_result) << ")";
            if (params_.gpu_direct) {
                ss << " - Check GPU Direct compatibility";
            }
            throw CameraException(ss.str());
        }

        is_streaming_ = true;
        std::cout << "Stream successfully opened in " 
                  << (params_.gpu_direct ? "GPU Direct" : "standard") 
                  << " mode" << std::endl;

    } catch (const std::exception& e) {
        is_streaming_ = false;
        throw CameraException(std::string("Failed to start stream: ") + e.what());
    }
}

void EmergentCamera::configureDefaults() {
    // Only set the absolute minimum required for initialization
    // Do not set any resolution, exposure, gain etc. here
    
    checkError(EVT_CameraSetEnumParam(camera_.get(), "AcquisitionMode", "Continuous"), 
        "Setting acquisition mode");
    checkError(EVT_CameraSetUInt32Param(camera_.get(), "AcquisitionFrameCount", 1),
        "Setting frame count");
        
    // Add any other essential initialization settings that don't depend on ranges
    
    LOG(INFO) << "Camera defaults configured";
}

void EmergentCamera::checkError(EVT_ERROR err, const std::string& operation) const {
    if (err != EVT_SUCCESS) {
        throw CameraException(
            operation + " failed: " + getErrorString(err), 
            err
        );
    }
}

EmergentCamera::EmergentCamera(EmergentCamera&& other) noexcept 
    : params_(std::move(other.params_)),
      camera_(std::move(other.camera_)),
      frame_buffers_(std::move(other.frame_buffers_)),
      is_open_(other.is_open_),
      is_streaming_(other.is_streaming_) {
    
    // Reset other's state
    other.is_open_ = false;
    other.is_streaming_ = false;
}

EmergentCamera& EmergentCamera::operator=(EmergentCamera&& other) noexcept {
    if (this != &other) {
        // Clean up current resources first
        if (is_streaming_) {
            stopStream();
        }
        if (is_open_) {
            close();
        }

        // Move resources from other
        params_ = std::move(other.params_);
        camera_ = std::move(other.camera_);
        frame_buffers_ = std::move(other.frame_buffers_);
        is_open_ = other.is_open_;
        is_streaming_ = other.is_streaming_;

        // Reset other's state
        other.is_open_ = false;
        other.is_streaming_ = false;
    }
    return *this;
}

void EmergentCamera::stopStream() {
    if (!is_streaming_) {
        return; // Nothing to do
    }

    try {
        // Release any allocated frame buffers first
        releaseFrameBuffers();
        
        // Close the stream
        checkError(EVT_CameraCloseStream(camera_.get()), "Closing stream");
        
        is_streaming_ = false;
    } catch (const std::exception& e) {
        // Log the error but don't throw - this allows destructor to continue cleanup
        std::cerr << "Error stopping stream: " << e.what() << std::endl;
        is_streaming_ = false; // Ensure flag is reset even if error occurs
    }
}

void EmergentCamera::setParameter(const std::string& param, const std::string& value) {
    if (!is_open_) {
        throw CameraException("Camera not open");
    }

    try {
        if (param == "TriggerSource" || param == "AcquisitionMode" || 
            param == "TriggerMode" || param == "PtpMode" || 
            param == "TriggerSelector" || param == "PixelFormat" ||
            param == "ColorTemp") {
            checkError(EVT_CameraSetEnumParam(camera_.get(), param.c_str(), value.c_str()),
                      "Setting " + param);
        } else {
            throw CameraException("Unsupported string parameter: " + param);
        }
    } catch (const std::exception& e) {
        throw CameraException(std::string("Failed to set parameter ") + param + ": " + e.what());
    }
}

void EmergentCamera::setParameter(const std::string& param, int value) {
    if (!is_open_) {
        throw CameraException("Camera not open");
    }

    try {
        if (param == "Gain") {
            updateGain(value);
        } else if (param == "Exposure") {
            updateExposure(value);
        } else if (param == "FrameRate") {
            updateFrameRate(value);
        } else if (param == "Focus") {
            updateFocus(value);
        } else if (param == "Width") {
            updateResolution(value, params_.height);
        } else if (param == "Height") {
            updateResolution(params_.width, value);
        } else if (param == "OffsetX" || param == "OffsetY") {
            if (param == "OffsetX") {
                updateOffset(value, params_.offsety);
            } else {
                updateOffset(params_.offsetx, value);
            }
        } else if (param == "PtpAcquisitionGateTimeHigh") {
            checkError(EVT_CameraSetUInt32Param(camera_.get(), param.c_str(), 
                static_cast<uint32_t>(value)), "Setting " + param);
        } else if (param == "PtpAcquisitionGateTimeLow") {
            checkError(EVT_CameraSetUInt32Param(camera_.get(), param.c_str(), 
                static_cast<uint32_t>(value)), "Setting " + param);
        } else {
            throw CameraException("Unsupported integer parameter: " + param);
        }
    } catch (const std::exception& e) {
        throw CameraException(std::string("Failed to set parameter ") + param + ": " + e.what());
    }
}

void EmergentCamera::setParameter(const std::string& param, uint32_t value) {
    setParameter(param, static_cast<int>(value));
}

void EmergentCamera::setParameter(const std::string& param, bool value) {
    if (!is_open_) {
        throw CameraException("Camera not open");
    }

    try {
        if (param == "LUTEnable" || param == "AutoGain") {
            checkError(EVT_CameraSetBoolParam(camera_.get(), param.c_str(), value),
                      "Setting " + param);
        } else {
            throw CameraException("Unsupported boolean parameter: " + param);
        }
    } catch (const std::exception& e) {
        throw CameraException(std::string("Failed to set parameter ") + param + ": " + e.what());
    }
}

void EmergentCamera::allocateFrameBuffers(int buffer_size) {
    if (!is_open_) {
        throw CameraException("Cannot allocate buffers - camera not open");
    }
    if (buffer_size <= 0) {
        throw CameraException("Invalid buffer size: " + std::to_string(buffer_size));
    }

    // First release any existing buffers
    releaseFrameBuffers();

    try {
        frame_buffers_.resize(buffer_size);
        
        for (auto& frame : frame_buffers_) {
            // Configure frame format based on current settings
            setFrameBufferFormat(&frame);
            
            // Allocate the frame buffer
            checkError(EVT_AllocateFrameBuffer(
                camera_.get(), 
                &frame, 
                EVT_FRAME_BUFFER_ZERO_COPY
            ), "Allocating frame buffer");

            // Queue the frame
            checkError(EVT_CameraQueueFrame(
                camera_.get(), 
                &frame
            ), "Queueing frame buffer");
        }
    } catch (const std::exception& e) {
        // Clean up any allocated buffers on failure
        releaseFrameBuffers();
        throw;
    }
}

void EmergentCamera::releaseFrameBuffers() {
    if (!is_open_ || frame_buffers_.empty()) {
        return;
    }

    for (auto& frame : frame_buffers_) {
        try {
            EVT_ReleaseFrameBuffer(camera_.get(), &frame);
        } catch (const std::exception& e) {
            // Log error but continue releasing other buffers
            std::cerr << "Error releasing frame buffer: " << e.what() << std::endl;
        }
    }
    
    frame_buffers_.clear();
}

void EmergentCamera::queueFrame(Emergent::CEmergentFrame* frame) {
    if (!is_open_) {
        throw CameraException("Cannot queue frame - camera not open");
    }
    if (!frame) {
        throw CameraException("Invalid frame pointer");
    }

    checkError(EVT_CameraQueueFrame(camera_.get(), frame), 
        "Queueing frame");
}

bool EmergentCamera::getFrame(Emergent::CEmergentFrame* frame, int timeout_ms) {
    if (!is_open_ || !is_streaming_) {
        throw CameraException("Cannot get frame - camera not open or not streaming");
    }
    if (!frame) {
        throw CameraException("Invalid frame pointer");
    }

    EVT_ERROR err = EVT_CameraGetFrame(camera_.get(), frame, timeout_ms);
    
    if (err == EVT_ERROR_AGAIN) {
        return false; // Timeout
    }
    
    checkError(err, "Getting frame");
    return true;
}

void EmergentCamera::setFrameBufferFormat(Emergent::CEmergentFrame* frame) const {
    if (!frame) {
        throw CameraException("Invalid frame pointer");
    }

    frame->size_x = params_.width;
    frame->size_y = params_.height;

    // Set pixel format based on camera configuration
    if (params_.pixel_format == "BayerRG8") {
        frame->pixel_type = GVSP_PIX_BAYRG8;
    }
    else if (params_.pixel_format == "RGB8Packed") {
        frame->pixel_type = GVSP_PIX_RGB8;
    }
    else if (params_.pixel_format == "BGR8Packed") {
        frame->pixel_type = GVSP_PIX_BGR8;
    }
    else if (params_.pixel_format == "YUV411Packed") {
        frame->pixel_type = GVSP_PIX_YUV411_PACKED;
    }
    else if (params_.pixel_format == "YUV422Packed") {
        frame->pixel_type = GVSP_PIX_YUV422_PACKED;
    }
    else if (params_.pixel_format == "YUV444Packed") {
        frame->pixel_type = GVSP_PIX_YUV444_PACKED;
    }
    else if (params_.pixel_format == "BayerGB8") {
        frame->pixel_type = GVSP_PIX_BAYGB8;
    }
    else {
        // Default to mono8 for other formats
        frame->pixel_type = GVSP_PIX_MONO8;
    }
}

void EmergentCamera::updateExposure(int exposure_value) {
    if (!is_open_) {
        throw CameraException("Cannot update exposure - camera not open");
    }

    unsigned int max_val, min_val, inc_val;
    
    // Get valid range for exposure
    checkError(EVT_CameraGetUInt32ParamMax(camera_.get(), "Exposure", &max_val),
        "Getting max exposure");
    checkError(EVT_CameraGetUInt32ParamMin(camera_.get(), "Exposure", &min_val),
        "Getting min exposure");
    checkError(EVT_CameraGetUInt32ParamInc(camera_.get(), "Exposure", &inc_val),
        "Getting exposure increment");

    // Validate exposure value
    if (exposure_value < static_cast<int>(min_val) || 
        exposure_value > static_cast<int>(max_val)) {
        throw CameraException(
            "Exposure value " + std::to_string(exposure_value) + 
            " outside valid range [" + std::to_string(min_val) + 
            ", " + std::to_string(max_val) + "]"
        );
    }

    logCurrentState("Before exposure update");
    
    // Update exposure
    checkError(EVT_CameraSetUInt32Param(camera_.get(), "Exposure", exposure_value),
        "Setting exposure");
    
    params_.exposure = exposure_value;
    
    // After exposure update, get new frame rate limits as they're linked
    ParameterRange new_frame_rate_range = getFrameRateRange();
    
    // Adjust current frame rate if it exceeds new maximum
    if (params_.frame_rate > new_frame_rate_range.max) {
        LOG(INFO) << "Adjusting frame rate from " << params_.frame_rate 
                 << " to " << new_frame_rate_range.max << " due to exposure change";
        updateFrameRate(new_frame_rate_range.max);
    }
    
    logCurrentState("After exposure change");
}

void EmergentCamera::updateGain(int gain_value) {
    if (!is_open_) {
        throw CameraException("Cannot update gain - camera not open");
    }

    unsigned int max_val, min_val, inc_val;
    
    // Get valid range for gain
    checkError(EVT_CameraGetUInt32ParamMax(camera_.get(), "Gain", &max_val),
        "Getting max gain");
    checkError(EVT_CameraGetUInt32ParamMin(camera_.get(), "Gain", &min_val),
        "Getting min gain");
    checkError(EVT_CameraGetUInt32ParamInc(camera_.get(), "Gain", &inc_val),
        "Getting gain increment");

    // Validate gain value
    if (gain_value < static_cast<int>(min_val) || 
        gain_value > static_cast<int>(max_val)) {
        throw CameraException(
            "Gain value " + std::to_string(gain_value) + 
            " outside valid range [" + std::to_string(min_val) + 
            ", " + std::to_string(max_val) + "]"
        );
    }

    logCurrentState("Before gain update");

    // Update gain
    checkError(EVT_CameraSetUInt32Param(camera_.get(), "Gain", gain_value),
        "Setting gain");
    
    params_.gain = gain_value;

    logCurrentState("After gain update");
}

void EmergentCamera::updateFrameRate(int frame_rate) {
    if (!is_open_) {
        throw CameraException("Cannot update frame rate - camera not open");
    }

    auto range = getFrameRateRange();  // Get current valid range

    // Validate frame rate
    if (frame_rate < static_cast<int>(range.min) || 
        frame_rate > static_cast<int>(range.max)) {
        throw CameraException(
            "Frame rate " + std::to_string(frame_rate) + 
            " outside valid range [" + std::to_string(range.min) + 
            ", " + std::to_string(range.max) + "]"
        );
    }

    logCurrentState("Before frame rate update");

    // Update frame rate
    checkError(EVT_CameraSetUInt32Param(camera_.get(), "FrameRate", frame_rate),
        "Setting frame rate");
    
    params_.frame_rate = frame_rate;

    logCurrentState("After frame rate update");
}

void EmergentCamera::updateResolution(int width, int height) {
    if (!is_open_) {
        throw CameraException("Cannot update resolution - camera not open");
    }

    // Get valid ranges for width and height
    unsigned int width_max, width_min, width_inc;
    unsigned int height_max, height_min, height_inc;

    checkError(EVT_CameraGetUInt32ParamMax(camera_.get(), "Width", &width_max),
        "Getting max width");
    checkError(EVT_CameraGetUInt32ParamMin(camera_.get(), "Width", &width_min),
        "Getting min width");
    checkError(EVT_CameraGetUInt32ParamInc(camera_.get(), "Width", &width_inc),
        "Getting width increment");

    checkError(EVT_CameraGetUInt32ParamMax(camera_.get(), "Height", &height_max),
        "Getting max height");
    checkError(EVT_CameraGetUInt32ParamMin(camera_.get(), "Height", &height_min),
        "Getting min height");
    checkError(EVT_CameraGetUInt32ParamInc(camera_.get(), "Height", &height_inc),
        "Getting height increment");

    // Validate width
    if (width < static_cast<int>(width_min) || width > static_cast<int>(width_max)) {
        throw CameraException(
            "Width " + std::to_string(width) + 
            " outside valid range [" + std::to_string(width_min) + 
            ", " + std::to_string(width_max) + "]"
        );
    }

    // Validate height
    if (height < static_cast<int>(height_min) || height > static_cast<int>(height_max)) {
        throw CameraException(
            "Height " + std::to_string(height) + 
            " outside valid range [" + std::to_string(height_min) + 
            ", " + std::to_string(height_max) + "]"
        );
    }

    // Validate increments
    if ((width - width_min) % width_inc != 0) {
        throw CameraException(
            "Width must be in increments of " + std::to_string(width_inc) + 
            " from minimum " + std::to_string(width_min)
        );
    }

    if ((height - height_min) % height_inc != 0) {
        throw CameraException(
            "Height must be in increments of " + std::to_string(height_inc) + 
            " from minimum " + std::to_string(height_min)
        );
    }

    // Update width first, as it might affect valid height ranges
    checkError(EVT_CameraSetUInt32Param(camera_.get(), "Width", width),
        "Setting width");
    params_.width = width;

    // Then update height
    checkError(EVT_CameraSetUInt32Param(camera_.get(), "Height", height),
        "Setting height");
    params_.height = height;

    // After changing resolution, current offsets might be invalid
    // Reset to 0,0 and let caller explicitly set new offsets if needed
    updateOffset(0, 0);

    std::cout << "Resolution updated to " << width << "x" << height << std::endl;
}

void EmergentCamera::enablePTPSync() {
    if (!is_open_) {
        throw CameraException("Cannot enable PTP sync - camera not open");
    }

    try {
        // Configure PTP triggering settings
        checkError(EVT_CameraSetEnumParam(camera_.get(), "TriggerSource", "Software"),
            "Setting trigger source for PTP");
        checkError(EVT_CameraSetEnumParam(camera_.get(), "AcquisitionMode", "MultiFrame"),
            "Setting acquisition mode for PTP");
        checkError(EVT_CameraSetUInt32Param(camera_.get(), "AcquisitionFrameCount", 1),
            "Setting frame count for PTP");
        checkError(EVT_CameraSetEnumParam(camera_.get(), "TriggerMode", "On"),
            "Enabling trigger mode for PTP");
        checkError(EVT_CameraSetEnumParam(camera_.get(), "PtpMode", "TwoStep"),
            "Setting PTP mode");

        std::cout << "PTP synchronization enabled" << std::endl;
    } catch (const std::exception& e) {
        throw CameraException(std::string("Failed to enable PTP sync: ") + e.what());
    }
}

void EmergentCamera::disablePTPSync() {
    if (!is_open_) {
        throw CameraException("Cannot disable PTP sync - camera not open");
    }

    try {
        // Reset to default continuous mode settings
        checkError(EVT_CameraSetEnumParam(camera_.get(), "AcquisitionMode", "Continuous"),
            "Resetting acquisition mode");
        checkError(EVT_CameraSetUInt32Param(camera_.get(), "AcquisitionFrameCount", 1),
            "Resetting frame count");
        checkError(EVT_CameraSetEnumParam(camera_.get(), "TriggerSelector", "AcquisitionStart"),
            "Resetting trigger selector");
        checkError(EVT_CameraSetEnumParam(camera_.get(), "TriggerMode", "Off"),
            "Disabling trigger mode");
        checkError(EVT_CameraSetEnumParam(camera_.get(), "TriggerSource", "Software"),
            "Resetting trigger source");

        std::cout << "PTP synchronization disabled" << std::endl;
    } catch (const std::exception& e) {
        throw CameraException(std::string("Failed to disable PTP sync: ") + e.what());
    }
}

uint64_t EmergentCamera::updatePTPGateTime(unsigned int high, unsigned int low) {
    checkError(EVT_CameraSetUInt32Param(camera_.get(), "PtpAcquisitionGateTimeHigh", high),
        "Setting PTP gate time high");
    checkError(EVT_CameraSetUInt32Param(camera_.get(), "PtpAcquisitionGateTimeLow", low),
        "Setting PTP gate time low");
    }

void EmergentCamera::getPTPStatus(char* status, size_t size, unsigned long* ret_size) const {
    checkError(EVT_CameraGetEnumParam(camera_.get(), "PtpStatus", 
            status, size, ret_size), "Getting PTP status");
    }

int32_t EmergentCamera::getPTPOffset() const {
    int32_t offset;
    checkError(EVT_CameraGetInt32Param(camera_.get(), "PtpOffset", &offset),
        "Getting PTP offset");
    return offset;
}

uint64_t EmergentCamera::getCurrentPTPTime() const {
    if (!is_open_) {
        throw CameraException("Cannot get PTP time - camera not open");
    }

    try {
        // Check PTP status first
        char ptp_status[100];
        unsigned long ptp_status_sz_ret;
        checkError(EVT_CameraGetEnumParam(camera_.get(), "PtpStatus", 
            ptp_status, sizeof(ptp_status), &ptp_status_sz_ret),
            "Getting PTP status");

        // Get current PTP timestamp
        checkError(EVT_CameraExecuteCommand(camera_.get(), "GevTimestampControlLatch"),
            "Latching timestamp");

        unsigned int ptp_time_high, ptp_time_low;
        checkError(EVT_CameraGetUInt32Param(camera_.get(), "GevTimestampValueHigh", &ptp_time_high),
            "Getting timestamp high value");
        checkError(EVT_CameraGetUInt32Param(camera_.get(), "GevTimestampValueLow", &ptp_time_low),
            "Getting timestamp low value");

        // Combine high and low values into 64-bit timestamp
        uint64_t ptp_time = (static_cast<uint64_t>(ptp_time_high) << 32) | 
                            static_cast<uint64_t>(ptp_time_low);

        return ptp_time;
    } catch (const std::exception& e) {
        throw CameraException(std::string("Failed to get PTP time: ") + e.what());
    }
}

void EmergentCamera::updateOffset(int offset_x, int offset_y) {
    if (!is_open_) {
        throw CameraException("Cannot update offset - camera not open");
    }

    try {
        // Get valid ranges for X and Y offsets
        // These ranges depend on current resolution settings
        unsigned int offsetx_max, offsetx_min, offsetx_inc;
        unsigned int offsety_max, offsety_min, offsety_inc;

        // Get X offset parameters
        checkError(EVT_CameraGetUInt32ParamMax(camera_.get(), "OffsetX", &offsetx_max),
            "Getting max X offset");
        checkError(EVT_CameraGetUInt32ParamMin(camera_.get(), "OffsetX", &offsetx_min),
            "Getting min X offset");
        checkError(EVT_CameraGetUInt32ParamInc(camera_.get(), "OffsetX", &offsetx_inc),
            "Getting X offset increment");

        // Store in params for future reference
        params_.offsetx_max = offsetx_max;
        params_.offsetx_min = offsetx_min;
        params_.offsetx_inc = offsetx_inc;

        // Get Y offset parameters
        checkError(EVT_CameraGetUInt32ParamMax(camera_.get(), "OffsetY", &offsety_max),
            "Getting max Y offset");
        checkError(EVT_CameraGetUInt32ParamMin(camera_.get(), "OffsetY", &offsety_min),
            "Getting min Y offset");
        checkError(EVT_CameraGetUInt32ParamInc(camera_.get(), "OffsetY", &offsety_inc),
            "Getting Y offset increment");

        // Store in params for future reference
        params_.offsety_max = offsety_max;
        params_.offsety_min = offsety_min;
        params_.offsety_inc = offsety_inc;

        // Validate X offset
        if (offset_x < static_cast<int>(offsetx_min) || offset_x > static_cast<int>(offsetx_max)) {
            throw CameraException(
                "X offset " + std::to_string(offset_x) + 
                " outside valid range [" + std::to_string(offsetx_min) + 
                ", " + std::to_string(offsetx_max) + "]"
            );
        }

        // Validate Y offset
        if (offset_y < static_cast<int>(offsety_min) || offset_y > static_cast<int>(offsety_max)) {
            throw CameraException(
                "Y offset " + std::to_string(offset_y) + 
                " outside valid range [" + std::to_string(offsety_min) + 
                ", " + std::to_string(offsety_max) + "]"
            );
        }

        // Validate increments
        if ((offset_x - offsetx_min) % offsetx_inc != 0) {
            throw CameraException(
                "X offset must be in increments of " + std::to_string(offsetx_inc) + 
                " from minimum " + std::to_string(offsetx_min)
            );
        }

        if ((offset_y - offsety_min) % offsety_inc != 0) {
            throw CameraException(
                "Y offset must be in increments of " + std::to_string(offsety_inc) + 
                " from minimum " + std::to_string(offsety_min)
            );
        }

        // Update X offset first
        checkError(EVT_CameraSetUInt32Param(camera_.get(), "OffsetX", offset_x),
            "Setting X offset");
        params_.offsetx = offset_x; 

        // Then update Y offset
        checkError(EVT_CameraSetUInt32Param(camera_.get(), "OffsetY", offset_y),
            "Setting Y offset");
        params_.offsety = offset_y;

        std::cout << "Offset updated to (" << offset_x << "," << offset_y << ")" << std::endl;

    } catch (const std::exception& e) {
        throw CameraException(std::string("Failed to update offset: ") + e.what());
    }
}

void EmergentCamera::updateIris(int iris_value) {
    if (!is_open_) {
        throw CameraException("Cannot update iris - camera not open");
    }

    try {
        // Get valid ranges for iris control
        unsigned int max_val, min_val, inc_val;
        
        // Get iris parameters
        checkError(EVT_CameraGetUInt32ParamMax(camera_.get(), "Iris", &max_val),
            "Getting max iris value");
        checkError(EVT_CameraGetUInt32ParamMin(camera_.get(), "Iris", &min_val),
            "Getting min iris value");
        checkError(EVT_CameraGetUInt32ParamInc(camera_.get(), "Iris", &inc_val),
            "Getting iris increment");

        // Store in params for future reference
        params_.iris_max = max_val;
        params_.iris_min = min_val;
        params_.iris_inc = inc_val;

        // Validate iris value
        if (iris_value < static_cast<int>(min_val) || iris_value > static_cast<int>(max_val)) {
            throw CameraException(
                "Iris value " + std::to_string(iris_value) + 
                " outside valid range [" + std::to_string(min_val) + 
                ", " + std::to_string(max_val) + "]"
            );
        }

        // Validate increment
        if ((iris_value - min_val) % inc_val != 0) {
            throw CameraException(
                "Iris value must be in increments of " + std::to_string(inc_val) + 
                " from minimum " + std::to_string(min_val)
            );
        }

        logCurrentState("Before iris update");

        // Update iris value
        checkError(EVT_CameraSetUInt32Param(camera_.get(), "Iris", iris_value),
            "Setting iris value");
        params_.iris = iris_value;

        logCurrentState("After iris update");

        std::cout << "Iris value updated to " << iris_value << std::endl;

    } catch (const std::exception& e) {
        throw CameraException(std::string("Failed to update iris: ") + e.what());
    }
}

void EmergentCamera::updateFocus(int focus_value) {
    if (!is_open_) {
        throw CameraException("Cannot update focus - camera not open");
    }

    try {
        // Get valid range for focus
        unsigned int max_val, min_val, inc_val;
        
        checkError(EVT_CameraGetUInt32ParamMax(camera_.get(), "Focus", &max_val),
            "Getting max focus value");
        checkError(EVT_CameraGetUInt32ParamMin(camera_.get(), "Focus", &min_val),
            "Getting min focus value");
        checkError(EVT_CameraGetUInt32ParamInc(camera_.get(), "Focus", &inc_val),
            "Getting focus increment");

        // Store in params for future reference
        params_.focus_max = max_val;
        params_.focus_min = min_val;
        params_.focus_inc = inc_val;

        // Validate focus value
        if (focus_value < static_cast<int>(min_val) || 
            focus_value > static_cast<int>(max_val)) {
            throw CameraException(
                "Focus value " + std::to_string(focus_value) + 
                " outside valid range [" + std::to_string(min_val) + 
                ", " + std::to_string(max_val) + "]"
            );
        }

        logCurrentState("Before focus update");

        // Update focus
        checkError(EVT_CameraSetUInt32Param(camera_.get(), "Focus", focus_value),
            "Setting focus value");
        
        params_.focus = focus_value;

        logCurrentState("After focus update");

    } catch (const std::exception& e) {
        throw CameraException(std::string("Failed to update focus: ") + e.what());
    }
}

int EmergentCamera::getSensorTemperature() const {
    if (!is_open_) {
        throw CameraException("Cannot get sensor temperature - camera not open");
    }

    try {
        // Get temperature range if not already stored
        if (params_.sens_temp_max == 0 && params_.sens_temp_min == 0) {
            int temp_max, temp_min;
            checkError(EVT_CameraGetInt32ParamMax(camera_.get(), "SensTemp", &temp_max),
                "Getting max sensor temperature");
            checkError(EVT_CameraGetInt32ParamMin(camera_.get(), "SensTemp", &temp_min),
                "Getting min sensor temperature");
            
            // Store in params for future reference
            params_.sens_temp_max = temp_max;
            params_.sens_temp_min = temp_min;
        }

        // Get current temperature
        int current_temp;
        checkError(EVT_CameraGetInt32Param(camera_.get(), "SensTemp", &current_temp),
            "Getting current sensor temperature");

        // Update stored temperature
        params_.sens_temp = current_temp;

        // Validate temperature is within expected range
        if (current_temp < params_.sens_temp_min || current_temp > params_.sens_temp_max) {
            std::cerr << "Warning: Sensor temperature " << current_temp 
                      << "°C outside expected range [" 
                      << params_.sens_temp_min << "°C, "
                      << params_.sens_temp_max << "°C]" << std::endl;
        }

        return current_temp;

    } catch (const std::exception& e) {
        throw CameraException(std::string("Failed to get sensor temperature: ") + e.what());
    }
}

void EmergentCamera::printDeviceInfo(const GigEVisionDeviceInfo* device_info) const {
    if (!device_info) {
        throw CameraException("Cannot print device info - device_info is null");
    }
    
    std::cout << "Camera Info:" << std::endl;
    std::cout << "userDefinedName: " << device_info->userDefinedName << std::endl;
    std::cout << "macAddress: " << device_info->macAddress << std::endl;
    std::cout << "deviceMode: " << device_info->deviceMode << std::endl;
    std::cout << "serialNumber: " << device_info->serialNumber << std::endl;
    std::cout << "currentIp: " << device_info->currentIp << std::endl;
    std::cout << "currentSubnetMask: " << device_info->currentSubnetMask << std::endl;
    std::cout << "defaultGateway: " << device_info->defaultGateway << std::endl;
    std::cout << "nic.ip4Address: " << device_info->nic.ip4Address << std::endl;
}

bool EmergentCamera::initializeGPUDirect() {
    try {
        auto& gpu_manager = GPUManager::getInstance();
        
        // Initialize GPU system
        gpu_manager.initialize();
        
        // Validate GPU configuration
        validateGPUConfiguration();
        
        // Select the specified GPU
        gpu_manager.selectDevice(params_.gpu_id);
        
        // Verify compatibility with this specific camera
        if (!gpu_manager.verifyGPUDirectCompatibility(params_.camera_serial)) {
            std::cerr << "GPU Direct not compatible with this configuration" << std::endl;
            return false;
        }
        
        // Configure camera for GPU Direct
        camera_->gpuDirectDeviceId = params_.gpu_id;
        
        std::cout << "GPU Direct successfully initialized on GPU " 
                  << params_.gpu_id << std::endl;
                  
        return true;
        
    } catch (const GPUException& e) {
        std::cerr << "GPU initialization failed: " << e.what() << std::endl;
        if (e.getErrorCode() != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(e.getErrorCode()) << std::endl;
        }
        return false;
    }
}

void EmergentCamera::validateGPUConfiguration() const {
    auto& gpu_manager = GPUManager::getInstance();
    
    // Get list of available devices
    auto devices = gpu_manager.getAvailableDevices();
    
    // Validate GPU ID
    if (params_.gpu_id >= static_cast<int>(devices.size())) {
        throw CameraException(
            "Invalid GPU ID " + std::to_string(params_.gpu_id) + 
            ". Maximum available ID is " + std::to_string(devices.size() - 1)
        );
    }
    
    // Check if selected GPU supports GPU Direct
    if (!gpu_manager.isGPUDirectSupported(params_.gpu_id)) {
        throw CameraException(
            "Selected GPU (ID " + std::to_string(params_.gpu_id) + 
            ") does not support GPU Direct"
        );
    }
    
    // Log GPU configuration
    const auto& selected_gpu = devices[params_.gpu_id];
    std::cout << "Selected GPU Configuration:" << std::endl
              << "- Name: " << selected_gpu.name << std::endl
              << "- Compute Capability: " 
              << selected_gpu.compute_capability_major << "."
              << selected_gpu.compute_capability_minor << std::endl
              << "- Total Memory: " 
              << (selected_gpu.total_memory / 1024 / 1024) << "MB" << std::endl
              << "- Free Memory: "
              << (selected_gpu.free_memory / 1024 / 1024) << "MB" << std::endl;
}

void EmergentCamera::updatePixelFormat(const std::string& format) {
    if (!is_open_) {
        throw CameraException("Cannot update pixel format - camera not open");
    }

    checkError(EVT_CameraSetEnumParam(camera_.get(), "PixelFormat", format.c_str()),
        "Setting pixel format");
    
    params_.pixel_format = format;
}

void EmergentCamera::close() {
    if (!is_open_) {
        return;  // Nothing to do if camera isn't open
    }

    try {
        checkError(EVT_CameraClose(camera_.get()), "Closing camera");
        is_open_ = false;
        std::cout << "Camera " << params_.camera_serial << " closed successfully" << std::endl;
    } catch (const std::exception& e) {
        // Log error but don't throw - this allows destructor to continue cleanup
        std::cerr << "Error closing camera: " << e.what() << std::endl;
        is_open_ = false;  // Ensure flag is reset even if error occurs
    }
}

evt::EmergentCamera::ParameterRange evt::EmergentCamera::getExposureRange() const {
    ParameterRange range;
    if (!is_open_) {
        throw CameraException("Cannot get exposure range - camera not open");
    }
    
    checkError(EVT_CameraGetUInt32ParamMax(camera_.get(), "Exposure", &range.max),
        "Getting max exposure");
    checkError(EVT_CameraGetUInt32ParamMin(camera_.get(), "Exposure", &range.min),
        "Getting min exposure");
    checkError(EVT_CameraGetUInt32ParamInc(camera_.get(), "Exposure", &range.increment),
        "Getting exposure increment");
        
    return range;
}

evt::EmergentCamera::ParameterRange evt::EmergentCamera::getGainRange() const {
    ParameterRange range;
    if (!is_open_) {
        throw CameraException("Cannot get gain range - camera not open");
    }
    
    checkError(EVT_CameraGetUInt32ParamMax(camera_.get(), "Gain", &range.max),
        "Getting max gain");
    checkError(EVT_CameraGetUInt32ParamMin(camera_.get(), "Gain", &range.min),
        "Getting min gain");
    checkError(EVT_CameraGetUInt32ParamInc(camera_.get(), "Gain", &range.increment),
        "Getting gain increment");
        
    return range;
}

EmergentCamera::ParameterRange EmergentCamera::getFrameRateRange() const {
    ParameterRange range;
    if (!is_open_) {
        throw CameraException("Cannot get frame rate range - camera not open");
    }
    
    checkError(EVT_CameraGetUInt32ParamMax(camera_.get(), "FrameRate", &range.max),
        "Getting max frame rate");
    checkError(EVT_CameraGetUInt32ParamMin(camera_.get(), "FrameRate", &range.min),
        "Getting min frame rate");
    checkError(EVT_CameraGetUInt32ParamInc(camera_.get(), "FrameRate", &range.increment),
        "Getting frame rate increment");
        
    return range;
}

evt::EmergentCamera::ParameterRange evt::EmergentCamera::getFocusRange() const {
    ParameterRange range;
    if (!is_open_) {
        throw CameraException("Cannot get focus range - camera not open");
    }
    
    checkError(EVT_CameraGetUInt32ParamMax(camera_.get(), "Focus", &range.max),
        "Getting max focus");
    checkError(EVT_CameraGetUInt32ParamMin(camera_.get(), "Focus", &range.min),
        "Getting min focus");
    checkError(EVT_CameraGetUInt32ParamInc(camera_.get(), "Focus", &range.increment),
        "Getting focus increment");
        
    return range;
}

evt::EmergentCamera::ParameterRange evt::EmergentCamera::getIrisRange() const {
    ParameterRange range;
    if (!is_open_) {
        throw CameraException("Cannot get iris range - camera not open");
    }
    
    checkError(EVT_CameraGetUInt32ParamMax(camera_.get(), "Iris", &range.max),
        "Getting max iris");
    checkError(EVT_CameraGetUInt32ParamMin(camera_.get(), "Iris", &range.min),
        "Getting min iris");
    checkError(EVT_CameraGetUInt32ParamInc(camera_.get(), "Iris", &range.increment),
        "Getting iris increment");
        
    return range;
}

evt::EmergentCamera::ResolutionRange evt::EmergentCamera::getResolutionRange() const {
    ResolutionRange range;
    if (!is_open_) {
        throw CameraException("Cannot get resolution range - camera not open");
    }
    
    checkError(EVT_CameraGetUInt32ParamMax(camera_.get(), "Width", &range.width_max),
        "Getting max width");
    checkError(EVT_CameraGetUInt32ParamMin(camera_.get(), "Width", &range.width_min),
        "Getting min width");
    checkError(EVT_CameraGetUInt32ParamInc(camera_.get(), "Width", &range.width_inc),
        "Getting width increment");
        
    checkError(EVT_CameraGetUInt32ParamMax(camera_.get(), "Height", &range.height_max),
        "Getting max height");
    checkError(EVT_CameraGetUInt32ParamMin(camera_.get(), "Height", &range.height_min),
        "Getting min height");
    checkError(EVT_CameraGetUInt32ParamInc(camera_.get(), "Height", &range.height_inc),
        "Getting height increment");
        
    return range;
}

evt::EmergentCamera::TemperatureRange evt::EmergentCamera::getTemperatureRange() const {
    TemperatureRange range;
    if (!is_open_) {
        throw CameraException("Cannot get temperature range - camera not open");
    }
    
    checkError(EVT_CameraGetInt32ParamMax(camera_.get(), "SensTemp", &range.max),
        "Getting max temperature");
    checkError(EVT_CameraGetInt32ParamMin(camera_.get(), "SensTemp", &range.min),
        "Getting min temperature");
        
    return range;
}

EmergentCamera::CameraState EmergentCamera::getCurrentState() const {
    if (!is_open_) {
        throw CameraException("Cannot get camera state - camera not open");
    }

    CameraState state;
    checkError(EVT_CameraGetUInt32Param(camera_.get(), "Exposure", 
        reinterpret_cast<unsigned int*>(&state.exposure)), "Getting current exposure");
    checkError(EVT_CameraGetUInt32Param(camera_.get(), "Gain", 
        reinterpret_cast<unsigned int*>(&state.gain)), "Getting current gain");
    checkError(EVT_CameraGetUInt32Param(camera_.get(), "FrameRate", 
        reinterpret_cast<unsigned int*>(&state.frame_rate)), "Getting current frame rate");
    checkError(EVT_CameraGetUInt32Param(camera_.get(), "Iris", 
        reinterpret_cast<unsigned int*>(&state.iris)), "Getting current iris");
    checkError(EVT_CameraGetUInt32Param(camera_.get(), "Focus", 
        reinterpret_cast<unsigned int*>(&state.focus)), "Getting current focus");
        
    return state;
}

void EmergentCamera::logCurrentState(const std::string& context) const {
    try {
        auto state = getCurrentState();
        LOG(INFO) << "\n=== CAMERA STATE: " << context << " ===\n"
                  << "  │ Exposure:   " << state.exposure << "\n"
                  << "  │ Gain:       " << state.gain << "\n"
                  << "  │ Frame Rate: " << state.frame_rate << "\n"
                  << "  │ Iris:       " << state.iris << "\n"
                  << "  │ Focus:      " << state.focus << "\n"
                  << "  ╰────────────────────────────────";
    } catch (const CameraException& e) {
        LOG(ERROR) << "Failed to get camera state: " << e.what();
    }
}

void evt::EmergentCamera::updateCameraRanges() {
    if (!is_open_) {
        throw CameraException("Cannot update ranges - camera not open");
    }

    // Width ranges
    checkError(EVT_CameraGetUInt32ParamMax(camera_.get(), "Width", &params_.width_max),
        "Getting width max");
    checkError(EVT_CameraGetUInt32ParamMin(camera_.get(), "Width", &params_.width_min),
        "Getting width min");
    checkError(EVT_CameraGetUInt32ParamInc(camera_.get(), "Width", &params_.width_inc),
        "Getting width increment");

    // Height ranges
    checkError(EVT_CameraGetUInt32ParamMax(camera_.get(), "Height", &params_.height_max),
        "Getting height max");
    checkError(EVT_CameraGetUInt32ParamMin(camera_.get(), "Height", &params_.height_min),
        "Getting height min");
    checkError(EVT_CameraGetUInt32ParamInc(camera_.get(), "Height", &params_.height_inc),
        "Getting height increment");

    // Exposure ranges
    checkError(EVT_CameraGetUInt32ParamMax(camera_.get(), "Exposure", &params_.exposure_max),
        "Getting exposure max");
    checkError(EVT_CameraGetUInt32ParamMin(camera_.get(), "Exposure", &params_.exposure_min),
        "Getting exposure min");
    checkError(EVT_CameraGetUInt32ParamInc(camera_.get(), "Exposure", &params_.exposure_inc),
        "Getting exposure increment");

    // Gain ranges
    checkError(EVT_CameraGetUInt32ParamMax(camera_.get(), "Gain", &params_.gain_max),
        "Getting gain max");
    checkError(EVT_CameraGetUInt32ParamMin(camera_.get(), "Gain", &params_.gain_min),
        "Getting gain min");
    checkError(EVT_CameraGetUInt32ParamInc(camera_.get(), "Gain", &params_.gain_inc),
        "Getting gain increment");

    // Frame rate ranges
    checkError(EVT_CameraGetUInt32ParamMax(camera_.get(), "FrameRate", &params_.frame_rate_max),
        "Getting frame rate max");
    checkError(EVT_CameraGetUInt32ParamMin(camera_.get(), "FrameRate", &params_.frame_rate_min),
        "Getting frame rate min");
    checkError(EVT_CameraGetUInt32ParamInc(camera_.get(), "FrameRate", &params_.frame_rate_inc),
        "Getting frame rate increment");

    // Focus ranges
    checkError(EVT_CameraGetUInt32ParamMax(camera_.get(), "Focus", &params_.focus_max),
        "Getting focus max");
    checkError(EVT_CameraGetUInt32ParamMin(camera_.get(), "Focus", &params_.focus_min),
        "Getting focus min");
    checkError(EVT_CameraGetUInt32ParamInc(camera_.get(), "Focus", &params_.focus_inc),
        "Getting focus increment");

    // Iris ranges
    checkError(EVT_CameraGetUInt32ParamMax(camera_.get(), "Iris", &params_.iris_max),
        "Getting iris max");
    checkError(EVT_CameraGetUInt32ParamMin(camera_.get(), "Iris", &params_.iris_min),
        "Getting iris min");
    checkError(EVT_CameraGetUInt32ParamInc(camera_.get(), "Iris", &params_.iris_inc),
        "Getting iris increment");

    // Temperature ranges
    checkError(EVT_CameraGetInt32ParamMax(camera_.get(), "SensTemp", &params_.sens_temp_max),
        "Getting temperature max");
    checkError(EVT_CameraGetInt32ParamMin(camera_.get(), "SensTemp", &params_.sens_temp_min),
        "Getting temperature min");

    LOG(INFO) << "Updated camera parameter ranges:"
              << "\nWidth: " << params_.width_min << " - " << params_.width_max
              << "\nHeight: " << params_.height_min << " - " << params_.height_max
              << "\nExposure: " << params_.exposure_min << " - " << params_.exposure_max
              << "\nGain: " << params_.gain_min << " - " << params_.gain_max
              << "\nFrame Rate: " << params_.frame_rate_min << " - " << params_.frame_rate_max;
}

} // namespace evt