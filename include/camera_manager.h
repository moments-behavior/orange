#pragma once

#include <vector>
#include <memory>
#include <string>
#include <cstdint>
#include <algorithm>
#include "emergent_camera.h"
#include "frame_streaming.h"
#include "gpu_streaming.h"
#include "camera_params.h"

namespace evt {

class CameraManager {
public:
    // Structure to hold camera instance and its associated objects
    struct CameraInstance {
        std::unique_ptr<EmergentCamera> camera;
        std::unique_ptr<GPUStreaming> gpu_stream;
        GPUStreaming::StreamingConfig config;
        CameraParams params;
        bool is_streaming{false};
        bool is_recording{false};
    };

    CameraManager() = default;
    ~CameraManager() = default;

    // Initialize cameras based on selected devices
    void initializeCameras(const std::vector<bool>& selected_cameras,
                          const std::vector<GigEVisionDeviceInfo>& device_info,
                          const std::vector<std::string>& config_files,
                          std::unordered_map<std::string, CameraParams>& known_cameras) {
        cameras.clear();
        cameras.reserve(device_info.size());
        
        LOG(INFO) << "Starting camera initialization with " << device_info.size() << " devices";
        
        for (size_t i = 0; i < device_info.size(); ++i) {
            if (!selected_cameras[i]) {
                LOG(INFO) << "Camera " << i << " not selected, skipping";
                continue;
            }

            std::string serial(device_info[i].serialNumber);
            LOG(INFO) << "Initializing camera " << i << " (serial: " << serial << ")";
            
            CameraInstance instance;
            
            // Set identification parameters
            instance.params.device_info = device_info[i];
            instance.params.camera_serial = serial;
            instance.params.camera_name = device_info[i].userDefinedName;
            
            try {
                instance.camera = std::make_unique<EmergentCamera>(instance.params);
                LOG(INFO) << "Camera instance created for " << serial;
                
                instance.camera->open(&device_info[i]);
                LOG(INFO) << "Camera " << serial << " opened successfully";
                
                // First get current camera state
                instance.camera->updateParamsFromCurrentState(instance.params);
                instance.camera->logCurrentState("Initial camera state");
                
                LOG(INFO) << "Current camera settings:"
                         << "\n  Exposure: " << instance.params.exposure
                         << "\n  Gain: " << instance.params.gain
                         << "\n  Frame Rate: " << instance.params.frame_rate
                         << "\n  Iris: " << instance.params.iris
                         << "\n  Focus: " << instance.params.focus;

                // Now apply user config if it exists
                auto config_it = known_cameras.find(serial);
                if (config_it != known_cameras.end()) {
                    LOG(INFO) << "Applying user configuration for camera " << serial;
                    
                    // Apply each parameter with logging
                    if (config_it->second.exposure != instance.params.exposure) {
                        instance.camera->updateExposure(config_it->second.exposure);
                    }
                    if (config_it->second.gain != instance.params.gain) {
                        instance.camera->updateGain(config_it->second.gain);
                    }
                    if (config_it->second.frame_rate != instance.params.frame_rate) {
                        instance.camera->updateFrameRate(config_it->second.frame_rate);
                    }
                    if (config_it->second.iris != instance.params.iris) {
                        instance.camera->updateIris(config_it->second.iris);
                    }
                    if (config_it->second.focus != instance.params.focus) {
                        instance.camera->updateFocus(config_it->second.focus);
                    }

                    // Get final state after applying config
                    instance.camera->updateParamsFromCurrentState(instance.params);
                    instance.camera->logCurrentState("After applying user config");
                }

                cameras.push_back(std::move(instance));
                LOG(INFO) << "Successfully initialized camera " << serial;
                
            } catch (const evt::CameraException& e) {
                LOG(ERROR) << "Failed to initialize camera " << serial << ": " << e.what();
            }
        }
        
        LOG(INFO) << "Finished initialization, active camera count: " << cameras.size();
    }

    // Start/stop streaming for specific camera
    void startStreaming(size_t camera_idx, bool enable_gpu = false) {
        if (camera_idx >= cameras.size()) return;
        
        auto& instance = cameras[camera_idx];
        if (instance.is_streaming) return;

        try {
            // Open the camera if not already open
            if (!instance.camera->isOpen()) {
                const GigEVisionDeviceInfo* device_info = nullptr;
                instance.camera->open(device_info);
                
                // Initialize camera with stored parameters
                instance.camera->updateExposure(instance.params.exposure);
                instance.camera->updateGain(instance.params.gain);
                instance.camera->updateFrameRate(instance.params.frame_rate);
                instance.camera->updateIris(instance.params.iris);
                instance.camera->updateFocus(instance.params.focus);
                
                LOG(INFO) << "Camera initialized with:"
                         << "\n  Exposure: " << instance.params.exposure
                         << "\n  Gain: " << instance.params.gain
                         << "\n  Frame Rate: " << instance.params.frame_rate
                         << "\n  Iris: " << instance.params.iris
                         << "\n  Focus: " << instance.params.focus;
            }

            // Start the actual stream
            instance.camera->startStream();

            // Configure GPU if needed
            if (enable_gpu) {
                // Configure GPU streaming
                instance.config.enable_gpu_direct = true;
                instance.config.gpu_device_id = 0; // Would come from settings
                instance.gpu_stream = std::make_unique<GPUStreaming>(
                    *instance.camera,
                    instance.config
                );
                
                // Start GPU streaming
                instance.gpu_stream->startStreaming([](const void* data, 
                                                   size_t size,
                                                   int width, 
                                                   int height,
                                                   uint64_t timestamp) {
                    // Frame callback - integrate with GUI display
                });
            }
            
            instance.is_streaming = true;
            LOG(INFO) << "Camera streaming started successfully";
        }
        catch (const CameraException& e) {
            LOG(ERROR) << "Failed to start streaming: " << e.what();
            throw;
        }
    }

    void stopStreaming(size_t camera_idx) {
        if (camera_idx >= cameras.size()) return;
        
        auto& instance = cameras[camera_idx];
        if (!instance.is_streaming) return;

        try {
            if (instance.gpu_stream) {
                instance.gpu_stream->stopStreaming();
            }
            
            instance.is_streaming = false;
        }
        catch (const CameraException& e) {
            std::cerr << "Error stopping stream: " << e.what() << std::endl;
            throw;
        }
    }

    // Start/stop recording
    void startRecording(const std::string& output_folder,
                       const std::string& encoder_setup) {
        if (recording_active) return;
        
        try {
            for (auto& instance : cameras) {
                if (instance.is_streaming) {
                    // Configure recording
                    instance.config.enable_encoding = true;
                    instance.config.output_folder = output_folder;
                    instance.config.encoder_setup = encoder_setup;
                    
                    // Restart streaming with recording enabled
                    if (instance.gpu_stream) {
                        instance.gpu_stream->stopStreaming();
                        setupGPUStreaming(instance, output_folder, encoder_setup);
                        instance.gpu_stream->startStreaming([](const void* data, 
                                                           size_t size,
                                                           int width, 
                                                           int height,
                                                           uint64_t timestamp) {
                            // Frame callback for recording
                        });
                    }
                }
            }
            
            recording_active = true;
        }
        catch (const CameraException& e) {
            std::cerr << "Failed to start recording: " << e.what() << std::endl;
            throw;
        }
    }

    void stopRecording() {
        if (!recording_active) return;
        
        try {
            for (auto& instance : cameras) {
                if (instance.is_streaming && instance.gpu_stream) {
                    // Restart streaming without recording
                    instance.gpu_stream->stopStreaming();
                    instance.config.enable_encoding = false;
                    setupGPUStreaming(instance, "", "");
                    instance.gpu_stream->startStreaming([](const void* data, 
                                                       size_t size,
                                                       int width, 
                                                       int height,
                                                       uint64_t timestamp) {
                        // Frame callback for display only
                    });
                }
            }
            
            recording_active = false;
        }
        catch (const CameraException& e) {
            std::cerr << "Error stopping recording: " << e.what() << std::endl;
            throw;
        }
    }

    // Camera control methods
    void updateExposure(size_t camera_idx, int exposure) {
        if (camera_idx >= cameras.size()) return;
        cameras[camera_idx].camera->updateExposure(exposure);
    }

    void updateGain(size_t camera_idx, int gain) {
        if (camera_idx >= cameras.size()) return;
        cameras[camera_idx].camera->updateGain(gain);
    }

    void updateFrameRate(size_t camera_idx, int frame_rate) {
        if (camera_idx >= cameras.size()) return;
        cameras[camera_idx].camera->updateFrameRate(frame_rate);
    }

    void updateIris(size_t camera_idx, int iris) {
        if (camera_idx >= cameras.size()) return;
        cameras[camera_idx].camera->updateIris(iris);
    }

    void updateFocus(size_t camera_idx, int focus) {
        if (camera_idx >= cameras.size()) return;
        cameras[camera_idx].camera->updateFocus(focus);
    }

    evt::EmergentCamera::ParameterRange getExposureRange(size_t camera_idx) const {
        if (camera_idx >= cameras.size()) {
            throw CameraException("Invalid camera index");
        }
        return cameras[camera_idx].camera->getExposureRange();
    }
    
    evt::EmergentCamera::ParameterRange getGainRange(size_t camera_idx) const {
        if (camera_idx >= cameras.size()) {
            throw CameraException("Invalid camera index");
        }
        return cameras[camera_idx].camera->getGainRange();
    }
    
    evt::EmergentCamera::ParameterRange getFrameRateRange(size_t camera_idx) const {
        if (camera_idx >= cameras.size()) {
            throw CameraException("Invalid camera index");
        }
        return cameras[camera_idx].camera->getFrameRateRange();
    }

    evt::EmergentCamera::ParameterRange getIrisRange(size_t camera_idx) const {
        if (camera_idx >= cameras.size()) {
            throw CameraException("Invalid camera index");
        }
        return cameras[camera_idx].camera->getIrisRange();
    }
    
    evt::EmergentCamera::ParameterRange getFocusRange(size_t camera_idx) const {
        if (camera_idx >= cameras.size()) {
            throw CameraException("Invalid camera index");
        }
        return cameras[camera_idx].camera->getFocusRange();
    }

    // Status queries
    size_t getCameraCount() const { return cameras.size(); }
    bool isStreaming(size_t camera_idx) const { 
        return camera_idx < cameras.size() && cameras[camera_idx].is_streaming;
    }
    bool isRecording() const { return recording_active; }
    
    // Access to camera instances for GUI
    const CameraInstance& getCamera(size_t idx) const { 
        if (idx >= cameras.size()) {
            throw CameraException("Invalid camera index");
        }
        return cameras[idx]; 
    }

    CameraInstance& getCamera(size_t idx) { 
        if (idx >= cameras.size()) {
            throw CameraException("Invalid camera index");
        }
        return cameras[idx]; 
    }

    void setKnownCameras(const std::unordered_map<std::string, CameraParams>& configs) {
        known_cameras_ = configs;
    }
private:
    std::vector<CameraInstance> cameras;
    bool recording_active{false};
    std::unordered_map<std::string, CameraParams> known_cameras_;
    
    void loadCameraConfig(CameraInstance& instance, 
                         const std::string& config_file,
                         const GigEVisionDeviceInfo& device_info) {
        // TODO: Implement configuration loading
    }

    void setupGPUStreaming(CameraInstance& instance,
                          const std::string& output_folder,
                          const std::string& encoder_setup) {
        // TODO: Implement GPU streaming setup
    }
};

} // namespace evt