#pragma once

#include <vector>
#include <memory>
#include <string>
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

void initializeCameras(const std::vector<bool>& selected_cameras,
                      const std::vector<GigEVisionDeviceInfo>& device_info,
                      const std::vector<std::string>& config_files) {
    cameras.clear();
    cameras.reserve(device_info.size());
    
    LOG(INFO) << "Starting camera initialization with " << device_info.size() << " devices";
    
    for (size_t i = 0; i < device_info.size(); ++i) {
        // Only initialize selected cameras
        if (!selected_cameras[i]) {
            LOG(INFO) << "Camera " << i << " not selected, skipping";
            continue;
        }

        LOG(INFO) << "Initializing camera " << i << " (serial: " << device_info[i].serialNumber << ")";
        
        CameraInstance instance;
        
        // Set up camera parameters
        instance.params.camera_serial = device_info[i].serialNumber;
        instance.params.camera_name = device_info[i].userDefinedName;
        
        // Create camera instance
        try {
            instance.camera = std::make_unique<EmergentCamera>(instance.params);
            
            // Open the camera with device info
            instance.camera->open(&device_info[i]);
            
            cameras.push_back(std::move(instance));
            LOG(INFO) << "Successfully initialized camera " << i;
        }
        catch (const evt::CameraException& e) {
            LOG(ERROR) << "Failed to initialize camera " << i << ": " << e.what();
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
                instance.camera->open(nullptr); // You'll need to pass proper device info here
            }

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
        }
        catch (const CameraException& e) {
            std::cerr << "Failed to start streaming: " << e.what() << std::endl;
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

    // Status queries
    size_t getCameraCount() const { return cameras.size(); }
    bool isStreaming(size_t camera_idx) const { 
        return camera_idx < cameras.size() && cameras[camera_idx].is_streaming;
    }
    bool isRecording() const { return recording_active; }
    
    // Access to camera instances for GUI
    const CameraInstance& getCamera(size_t idx) const { return cameras[idx]; }
    CameraInstance& getCamera(size_t idx) { return cameras[idx]; }

    // Add a safe method to get camera parameters
    const CameraParams& getCameraParams(size_t idx) const {
        if (idx >= cameras.size() || !cameras[idx].camera) {
            static const CameraParams default_params;
            return default_params;
        }
        return cameras[idx].params;
    }

private:
    std::vector<CameraInstance> cameras;
    bool recording_active{false};
    
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