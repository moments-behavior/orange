#pragma once

#include <memory>
#include <functional>
#include <atomic>
#include "emergent_camera.h"
#include "gpu_video_encoder.h"
#include "gpu_manager.h"

namespace evt {

// Forward declarations
class GPUStreamingImpl;

class GPUStreaming {
public:
    struct StreamingConfig {
        bool enable_gpu_direct = false;
        int gpu_device_id = 0;
        bool enable_encoding = false;
        std::string encoder_setup;
        std::string output_folder;
        size_t buffer_count = 5;
    };

    // Callback type for frame processing
    using FrameCallback = std::function<void(const void* data, 
                                           size_t size,
                                           int width, 
                                           int height,
                                           uint64_t timestamp)>;

    explicit GPUStreaming(EmergentCamera& camera, const StreamingConfig& config);
    ~GPUStreaming();

    // Prevent copying
    GPUStreaming(const GPUStreaming&) = delete;
    GPUStreaming& operator=(const GPUStreaming&) = delete;

    // Core streaming operations
    void startStreaming(FrameCallback callback);
    void stopStreaming();

    // Status information
    bool isStreaming() const;
    uint64_t getFrameCount() const;
    uint64_t getDroppedFrames() const;
    double getCurrentFPS() const;

    // GPU-specific operations
    bool isGPUReady() const;
    void getGPUMemoryStatus(size_t& free, size_t& total) const;

private:
    std::unique_ptr<GPUStreamingImpl> impl_;
};

} // namespace evt
