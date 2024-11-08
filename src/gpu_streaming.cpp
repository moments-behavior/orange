// gpu_streaming.cpp
#include "gpu_streaming.h"
#include "gpu_video_encoder.h"
#include <thread>
#include <chrono>
#include <queue>
#include <mutex>
#include <condition_variable>

namespace evt {

class GPUStreamingImpl {
public:
    GPUStreamingImpl(EmergentCamera& camera, const GPUStreaming::StreamingConfig& config)
        : camera_(camera),
          config_(config),
          gpu_manager_(GPUManager::getInstance()) {
        
        if (config.enable_gpu_direct) {
            initializeGPU();
        }
    }

    void initializeGPU() {
        try {
            gpu_manager_.initialize();
            gpu_manager_.selectDevice(config_.gpu_device_id);

            if (!gpu_manager_.isGPUDirectSupported(config_.gpu_device_id)) {
                throw GPUException("GPU Direct not supported on selected device");
            }

            const auto& camera_params = camera_.getParams();
            if (!gpu_manager_.verifyGPUDirectCompatibility(camera_params.camera_serial)) {
                throw GPUException("GPU Direct not compatible with this camera configuration");
            }

            if (config_.enable_encoding) {
                initializeEncoder();
            }

            gpu_ready_ = true;
        } catch (const GPUException& e) {
            std::cerr << "GPU initialization failed: " << e.what() << std::endl;
            if (e.getErrorCode() != cudaSuccess) {
                std::cerr << "CUDA error: " << cudaGetErrorString(e.getErrorCode()) << std::endl;
            }
            gpu_ready_ = false;
        }
    }

    void initializeEncoder() {
        const auto& params = camera_.getParams();
        
        // Validate encoder parameters
        if (!validate_encoder_parameters(&params, config_.encoder_setup)) {
            throw GPUException("Invalid encoder parameters for current resolution");
        }

        encoder_ = std::make_unique<GPUVideoEncoder>(
            "GPU Encoder",
            &params,
            config_.encoder_setup,
            config_.output_folder,
            &encoder_ready_
        );

        if (encoder_) {
            encoder_->StartThread();
            
            // Wait for encoder initialization
            auto start = std::chrono::steady_clock::now();
            while (!encoder_ready_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                if (std::chrono::steady_clock::now() - start > std::chrono::seconds(5)) {
                    throw GPUException("Encoder initialization timeout");
                }
            }
        }
    }

    void startStreaming(GPUStreaming::FrameCallback callback) {
        if (is_streaming_) {
            throw CameraException("Already streaming");
        }

        frame_callback_ = std::move(callback);
        is_streaming_ = true;

        // Allocate frame buffers
        camera_.allocateFrameBuffers(config_.buffer_count);

        // Start streaming thread
        streaming_thread_ = std::thread(&GPUStreamingImpl::streamingThread, this);

        // Start camera streaming
        camera_.startStream();
    }

    void stopStreaming() {
        if (!is_streaming_) {
            return;
        }

        is_streaming_ = false;

        try {
            camera_.stopStream();

            if (streaming_thread_.joinable()) {
                streaming_thread_.join();
            }

            if (encoder_) {
                encoder_->StopThread();
            }

            camera_.releaseFrameBuffers();
        } catch (const std::exception& e) {
            std::cerr << "Error during stream shutdown: " << e.what() << std::endl;
        }
    }

    void streamingThread() {
        Emergent::CEmergentFrame frame;
        const int timeout_ms = 1000;
        
        auto last_fps_update = std::chrono::steady_clock::now();
        uint32_t frames_this_second = 0;

        while (is_streaming_) {
            try {
                if (camera_.getFrame(&frame, timeout_ms)) {
                    handleFrame(frame);
                    camera_.queueFrame(&frame);

                    // Update FPS calculation
                    frames_this_second++;
                    auto now = std::chrono::steady_clock::now();
                    if (now - last_fps_update >= std::chrono::seconds(1)) {
                        current_fps_ = frames_this_second;
                        frames_this_second = 0;
                        last_fps_update = now;
                    }
                } else {
                    dropped_frames_++;
                }
            } catch (const std::exception& e) {
                std::cerr << "Streaming error: " << e.what() << std::endl;
                dropped_frames_++;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }

    void handleFrame(const Emergent::CEmergentFrame& frame) {
        frame_count_++;

        // Get system timestamp
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count();

        if (gpu_ready_ && encoder_) {
            // Push frame to GPU encoder
            encoder_->PushToDisplay(
                frame.imagePtr,
                frame.bufferSize,
                frame.size_x,
                frame.size_y,
                frame.pixel_type,
                frame.timestamp,
                frame_count_,
                timestamp
            );
        }

        // Call user callback if registered
        if (frame_callback_) {
            frame_callback_(
                frame.imagePtr,
                frame.bufferSize,
                frame.size_x,
                frame.size_y,
                timestamp
            );
        }
    }

    // Status accessors
    bool isStreaming() const { return is_streaming_; }
    uint64_t getFrameCount() const { return frame_count_; }
    uint64_t getDroppedFrames() const { return dropped_frames_; }
    double getCurrentFPS() const { return current_fps_; }
    bool isGPUReady() const { return gpu_ready_; }
    
    void getGPUMemoryStatus(size_t& free, size_t& total) const {
        if (gpu_ready_) {
            gpu_manager_.getMemoryInfo(free, total);
        } else {
            free = 0;
            total = 0;
        }
    }

private:
    EmergentCamera& camera_;
    GPUStreaming::StreamingConfig config_;
    GPUManager& gpu_manager_;
    
    std::unique_ptr<GPUVideoEncoder> encoder_;
    bool encoder_ready_ = false;
    bool gpu_ready_ = false;

    std::atomic<bool> is_streaming_{false};
    std::atomic<uint64_t> frame_count_{0};
    std::atomic<uint64_t> dropped_frames_{0};
    std::atomic<double> current_fps_{0.0};

    std::thread streaming_thread_;
    GPUStreaming::FrameCallback frame_callback_;
};

// GPUStreaming implementation
GPUStreaming::GPUStreaming(EmergentCamera& camera, const StreamingConfig& config)
    : impl_(std::make_unique<GPUStreamingImpl>(camera, config)) {}

GPUStreaming::~GPUStreaming() = default;

void GPUStreaming::startStreaming(FrameCallback callback) {
    impl_->startStreaming(std::move(callback));
}

void GPUStreaming::stopStreaming() {
    impl_->stopStreaming();
}

bool GPUStreaming::isStreaming() const {
    return impl_->isStreaming();
}

uint64_t GPUStreaming::getFrameCount() const {
    return impl_->getFrameCount();
}

uint64_t GPUStreaming::getDroppedFrames() const {
    return impl_->getDroppedFrames();
}

double GPUStreaming::getCurrentFPS() const {
    return impl_->getCurrentFPS();
}

bool GPUStreaming::isGPUReady() const {
    return impl_->isGPUReady();
}

void GPUStreaming::getGPUMemoryStatus(size_t& free, size_t& total) const {
    impl_->getGPUMemoryStatus(free, total);
}

} // namespace evt