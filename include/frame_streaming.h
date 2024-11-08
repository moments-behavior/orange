// frame_streaming.h
#pragma once

#include <functional>
#include <atomic>
#include <thread>
#include "emergent_camera.h"

namespace evt {

// Callback type for frame processing
using FrameCallback = std::function<void(const Emergent::CEmergentFrame&, uint64_t timestamp)>;

class FrameStreaming {
public:
    explicit FrameStreaming(EmergentCamera& camera, size_t buffer_count = 5);
    ~FrameStreaming();

    // Prevent copying
    FrameStreaming(const FrameStreaming&) = delete;
    FrameStreaming& operator=(const FrameStreaming&) = delete;

    // Core streaming operations
    void startStreaming(FrameCallback callback);
    void stopStreaming();
    
    // Status
    bool isStreaming() const { return is_streaming_; }
    uint64_t getFrameCount() const { return frame_count_; }
    uint64_t getDroppedFrames() const { return dropped_frames_; }
    
private:
    void streamingThread();
    void handleFrame(Emergent::CEmergentFrame& frame);
    
    EmergentCamera& camera_;
    std::thread streaming_thread_;
    std::atomic<bool> is_streaming_{false};
    std::atomic<uint64_t> frame_count_{0};
    std::atomic<uint64_t> dropped_frames_{0};
    std::atomic<uint16_t> last_frame_id_{0};
    FrameCallback frame_callback_;
    const size_t buffer_count_;
};

} // namespace evt
