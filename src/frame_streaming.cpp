// frame_streaming.cpp
#include "frame_streaming.h"
#include <chrono>
#include <iostream>

namespace evt {

FrameStreaming::FrameStreaming(EmergentCamera& camera, size_t buffer_count)
    : camera_(camera), buffer_count_(buffer_count) {
    if (!camera_.isOpen()) {
        throw CameraException("Camera must be opened before creating streaming object");
    }
}

FrameStreaming::~FrameStreaming() {
    if (is_streaming_) {
        stopStreaming();
    }
}

void FrameStreaming::startStreaming(FrameCallback callback) {
    if (is_streaming_) {
        throw CameraException("Streaming already in progress");
    }

    try {
        frame_callback_ = std::move(callback);
        
        // Allocate frame buffers
        camera_.allocateFrameBuffers(buffer_count_);
        
        // Reset counters
        frame_count_ = 0;
        dropped_frames_ = 0;
        last_frame_id_ = 0;
        
        // Start the streaming thread
        is_streaming_ = true;
        streaming_thread_ = std::thread(&FrameStreaming::streamingThread, this);
        
        // Start camera streaming
        camera_.startStream();
        
    } catch (...) {
        is_streaming_ = false;
        throw;
    }
}

void FrameStreaming::stopStreaming() {
    if (!is_streaming_) {
        return;
    }

    is_streaming_ = false;
    
    try {
        // Stop camera streaming
        camera_.stopStream();
        
        // Wait for streaming thread to finish
        if (streaming_thread_.joinable()) {
            streaming_thread_.join();
        }
        
        // Release frame buffers
        camera_.releaseFrameBuffers();
        
    } catch (const std::exception& e) {
        std::cerr << "Error during stream shutdown: " << e.what() << std::endl;
        // Continue cleanup despite errors
    }
}

void FrameStreaming::streamingThread() {
    Emergent::CEmergentFrame frame;
    const int timeout_ms = 1000; // 1 second timeout
    
    while (is_streaming_) {
        try {
            // Try to get a frame
            if (camera_.getFrame(&frame, timeout_ms)) {
                handleFrame(frame);
                
                // Re-queue the frame buffer
                camera_.queueFrame(&frame);
            } else {
                // Timeout occurred
                dropped_frames_++;
                std::cerr << "Frame acquisition timeout" << std::endl;
            }
            
        } catch (const std::exception& e) {
            dropped_frames_++;
            std::cerr << "Error during frame acquisition: " << e.what() << std::endl;
            
            // Brief sleep to prevent tight error loop
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

void FrameStreaming::handleFrame(Emergent::CEmergentFrame& frame) {
    // Check for dropped frames using frame ID
    if (frame_count_ > 0 && frame.frame_id != last_frame_id_ + 1) {
        // Account for ID wraparound at 65535
        if (!(last_frame_id_ == 65535 && frame.frame_id == 1)) {
            dropped_frames_++;
        }
    }
    
    last_frame_id_ = frame.frame_id;
    frame_count_++;
    
    // Get system timestamp
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count();
    
    // Call user callback with frame data
    if (frame_callback_) {
        frame_callback_(frame, timestamp);
    }
}

} // namespace evt