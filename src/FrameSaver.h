#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <opencv2/opencv.hpp>
#include "camera.h"
#include "video_capture.h"
#include "image_processing.h"

class FrameSaver {
public:
    FrameSaver(CameraParams* params,
               CameraEachSelect* select);
    ~FrameSaver();

    void start();
    void stop();

    // Called from capture thread when a frame is ready
    void notify_frame_ready(void* device_image_ptr);

private:
    void thread_loop();

    cudaStream_t stream;
    CameraParams* camera_params;
    CameraEachSelect* camera_select;
    FrameProcess frame_process;
    std::atomic<bool> running;

    std::mutex mtx;
    std::condition_variable cv;
    std::thread worker_thread;
};