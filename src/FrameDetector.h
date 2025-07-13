#pragma once

#include "camera.h"
#include "image_processing.h"
#include "video_capture.h"
#include "yolov8_det.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

class FrameDetector {
  public:
    FrameDetector(CameraParams *params, CameraEachSelect *select);
    ~FrameDetector();

    void start();
    void stop();

    // Called from capture thread when a frame is ready
    void notify_frame_ready(void *device_image_ptr, cudaStream_t copy_stream);

  private:
    void thread_loop();

    cudaStream_t stream;
    cudaEvent_t copy_done_event;
    CameraParams *camera_params;
    CameraEachSelect *camera_select;
    FrameProcess frame_process;
    YOLOv8 *yolov8;

    std::atomic<bool> running;
    std::mutex mtx;
    std::condition_variable cv;
    std::thread worker_thread;
};
