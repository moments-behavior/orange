#pragma once

#include "camera.h"
#include "image_processing.h"
#include "video_capture.h"
#include "jarvis_pose_det.h"
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <thread>

class JarvisFrameDetector {
  public:
    JarvisFrameDetector(CameraParams *params, CameraEachSelect *select);
    ~JarvisFrameDetector();

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
    JarvisPoseDetector *jarvis_detector;

    std::atomic<bool> running;
    std::mutex mtx;
    std::condition_variable cv;
    std::thread worker_thread;
};
