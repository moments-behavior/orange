#include "JarvisFrameDetector.h"
#include "global.h"
#include "utils.h"
#include <iostream>

JarvisFrameDetector::JarvisFrameDetector(CameraParams *params, CameraEachSelect *select)
    : camera_params(params), camera_select(select), running(false) {
    
    CHECK(cudaEventCreate(&copy_done_event));
}

JarvisFrameDetector::~JarvisFrameDetector() {
    stop();
    CHECK(cudaEventDestroy(copy_done_event));
    if (jarvis_detector) {
        delete jarvis_detector;
    }
}

void JarvisFrameDetector::start() {
    if (running.load()) {
        return;
    }
    
    running.store(true);
    worker_thread = std::thread(&JarvisFrameDetector::thread_loop, this);
}

void JarvisFrameDetector::stop() {
    if (!running.load()) {
        return;
    }
    
    running.store(false);
    cv.notify_one();
    
    if (worker_thread.joinable()) {
        worker_thread.join();
    }
}

void JarvisFrameDetector::notify_frame_ready(void *device_image_ptr, cudaStream_t copy_stream) {
    if (!running.load()) {
        return;
    }
    
    // Copy frame to our processing buffer
    size_t frame_size = camera_params->width * camera_params->height * 3 * sizeof(unsigned char);
    CHECK(cudaMemcpyAsync(frame_process.frame_original.d_orig, device_image_ptr, 
                          frame_size, cudaMemcpyDeviceToDevice, copy_stream));
    
    // Record event when copy is done
    CHECK(cudaEventRecord(copy_done_event, copy_stream));
    
    // Signal that frame is ready for processing
    camera_select->frame_detect_state.store(State_Frame_Copy_Done);
    cv.notify_one();
}

void JarvisFrameDetector::thread_loop() {
    CHECK(cudaSetDevice(camera_params->gpu_id));
    CHECK(cudaStreamCreate(&stream));
    NppStreamContext npp_ctx = make_npp_stream_context(camera_params->gpu_id, stream);
    
    initalize_gpu_frame_async(&frame_process.frame_original, camera_params, stream);
    initialize_gpu_debayer_async(&frame_process.debayer, camera_params, 3, stream);

    printf("Jarvis detector counter %lu\n", detector_counter.load());
    const std::string model_dir = camera_select->jarvis_model_dir;
    
    {
        std::lock_guard<std::mutex> lock(graph_capture_mutex);
        printf("make jarvis pipe for gpu %d\n", camera_params->gpu_id);
        
        // Initialize Jarvis detector
        jarvis_detector = new JarvisPoseDetector(model_dir, 4, stream); // Assuming 4 cameras
        jarvis_detector->make_pipe(true);
    }
    
    uint64_t current_counter = detector_counter.fetch_add(1);
    printf("Jarvis detector initialized: %lu\n", current_counter);

    std::cout << "Jarvis camera detector: " << camera_params->camera_serial << std::endl;

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::time_point();
    int count = 0;
    
    while (running.load()) {
        // Start timing after 10 frames
        if (count == 10) {
            start = std::chrono::high_resolution_clock::now();
        }

        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] {
            return camera_select->frame_detect_state.load() == State_Frame_Copy_Done ||
                   !running.load();
        });

        if (!running.load()) {
            break;
        }
        lock.unlock();

        // GPU processing
        cudaStreamWaitEvent(stream, copy_done_event, 0);
        
        // Debayer the frame
        if (camera_params->color) {
            debayer_frame_gpu_rgb_ctx(camera_params, &frame_process.frame_original,
                                      &frame_process.debayer, npp_ctx);
        } else {
            duplicate_channel_gpu_3_ctx(camera_params, &frame_process.frame_original,
                                        &frame_process.debayer, npp_ctx);
        }

        // Run Jarvis center detection on this camera's frame
        unsigned char *d_input_images[1] = {frame_process.debayer.d_debayer};
        jarvis_detector->detect_centers(d_input_images, camera_params->width, camera_params->height);
        
        // Get results and update detection data
        std::vector<cv::Point2f> centers;
        std::vector<float> confidences;
        jarvis_detector->get_center_results(centers, confidences);
        
        // Update detection2d for this camera
        if (centers.size() > 0 && confidences[0] > 0.5f) {
            // Store center detection results
            detection2d[camera_select->idx2d].jarvis_center.center = centers[0];
            detection2d[camera_select->idx2d].jarvis_center.confidence = confidences[0];
            detection2d[camera_select->idx2d].jarvis_center.find_center.store(true);
            
            // Run keypoint detection on cropped region around center
            // For now, use the full frame as the "cropped" region
            // In real implementation, this would crop around the center point
            jarvis_detector->detect_keypoints(d_input_images, camera_params->width, camera_params->height);
            
            // Get keypoint results
            std::vector<cv::Point2f> keypoints;
            std::vector<float> keypoint_confidences;
            jarvis_detector->get_keypoint_results(keypoints, keypoint_confidences);
            
            // Store keypoint results
            if (keypoints.size() >= JARVIS_NUM_KEYPOINTS) {
                for (int k = 0; k < JARVIS_NUM_KEYPOINTS; k++) {
                    detection2d[camera_select->idx2d].jarvis_keypoints.keypoints[k] = keypoints[k];
                    detection2d[camera_select->idx2d].jarvis_keypoints.confidence[k] = keypoint_confidences[k];
                }
                detection2d[camera_select->idx2d].jarvis_keypoints.find_keypoints.store(true);
            } else {
                detection2d[camera_select->idx2d].jarvis_keypoints.find_keypoints.store(false);
            }
            
            std::cout << "Jarvis center detected at: (" << centers[0].x << ", " << centers[0].y 
                      << ") with confidence: " << confidences[0] << std::endl;
        } else {
            detection2d[camera_select->idx2d].jarvis_center.find_center.store(false);
            detection2d[camera_select->idx2d].jarvis_keypoints.find_keypoints.store(false);
        }

        // Signal that detection is ready
        camera_select->frame_detect_state.store(State_Frame_Detection_Ready);
        count++;
    }

    if (start != std::chrono::high_resolution_clock::time_point()) {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        float calc_frame_rate = (count - 10) / elapsed.count();
        std::cout << "Jarvis Detection Frame Rate: " << calc_frame_rate << std::endl;
    }

    CHECK(cudaStreamDestroy(stream));
}
