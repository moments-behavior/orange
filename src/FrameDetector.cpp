#include "FrameDetector.h"
#include "global.h"
#include "kernel.cuh"
#include "video_capture.h"
#include <mutex>
#include <npp.h>

FrameDetector::FrameDetector(CameraParams *params, CameraEachSelect *select)
    : camera_params(params), camera_select(select), running(false) {

    ck(cudaStreamCreate(&stream));
    initalize_gpu_frame(&frame_process.frame_original, camera_params);
    initialize_gpu_debayer(&frame_process.debayer, camera_params);
    initialize_pinned_cpu_frame(&frame_process.frame_cpu, camera_params);

    ck(cudaMalloc((void **)&frame_process.d_convert,
                  camera_params->width * camera_params->height * 3));
    // initialize yolo
    const std::string engine_file_path = camera_select->yolo_model;
    yolov8 = new YOLOv8(engine_file_path, camera_params->width,
                        camera_params->height);
    yolov8->make_pipe(true);
}

FrameDetector::~FrameDetector() {
    stop();
    cudaFree(frame_process.frame_original.d_orig);
    cudaFree(frame_process.debayer.d_debayer);
    cudaFreeHost(frame_process.frame_cpu.frame);
    ck(cudaStreamDestroy(stream));
}

void FrameDetector::start() {
    running.store(true);
    worker_thread = std::thread(&FrameDetector::thread_loop, this);
}

void FrameDetector::stop() {
    if (running.load()) {
        running.store(false);
        cv.notify_all();
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
    }
}

void FrameDetector::notify_frame_ready(void *device_image_ptr) {
    if (camera_params->gpu_direct) {
        ck(cudaMemcpy2D(frame_process.frame_original.d_orig,
                        camera_params->width, device_image_ptr,
                        camera_params->width, camera_params->width,
                        camera_params->height, cudaMemcpyDeviceToDevice));
    } else {
        ck(cudaMemcpy2D(frame_process.frame_original.d_orig,
                        camera_params->width, device_image_ptr,
                        camera_params->width, camera_params->width,
                        camera_params->height, cudaMemcpyHostToDevice));
    }
    camera_select->frame_detect_state.store(State_Frame_Copy_Done);
    cv.notify_one();
}

void FrameDetector::thread_loop() {
    ck(cudaSetDevice(camera_params->gpu_id));
    ck(nppSetStream(stream));
    std::vector<Bbox> objs;
    std::cout << "camera detector: " << camera_params->camera_serial
              << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    int count = 0;
    while (running.load()) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] {
            return camera_select->frame_detect_state.load() ==
                       State_Frame_Copy_Done ||
                   !running.load();
        });

        if (!running.load())
            break;
        lock.unlock();

        // GPU processing
        if (camera_params->color) {
            debayer_frame_gpu(camera_params, &frame_process.frame_original,
                              &frame_process.debayer);
        } else {
            duplicate_channel_gpu(camera_params, &frame_process.frame_original,
                                  &frame_process.debayer);
        }

        rgba2rgb_convert(frame_process.d_convert,
                         frame_process.debayer.d_debayer, camera_params->width,
                         camera_params->height, stream);
        yolov8->preprocess_gpu(frame_process.d_convert);
        yolov8->infer();
        yolov8->postprocess(objs);

        if (objs.size() > 0) {
            f32 bbox_center_x = objs[0].rect.x + objs[0].rect.width / 2.0;
            f32 bbox_center_y = objs[0].rect.y + objs[0].rect.height / 2.0;

            detection2d[camera_select->idx2d].ball2d.center[0] = {
                bbox_center_x, bbox_center_y};
            // std::cout << detection2d[camera_select->idx2d].ball2d.center[0].x
            //           << std::endl;
            detection2d[camera_select->idx2d].ball2d.find_ball.store(true);
        } else {
            detection2d[camera_select->idx2d].ball2d.find_ball.store(false);
        }

        ck(cudaStreamSynchronize(stream));

        // running detection
        if (camera_select->detect_mode == Detect3d_Standoff) {
            camera_select->frame_detect_state.store(
                State_Frame_Detection_Ready);
            std::lock_guard<std::mutex> lock(mtx3d);
            cv3d.notify_one();
        } else {
            camera_select->frame_detect_state.store(State_Copy_New_Frame);
        }
        count++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    float calc_frame_rate = count / elapsed.count();
    std::cout << camera_params->camera_serial
              << ", Detect Frame Rate : " + std::to_string(calc_frame_rate)
              << std::endl;
}
