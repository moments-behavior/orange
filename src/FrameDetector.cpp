#include "FrameDetector.h"
#include "global.h"
#include "video_capture.h"
#include <mutex>
#include <npp.h>
#include <nvToolsExt.h>

FrameDetector::FrameDetector(CameraParams *params, CameraEachSelect *select)
    : camera_params(params), camera_select(select), running(false) {
    cudaEventCreateWithFlags(&copy_done_event, cudaEventDisableTiming);
}

FrameDetector::~FrameDetector() {
    stop();
    cudaFree(frame_process.frame_original.d_orig);
    cudaFree(frame_process.debayer.d_debayer);
    cudaEventDestroy(copy_done_event);
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

void FrameDetector::notify_frame_ready(void *device_image_ptr,
                                       cudaStream_t copy_stream) {
    // nvtxRangePush("copy_frame_for_detection");
    if (camera_params->gpu_direct) {
        ck(cudaMemcpy2DAsync(
            frame_process.frame_original.d_orig, camera_params->width,
            device_image_ptr, camera_params->width, camera_params->width,
            camera_params->height, cudaMemcpyDeviceToDevice, copy_stream));
    } else {
        ck(cudaMemcpy2DAsync(
            frame_process.frame_original.d_orig, camera_params->width,
            device_image_ptr, camera_params->width, camera_params->width,
            camera_params->height, cudaMemcpyHostToDevice, copy_stream));
    }
    // nvtxRangePop();
    cudaEventRecord(copy_done_event, copy_stream); // mark when copy is done
    camera_select->frame_detect_state.store(State_Frame_Copy_Done);
    cv.notify_one();
}

void FrameDetector::thread_loop() {
    ck(cudaSetDevice(camera_params->gpu_id));
    ck(cudaStreamCreate(&stream));
    ck(nppSetStream(stream));
    initalize_gpu_frame(&frame_process.frame_original, camera_params);
    initialize_gpu_debayer(&frame_process.debayer, camera_params, 3);

    ck(cudaMalloc((void **)&frame_process.d_convert,
                  camera_params->width * camera_params->height * 3));
    // initialize yolo
    const std::string engine_file_path = camera_select->yolo_model;
    yolov8 = new YOLOv8(engine_file_path, camera_params->width,
                        camera_params->height, frame_process.debayer.d_debayer,
                        true, stream);
    // nvtxRangePush("warmup");
    yolov8->make_pipe(true);
    // nvtxRangePop();
    std::vector<Bbox> objs;
    std::cout << "camera detector: " << camera_params->camera_serial
              << std::endl;

    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::time_point();
    int count = 0;
    while (running.load()) {
        // start timing after 10 frames
        if (count == 10) {
            start = std::chrono::high_resolution_clock::now();
        }

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
        cudaStreamWaitEvent(stream, copy_done_event, 0);
        if (camera_params->color) {
            debayer_frame_gpu_rgb(camera_params, &frame_process.frame_original,
                                  &frame_process.debayer);
        } else {
            duplicate_channel_gpu_3(camera_params,
                                    &frame_process.frame_original,
                                    &frame_process.debayer);
        }

        if (yolov8->graph_captured) {
            // nvtxRangePush("graph");
            CHECK(cudaGraphLaunch(yolov8->inference_graph_exec, stream));
            CHECK(cudaStreamSynchronize(stream));
            // nvtxRangePop();
        } else {
            yolov8->preprocess_gpu();
            yolov8->infer(); // it sync gpu with cpu here
        }
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

    if (start == std::chrono::high_resolution_clock::time_point()) {
        // start is zero (uninitialized)
        std::cout << "Run it longer for meaning report of detection fps.\n";
    } else {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        float calc_frame_rate = (count - 10) / elapsed.count();
        std::cout << camera_params->camera_serial
                  << ", Detect Frame Rate : " + std::to_string(calc_frame_rate)
                  << std::endl;
    }
}
