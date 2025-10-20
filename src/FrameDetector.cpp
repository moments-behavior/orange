#include "FrameDetector.h"
#include "enet_utils.h"
#include "global.h"
#include "utils.h"
#include "video_capture.h"
#include <mutex>
#include <npp.h>
#include <nvToolsExt.h>

FrameDetector::FrameDetector(CameraParams *params, CameraEachSelect *select,
                             AppContext *ctx, std::string folder_name)
    : camera_params(params), camera_select(select), ctx(ctx),
      folder_name(folder_name), running(false) {

    CHECK(cudaSetDevice(camera_params->gpu_id));
    stream = nullptr;
    cudaEventCreateWithFlags(&copy_done_event, cudaEventDisableTiming);
}

FrameDetector::~FrameDetector() {
    stop();
    delete yolov8;
    yolov8 = nullptr;
    cudaFreeAsync(frame_process.frame_original.d_orig, stream);
    cudaFreeAsync(frame_process.debayer.d_debayer, stream);
    cudaEventDestroy(copy_done_event);
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaStreamDestroy(stream));
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
    nvtxRangePush("copy_frame_for_detection");
    if (camera_params->gpu_direct) {
        CHECK(cudaMemcpy2DAsync(
            frame_process.frame_original.d_orig, camera_params->width,
            device_image_ptr, camera_params->width, camera_params->width,
            camera_params->height, cudaMemcpyDeviceToDevice, copy_stream));
    } else {
        CHECK(cudaMemcpy2DAsync(
            frame_process.frame_original.d_orig, camera_params->width,
            device_image_ptr, camera_params->width, camera_params->width,
            camera_params->height, cudaMemcpyHostToDevice, copy_stream));
    }
    nvtxRangePop();
    cudaEventRecord(copy_done_event, copy_stream); // mark when copy is done
    camera_select->sigs->frame_detect_state.store(State_Frame_Copy_Done);
    cv.notify_one();
}

void FrameDetector::thread_loop() {
    CHECK(cudaSetDevice(camera_params->gpu_id));
    CHECK(cudaStreamCreate(&stream));
    NppStreamContext npp_ctx =
        make_npp_stream_context(camera_params->gpu_id, stream);
    initalize_gpu_frame_async(&frame_process.frame_original, camera_params,
                              stream);
    initialize_gpu_debayer_async(&frame_process.debayer, camera_params, 3,
                                 stream);

    printf("detector counter %lu\n", detector_counter.load());
    const std::string engine_file_path = camera_select->yolo_model;
    {
        std::lock_guard<std::mutex> lock(graph_capture_mutex);
        printf("make pipe for gpu %d\n", camera_params->gpu_id);
        // nvtxRangePush("warmup");
        yolov8 = new YOLOv8(engine_file_path, camera_params->width,
                            camera_params->height, stream,
                            frame_process.debayer.d_debayer, npp_ctx);
        yolov8->make_pipe(true);
        // nvtxRangePop();
    }
    uint64_t current_counter =
        detector_counter.fetch_add(1); // Atomic increment
    printf("%lu\n", current_counter);
    flatbuffers::FlatBufferBuilder flatb_builder(256);

    std::vector<Bbox> objs;
    std::cout << "camera detector: " << camera_params->camera_serial
              << std::endl;

    bool log_detection = !folder_name.empty();

    std::ofstream write_file;
    if (log_detection) {
        std::string detection_file_name = folder_name + "/Cam" +
                                          camera_params->camera_serial +
                                          "_detection.csv";

        write_file.open(detection_file_name, std::ios::app);
        if (!write_file.is_open()) {
            throw std::runtime_error("Cannot open file: " +
                                     detection_file_name);
        }
        write_file << "timestamp_sys,x,y,width,height,label,prob\n";
    }

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
            return camera_select->sigs->frame_detect_state.load() ==
                       State_Frame_Copy_Done ||
                   !running.load();
        });

        if (!running.load())
            break;
        lock.unlock();

        // GPU processing
        cudaStreamWaitEvent(stream, copy_done_event, 0);
        nvtxRangePush("yolo-pre");
        if (camera_params->color) {
            debayer_frame_gpu_rgb_ctx(camera_params,
                                      &frame_process.frame_original,
                                      &frame_process.debayer, npp_ctx);
        } else {
            duplicate_channel_gpu_3_ctx(camera_params,
                                        &frame_process.frame_original,
                                        &frame_process.debayer, npp_ctx);
        }
        nvtxRangePop();

        nvtxRangePush("yolo-infer");
        if (yolov8->graph_captured) {
            CHECK(cudaGraphLaunch(yolov8->inference_graph_exec, stream));
            CHECK(cudaStreamSynchronize(stream));
        } else {
            yolov8->preprocess_gpu();
            yolov8->infer(); // it sync gpu with cpu here
        }
        nvtxRangePop();
        yolov8->postprocess(objs);

        if (objs.size() > 0) {
            f32 bbox_center_x = objs[0].rect.x + objs[0].rect.width / 2.0;
            f32 bbox_center_y = objs[0].rect.y + objs[0].rect.height / 2.0;

            detection2d[camera_select->idx2d].ball2d.center[0] = {
                bbox_center_x, bbox_center_y};
            detection2d[camera_select->idx2d].ball2d.rects.clear();
            for (const auto &bbox : objs) {
                detection2d[camera_select->idx2d].ball2d.rects.push_back(
                    bbox.rect);
            }
            // std::cout << detection2d[camera_select->idx2d].ball2d.center[0].x
            //           << std::endl;
            detection2d[camera_select->idx2d].ball2d.find_ball.store(true);
            // send to indigo
            if (objs[0].rect.x < 2800.0 && objs[0].rect.x > 2100.0) {
                send_indigo_trigger_message(ctx, flatb_builder);
            }

            if (log_detection) {
                struct timespec ts_rt1;
                clock_gettime(CLOCK_REALTIME, &ts_rt1);
                uint64_t real_time =
                    (ts_rt1.tv_sec * 1000000000LL) + ts_rt1.tv_nsec;

                write_file << real_time << "," << objs[0].rect.x << ","
                           << objs[0].rect.y << "," << objs[0].rect.width << ","
                           << objs[0].rect.height << "," << objs[0].label << ","
                           << objs[0].prob << "\n";
            }
        } else {
            detection2d[camera_select->idx2d].ball2d.find_ball.store(false);
        }

        // running detection
        if (camera_select->detect_mode == Detect3D_Standoff) {
            camera_select->sigs->frame_detect_state.store(
                State_Frame_Detection_Ready);
            std::lock_guard<std::mutex> lock(mtx3d);
            cv3d.notify_one();
        } else {
            camera_select->sigs->frame_detect_state.store(State_Copy_New_Frame);
        }
        count++;
        detection2d[camera_select->idx2d].fps_estimator.update();
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
    if (log_detection) {
        write_file.close();
    }
}
