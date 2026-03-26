#include "video_capture.h"
#if defined(__GNUC__)
#include <unistd.h>
#endif
#include "global.h"
#include "kernel.cuh"
#include "opengldisplay.h"
#include "utils.h"
#include "obj_generated.h"
#include <cuda_runtime_api.h>
#include <nvToolsExt.h>
#include <stdio.h>
#include <string.h>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <iostream>

COpenGLDisplay::COpenGLDisplay(const char *name, CameraParams *camera_params,
                               CameraEachSelect *camera_select,
                               unsigned char *display_buffer,
                               INDIGOSignalBuilder *indigo_signal_builder)
    : CThreadWorker(name), camera_params(camera_params),
      camera_select(camera_select), display_buffer(display_buffer),
      indigo_signal_builder(indigo_signal_builder) {
    input_image_size.width = camera_params->width;
    input_image_size.height = camera_params->height;
    input_image_roi.x = 0;
    input_image_roi.y = 0;
    input_image_roi.width = camera_params->width;
    input_image_roi.height = camera_params->height;

    output_image_size.width =
        int(camera_params->width / camera_select->downsample);
    output_image_size.height =
        int(camera_params->height / camera_select->downsample);

    output_image_roi.x = 0;
    output_image_roi.y = 0;
    output_image_roi.width = output_image_size.width;
    output_image_roi.height = output_image_size.height;

    memset(workerEntries, 0, sizeof(workerEntries));
    workerEntriesFreeQueueCount = WORK_ENTRIES_MAX;
    for (int i = 0; i < workerEntriesFreeQueueCount; i++) {
        workerEntriesFreeQueue[i] = &workerEntries[i];
    }
    
    if (camera_select->detect_mode == Detect2D_GLThread &&
        camera_select->yolo_model.empty()) {
        std::cerr << "COpenGLDisplay: Detect2D_GLThread selected but YOLO model "
                     "path is empty for camera "
                  << camera_params->camera_serial << std::endl;
    }

    if (camera_select->enable_obb &&
        camera_select->detect_mode != Detect2D_GLThread) {
        std::cerr
            << "COpenGLDisplay: OBB is enabled for camera "
            << camera_params->camera_serial
            << " but stream overlay path only feeds YOLO boxes in Detect2D_GLThread mode."
            << std::endl;
    }

    // Initialize OBB detector if enabled
    if (camera_select->enable_obb) {
        if (camera_select->obb_csv_path.empty()) {
            std::cerr
                << "COpenGLDisplay: OBB is enabled but obb_csv_path is empty for camera "
                << camera_params->camera_serial << std::endl;
            return;
        }

        OBBDetectorParams obb_params;
        std::vector<std::string> csv_paths = {camera_select->obb_csv_path};
        obb_detector = new OBBDetector(camera_params, csv_paths, obb_params);
        
        if (!obb_detector->initialize()) {
            std::cerr << "COpenGLDisplay: Failed to initialize OBB detector with CSV "
                      << camera_select->obb_csv_path << " for camera "
                      << camera_params->camera_serial << std::endl;
            delete obb_detector;
            obb_detector = nullptr;
        } else {
            std::cout << "COpenGLDisplay: OBB detector initialized for camera "
                      << camera_params->camera_serial << std::endl;
            obb_detector->start();
        }
    }
}

COpenGLDisplay::~COpenGLDisplay() {
    cudaFree(frame_original.d_orig);
    cudaFree(debayer.d_debayer);
    if (camera_select->detect_mode == Detect2D_GLThread) {
        delete yolov8;
    }
    
    if (obb_detector) {
        obb_detector->stop();
        delete obb_detector;
        obb_detector = nullptr;
    }
    
    if (d_obb_points) {
        cudaFree(d_obb_points);
        d_obb_points = nullptr;
    }
}

void COpenGLDisplay::ThreadRunning() {
    ck(cudaSetDevice(camera_params->gpu_id));

    if (camera_select->downsample != 1) {
        ck(cudaMalloc((void **)&d_resize,
                      output_image_size.width * output_image_size.height * 4));
    }

    // innitialization
    initalize_gpu_frame(&frame_original, camera_params);
    initialize_gpu_debayer(&debayer, camera_params, 4);
    initialize_cpu_frame(&frame_cpu, camera_params);

    ck(cudaMalloc((void **)&d_convert,
                  camera_params->width * camera_params->height * 3));

    unsigned int skeleton[8] = {0, 1, 1, 2, 2, 3, 3, 0}; // box
    NppStreamContext npp_ctx =
        make_npp_stream_context(camera_params->gpu_id, 0);
    const bool yolo_glthread_enabled =
        (camera_select->detect_mode == Detect2D_GLThread);
    if (yolo_glthread_enabled) {
        printf("YOLO initialization...\n");

        const std::string engine_file_path = camera_select->yolo_model;
        yolov8 = new YOLOv8(engine_file_path, camera_params->width,
                            camera_params->height, 0, d_convert, npp_ctx);
        yolov8->make_pipe(false);

        cudaMalloc((void **)&d_points, sizeof(float) * 8);
        cudaMalloc((void **)&d_skeleton, sizeof(unsigned int) * 8);
        CHECK(cudaMemcpy(d_skeleton, skeleton, sizeof(unsigned int) * 8,
                         cudaMemcpyHostToDevice));
    }
    
    // Allocate OBB GPU resources if needed
    const bool obb_overlay_enabled =
        (camera_select->enable_obb && obb_detector && yolo_glthread_enabled);
    if (camera_select->enable_obb && obb_detector && !yolo_glthread_enabled) {
        std::cerr << "COpenGLDisplay: OBB detector is running but will not receive "
                     "YOLO boxes until detect mode is Detect2D_GLThread."
                  << std::endl;
    }
    if (obb_overlay_enabled) {
        cudaMalloc((void **)&d_obb_points, sizeof(float) * 8 * 10); // Support up to 10 OBBs
    }

    std::vector<Bbox> objs;
    std::vector<Bbox> objs_last_frame;
    std::vector<OBB> obb_detections;  // Add OBB results

    using clock = std::chrono::steady_clock;
    
    // Static variables for OBB lock-in system tracking
    static std::vector<OBB> last_locked_detections;
    static bool last_detections_sent = false;
    
    // Static variables for persistent slot tracking
    static int persistent_slot_assignments[2] = {-1, -1};  // Object IDs assigned to each slot
    static bool slots_initialized = false;
    
    // OBB Background Building

    int frameCount = 0;
    auto lastFPSUpdate = clock::now();
    auto lastDetectionStatsUpdate = clock::now();
    uint64_t yolo_frame_counter = 0;
    uint64_t yolo_nonempty_counter = 0;
    uint64_t obb_nonempty_counter = 0;
    size_t last_yolo_obj_count = 0;
    size_t last_obb_obj_count = 0;

    while (IsMachineOn()) {
        auto frameStart = clock::now();
        std::chrono::duration<double, std::milli> targetFrameDuration(
            1000.0 / streaming_target_fps.load());
        void *f = GetObjectFromQueueIn();
        if (f) {
            WORKER_ENTRY entry = *(WORKER_ENTRY *)f;
            PutObjectToQueueOut(f);

            // nvtxRangePush("display_gl_copy_debayer");
            // copy frame from cpu to gpu
            CHECK(cudaMemcpy2D(frame_original.d_orig, camera_params->width,
                               entry.imagePtr, camera_params->width,
                               camera_params->width, camera_params->height,
                               cudaMemcpyHostToDevice));

            if (camera_params->color) {
                debayer_frame_gpu(camera_params, &frame_original, &debayer);
            } else {
                duplicate_channel_gpu(camera_params, &frame_original, &debayer);
            }
            // nvtxRangePop();

            if (yolo_glthread_enabled) {
                // Match yolo_offline input layout (OpenCV frames are BGR).
                rgba2bgr_convert(d_convert, debayer.d_debayer,
                                 camera_params->width, camera_params->height,
                                 0);

                if (yolov8->graph_captured) {
                    // nvtxRangePush("graph");
                    CHECK(cudaGraphLaunch(yolov8->inference_graph_exec, 0));
                    CHECK(cudaStreamSynchronize(0));
                    // nvtxRangePop();
                } else {
                    yolov8->preprocess_gpu();
                    yolov8->infer(); // it sync gpu with cpu here
                }

                yolov8->postprocess(objs);
                yolo_frame_counter++;
                last_yolo_obj_count = objs.size();
                if (!objs.empty()) {
                    yolo_nonempty_counter++;
                    // Print a few early detections to validate runtime coordinates.
                    if (yolo_nonempty_counter <= 5) {
                        const auto &b0 = objs[0];
                        std::cout << "YOLO sample [" << camera_params->camera_serial
                                  << "]: n=" << objs.size() << " first=("
                                  << b0.rect.x << "," << b0.rect.y << ","
                                  << b0.rect.width << "," << b0.rect.height
                                  << ") conf=" << b0.prob
                                  << " label=" << b0.label << std::endl;
                    }
                }
                if (objs.size() > 0) {

                    // std::cout << objs[0].rect.x << ", " << objs[0].rect.y <<
                    // std::endl; f32 bbox_center_x = objs[0].rect.x +
                    // objs[0].rect.width / 2.0; std::cout << bbox_center_x <<
                    // std::endl; if (objs[0].rect.x < 2260.41 && objs[0].rect.x
                    // < objs_last_frame[0].rect.x) { if (objs[0].rect.x <
                    // 2500.0 && objs[0].rect.x > 2100.0) {
                    if (objs[0].rect.x < 2600.0 &&
                        objs[0].rect.x > 2100.0) { // trigger earlier
                        // std::cout << "trigger ball drop" << std::endl;
                        if (indigo_signal_builder->indigo_connection != NULL) {
                            send_indigo_message(
                                indigo_signal_builder->server,
                                indigo_signal_builder->builder,
                                indigo_signal_builder->indigo_connection,
                                FetchGame::SignalType_INDIGO_TRIAL_TRIGGER);
                        }
                    }
                    objs_last_frame.push_back(objs[0]);
                } else {
                    objs_last_frame.clear();
                }
            }
            
            // OBB Detection (async, non-blocking)
            if (obb_overlay_enabled) {
                obb_detector->set_yolo_boxes(objs);
                obb_detector->notify_frame_ready(debayer.d_debayer, 0);
                
                std::vector<OBB> obb_detections = obb_detector->get_latest_detections();
                last_obb_obj_count = obb_detections.size();
                if (!obb_detections.empty()) {
                    obb_nonempty_counter++;
                }
                
                if (obb_detections.size() > 0) {
                    // Check if detections changed significantly
                    bool detections_changed = (last_locked_detections.size() != obb_detections.size());
                    if (!detections_changed) {
                        for (size_t i = 0; i < obb_detections.size() && i < last_locked_detections.size(); i++) {
                            auto cur = obb_detector->obb_to_xywhr(obb_detections[i]);
                            auto prev = obb_detector->obb_to_xywhr(last_locked_detections[i]);
                            if (std::hypot(cur.x - prev.x, cur.y - prev.y) > 50.0f) {
                                detections_changed = true;
                                break;
                            }
                        }
                    }
                    
                    if (detections_changed || !last_detections_sent) {
                        last_locked_detections = obb_detections;
                        last_detections_sent = true;
                        
                        if (detections_changed) {
                            slots_initialized = false;
                            persistent_slot_assignments[0] = -1;
                            persistent_slot_assignments[1] = -1;
                        }
                    }
                    
                    // Draw OBBs on video stream
                    const auto& stable_detections = last_locked_detections;
                    for (size_t i = 0; i < stable_detections.size() && i < 10; i++) {
                        const OBB& obb = stable_detections[i];
                        float obb_points[8] = {
                            obb.x1, obb.y1, obb.x2, obb.y2,
                            obb.x3, obb.y3, obb.x4, obb.y4
                        };
                        CHECK(cudaMemcpyAsync(d_obb_points + i * 8, obb_points,
                                             sizeof(float) * 8, cudaMemcpyHostToDevice, 0));
                        gpu_draw_obb(debayer.d_debayer, camera_params->width,
                                    camera_params->height, d_obb_points + i * 8,
                                    obb.class_id, 0, 255, 0, 0);
                    }
                    
                    // Build FlatBuffer message (up to 2 slots)
                    flatbuffers::FlatBufferBuilder* fb = indigo_signal_builder->builder;
                    fb->Clear();
                    ::flatbuffers::Offset<Obj::obb> fb_obj_a{};
                    ::flatbuffers::Offset<Obj::obb> fb_obj_b{};
                    
                    for (size_t i = 0; i < stable_detections.size() && i < 10; i++) {
                        const OBB& obb = stable_detections[i];
                        auto xywhr = obb_detector->obb_to_xywhr(obb);
                        float fb_label = obb.shape_verified ? 0.0f : 1.0f;
                        
                        // Assign slot
                        int assigned_slot = -1;
                        if (!slots_initialized) {
                            if (i == 0) { assigned_slot = 0; persistent_slot_assignments[0] = i; }
                            else if (i == 1) { assigned_slot = 1; persistent_slot_assignments[1] = i; }
                        } else {
                            if (persistent_slot_assignments[0] == (int)i) assigned_slot = 0;
                            else if (persistent_slot_assignments[1] == (int)i) assigned_slot = 1;
                        }
                        
                        if (assigned_slot != -1) {
                            auto fb_obj = Obj::Createobb(*fb, xywhr.x, xywhr.y, xywhr.w, xywhr.h, xywhr.r, fb_label);
                            if (assigned_slot == 0) { fb_obj_a = fb_obj; obb_slot_valid[0] = 1; obb_slot_cx[0] = xywhr.x; obb_slot_cy[0] = xywhr.y; }
                            else { fb_obj_b = fb_obj; obb_slot_valid[1] = 1; obb_slot_cx[1] = xywhr.x; obb_slot_cy[1] = xywhr.y; }
                        }
                    }
                    
                    if (!slots_initialized) slots_initialized = true;
                    
                    if (!fb_obj_a.o) { fb_obj_a = Obj::Createobb(*fb, 0,0,0,0,0,0); obb_slot_valid[0] = 0; }
                    if (!fb_obj_b.o) { fb_obj_b = Obj::Createobb(*fb, 0,0,0,0,0,0); obb_slot_valid[1] = 0; }
                    
                    auto obj_msg = Obj::Createobj_msg(*fb, fb_obj_a, fb_obj_b);
                    fb->Finish(obj_msg);
                    
                    // Send to CBOT
                    if (indigo_signal_builder->indigo_connection) {
                        send_cbot_obj_pos2d(indigo_signal_builder->server, fb, indigo_signal_builder->indigo_connection);
                    }
                } else {
                    last_locked_detections.clear();
                    last_detections_sent = false;
                }
            }
            // nvtxRangePush("display_gl_copy_to_interop_buffer");
            if (camera_select->downsample != 1) {
                const NppStatus npp_result = nppiResize_8u_C4R(
                    debayer.d_debayer, camera_params->width * sizeof(uchar4),
                    input_image_size, input_image_roi, (Npp8u *)d_resize,
                    output_image_size.width * sizeof(uchar4), output_image_size,
                    output_image_roi, NPPI_INTER_SUPER);
                if (npp_result != NPP_SUCCESS) {
                    std::cerr << "Error executing resize in display -- code: "
                              << npp_result << std::endl;
                }
                CHECK(cudaMemcpy2D(
                    display_buffer, output_image_size.width * 4, d_resize,
                    output_image_size.width * 4, output_image_size.width * 4,
                    output_image_size.height, cudaMemcpyDeviceToDevice));

            } else {
                CHECK(cudaMemcpy2D(
                    display_buffer, output_image_size.width * 4,
                    debayer.d_debayer, output_image_size.width * 4,
                    output_image_size.width * 4, output_image_size.height,
                    cudaMemcpyDeviceToDevice));
            }
            // nvtxRangePop();
            cudaDeviceSynchronize();
        }
        // Count frame for FPS
        frameCount++;
        auto now = clock::now();
        std::chrono::duration<double> timeSinceLastFPSUpdate =
            now - lastFPSUpdate;
        if (timeSinceLastFPSUpdate.count() >= 1.0) {
            streaming_fps.store(frameCount / timeSinceLastFPSUpdate.count());
            frameCount = 0;
            lastFPSUpdate = now;
        }
        std::chrono::duration<double> timeSinceLastDetStatsUpdate =
            now - lastDetectionStatsUpdate;
        if (timeSinceLastDetStatsUpdate.count() >= 1.0 && yolo_glthread_enabled) {
            std::cout << "Detection stats [" << camera_params->camera_serial
                      << "]: yolo_nonempty=" << yolo_nonempty_counter << "/"
                      << yolo_frame_counter << " last_yolo_n=" << last_yolo_obj_count;
            if (obb_overlay_enabled) {
                std::cout << " obb_nonempty=" << obb_nonempty_counter << "/"
                          << yolo_frame_counter << " last_obb_n="
                          << last_obb_obj_count;
            }
            std::cout << std::endl;
            lastDetectionStatsUpdate = now;
        }
        // Frame duration (GPU time included)
        std::chrono::duration<double, std::milli> frameDuration =
            now - frameStart;
        if (frameDuration < targetFrameDuration) {
            std::this_thread::sleep_for(targetFrameDuration - frameDuration);
        }
    }
}

bool COpenGLDisplay::PushToDisplay(void *imagePtr, size_t bufferSize, int width,
                                   int height, int pixelFormat,
                                   unsigned long long timestamp,
                                   unsigned long long frame_id) {
    WORKER_ENTRY *entriesOut[WORK_ENTRIES_MAX]; // entris got out from saver
                                                // thread, their frames should
                                                // be returned to driver queue.
    int entriesOutCount = WORK_ENTRIES_MAX;
    GetObjectsFromQueueOut((void **)entriesOut, &entriesOutCount);
    if (entriesOutCount) { // return the frames to driver, and put entries back
                           // to frameSaveEntriesFreeQueue
        // printf("++++++++++++++++++++++++ %s %s %d get WORKER_ENTRY from out
        // entriesOutCount: %d\n", __FILE__, __FUNCTION__, __LINE__,
        // entriesOutCount);
        for (int j = 0; j < entriesOutCount; j++) {
            workerEntriesFreeQueue[workerEntriesFreeQueueCount] = entriesOut[j];
            workerEntriesFreeQueueCount++;
        }
    }

    // get the free entry if there is one and put in to QueueIn, otherwise
    // EVT_CameraQueueFrame.
    if (workerEntriesFreeQueueCount) {
        // printf("++++++++++++++++++++++++ %s %s %d put WORKER_ENTRY to in
        // workerEntriesFreeQueueCount: %d\n", __FILE__, __FUNCTION__, __LINE__,
        // workerEntriesFreeQueueCount);
        WORKER_ENTRY *entry =
            workerEntriesFreeQueue[workerEntriesFreeQueueCount - 1];
        workerEntriesFreeQueueCount--;
        entry->imagePtr = imagePtr;
        entry->bufferSize = bufferSize;
        entry->width = width;
        entry->height = height;
        entry->pixelFormat = pixelFormat;
        entry->timestamp = timestamp;
        entry->frame_id = frame_id;
        PutObjectToQueueIn(entry);
        return true;
    }
    return false;
}
