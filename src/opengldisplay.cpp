#include "video_capture.h"
#if defined(__GNUC__)
#include <unistd.h>
#endif
#include "global.h"
#include "kernel.cuh"
#include "opengldisplay.h"
#include "utils.h"
#include <cuda_runtime_api.h>
#include <nvToolsExt.h>
#include <stdio.h>
#include <string.h>
#include <iomanip>

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
    
    // Initialize OBB detector if enabled
    if (camera_select->enable_obb) {
        std::cout << "=== OBB Detector Initialization ===" << std::endl;
        std::cout << "OBB enabled for camera: " << camera_params->camera_serial << std::endl;
        std::cout << "CSV path: " << camera_select->obb_csv_path << std::endl;
        std::cout << "Threshold: " << camera_select->obb_threshold << std::endl;
        std::cout << "Background frames: " << camera_select->obb_bg_frames << std::endl;
        
        // Set up OBB detector parameters
        OBBDetectorParams obb_params;
        obb_params.threshold = camera_select->obb_threshold;
        obb_params.bg_frames = camera_select->obb_bg_frames;
        
        // Create CSV paths vector
        std::vector<std::string> csv_paths = {camera_select->obb_csv_path};
        
        // Create OBB detector with proper parameters
        obb_detector = new OBBDetector(camera_params, csv_paths, obb_params);
        
        // Initialize the detector (learn priors from CSV files)
        if (!obb_detector->initialize()) {
            std::cout << "Failed to initialize OBB detector with CSV: " 
                      << camera_select->obb_csv_path << std::endl;
            delete obb_detector;
            obb_detector = nullptr;
        } else {
            std::cout << "OBB detector initialized successfully (async mode)" << std::endl;
            
            // Start the OBB detection thread for async processing
            obb_detector->start();
        }
        std::cout << "=== End OBB Initialization ===" << std::endl;
    } else {
        std::cout << "OBB detection disabled for camera: " << camera_params->camera_serial << std::endl;
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
    if (camera_select->detect_mode == Detect2D_GLThread) {
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
    if (camera_select->enable_obb && obb_detector) {
        cudaMalloc((void **)&d_obb_points, sizeof(float) * 8 * 10); // Support up to 10 OBBs
    }

    std::vector<Bbox> objs;
    std::vector<Bbox> objs_last_frame;
    std::vector<OBB> obb_detections;  // Add OBB results

    using clock = std::chrono::steady_clock;
    
    // OBB Background Building

    int frameCount = 0;
    auto lastFPSUpdate = clock::now();

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

            if (camera_select->detect_mode == Detect2D_GLThread) {
                rgba2rgb_convert(d_convert, debayer.d_debayer,
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
                if (objs.size() > 0) {

                    for (int obj = 0; obj < objs.size(); obj++) {
                        // draw all bounding boxes when objects are detected
                        // default to highlighting by class color
                        yolov8->copy_keypoints_gpu(d_points, objs[obj]);
                        gpu_draw_box(debayer.d_debayer, camera_params->width,
                                     camera_params->height, d_points,
                                     objs[obj].label, yolov8->stream);
                    }

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
            if (camera_select->enable_obb && obb_detector) {
                // Notify OBB detector of new frame (async, non-blocking)
                obb_detector->notify_frame_ready(debayer.d_debayer, 0);
                
                // Get latest detections (non-blocking, returns immediately)
                std::vector<OBB> obb_detections = obb_detector->get_latest_detections();
                
                // Draw OBB overlays on GPU (lightweight, non-blocking)
                if (obb_detections.size() > 0) {
                    for (size_t i = 0; i < obb_detections.size() && i < 10; i++) {
                        const OBB& obb = obb_detections[i];
                        
                        // Print detection coordinates in xywhr format (synchronized with drawing)
                        auto xywhr = obb_detector->obb_to_xywhr(obb);
                        std::cout << "OBB: Object detected - Class " << obb.class_id 
                                  << " at xywhr(" << xywhr.x << ", " << xywhr.y << ", " 
                                  << xywhr.w << ", " << xywhr.h << ", " << xywhr.r << ")" << std::endl;
                        
                        // Copy OBB corners to GPU (use original coordinates, resize will handle scaling)
                        float obb_points[8] = {
                            obb.x1, obb.y1,  // Top-left
                            obb.x2, obb.y2,  // Top-right  
                            obb.x3, obb.y3,  // Bottom-right
                            obb.x4, obb.y4   // Bottom-left
                        };
                        
                        CHECK(cudaMemcpyAsync(d_obb_points + i * 8, obb_points, 
                                             sizeof(float) * 8, cudaMemcpyHostToDevice, 0));
                        
                        // Draw OBB on original buffer (will be resized later if needed)
                        gpu_draw_obb(debayer.d_debayer, camera_params->width, 
                                    camera_params->height, d_obb_points + i * 8, 
                                    obb.class_id, 0);
                    }
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
