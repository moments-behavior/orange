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
                                    
                // Draw OBB overlays on GPU (lightweight, non-blocking) and populate flatbuffer message
                if (obb_detections.size() > 0) {
                    // Prepare flatbuffer message with up to 2 objects
                    flatbuffers::FlatBufferBuilder* fb = indigo_signal_builder->builder;
                    fb->Clear();

                    // Hold up to two object offsets
                    ::flatbuffers::Offset<Obj::obb> fb_obj_a{};
                    ::flatbuffers::Offset<Obj::obb> fb_obj_b{};
                    for (size_t i = 0; i < obb_detections.size() && i < 10; i++) {
                        const OBB& obb = obb_detections[i];
                        
                        // Print detection coordinates in xywhr format (synchronized with drawing)
                        auto xywhr = obb_detector->obb_to_xywhr(obb);
                        std::cout << "OBB: Object " << obb.object_id << " detected - Class " << obb.class_id 
                                  << " at xywhr(" << xywhr.x << ", " << xywhr.y << ", " 
                                  << xywhr.w << ", " << xywhr.h << ", " << xywhr.r << ")" << std::endl;
                        
                        // Check if object is flickering between classes
                        bool is_flickering = obb_detector->is_object_flickering(obb.object_id);
                        // Map label: 0 = stable (oriented), 1 = flickering (green AABB)
                        float fb_label = is_flickering ? 1.0f : 0.0f;
                        
                        // Decide which slot (0 or 1) this object should occupy based on centroid proximity
                        auto center = obb_detector->obb_to_xywhr(obb);
                        float cx = center.x;
                        float cy = center.y;
                        int assigned_slot = -1;
                        if (obb_slot_valid[0] && !obb_slot_valid[1]) {
                            float d0 = std::hypot(cx - obb_slot_cx[0], cy - obb_slot_cy[0]);
                            assigned_slot = (d0 <= OBB_SLOT_ASSIGN_DISTANCE) ? 0 : -1;
                        } else if (!obb_slot_valid[0] && obb_slot_valid[1]) {
                            float d1 = std::hypot(cx - obb_slot_cx[1], cy - obb_slot_cy[1]);
                            assigned_slot = (d1 <= OBB_SLOT_ASSIGN_DISTANCE) ? 1 : -1;
                        } else if (obb_slot_valid[0] && obb_slot_valid[1]) {
                            float d0 = std::hypot(cx - obb_slot_cx[0], cy - obb_slot_cy[0]);
                            float d1 = std::hypot(cx - obb_slot_cx[1], cy - obb_slot_cy[1]);
                            if (d0 <= d1 && d0 <= OBB_SLOT_ASSIGN_DISTANCE) assigned_slot = 0;
                            else if (d1 < d0 && d1 <= OBB_SLOT_ASSIGN_DISTANCE) assigned_slot = 1;
                            else assigned_slot = -1;
                        }
                        // If no valid slot found, occupy a free slot if any
                        if (assigned_slot == -1) {
                            if (!obb_slot_valid[0]) assigned_slot = 0;
                            else if (!obb_slot_valid[1]) assigned_slot = 1;
                        }

                        if (is_flickering) {
                            // Draw non-oriented bounding box for flickering objects
                            // Convert OBB to axis-aligned bounding box
                            float min_x = std::min({obb.x1, obb.x2, obb.x3, obb.x4});
                            float max_x = std::max({obb.x1, obb.x2, obb.x3, obb.x4});
                            float min_y = std::min({obb.y1, obb.y2, obb.y3, obb.y4});
                            float max_y = std::max({obb.y1, obb.y2, obb.y3, obb.y4});
                            
                            // Create axis-aligned bounding box points
                            float aabb_points[8] = {
                                min_x, min_y,  // Top-left
                                max_x, min_y,  // Top-right
                                max_x, max_y,  // Bottom-right
                                min_x, max_y   // Bottom-left
                            };
                            
                            CHECK(cudaMemcpyAsync(d_obb_points + i * 8, aabb_points, 
                                                 sizeof(float) * 8, cudaMemcpyHostToDevice, 0));
                            
                            // Draw axis-aligned bounding box in green color for flickering objects
                            gpu_draw_obb(debayer.d_debayer, camera_params->width, 
                                        camera_params->height, d_obb_points + i * 8, 
                                        obb.class_id, 0, 0, 255, 0);  // Green color for flickering

                            // Populate flatbuffer with AABB (theta = 0)
                            if (assigned_slot != -1) {
                                float cx = 0.5f * (min_x + max_x);
                                float cy = 0.5f * (min_y + max_y);
                                float w = (max_x - min_x);
                                float h = (max_y - min_y);
                                float theta = 0.0f;
                                auto fb_obj = Obj::Createobb(*fb, cx, cy, w, h, theta, fb_label);
                                if (assigned_slot == 0) { fb_obj_a = fb_obj; obb_slot_valid[0] = 1; obb_slot_cx[0] = cx; obb_slot_cy[0] = cy; }
                                else { fb_obj_b = fb_obj; obb_slot_valid[1] = 1; obb_slot_cx[1] = cx; obb_slot_cy[1] = cy; }
                            }
                        } else {
                            // Draw oriented bounding box for stable objects
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

                            // Populate flatbuffer with oriented OBB (theta from xywhr)
                            if (assigned_slot != -1) {
                                auto fb_obj = Obj::Createobb(*fb, xywhr.x, xywhr.y, xywhr.w, xywhr.h, xywhr.r, fb_label);
                                if (assigned_slot == 0) { fb_obj_a = fb_obj; obb_slot_valid[0] = 1; obb_slot_cx[0] = xywhr.x; obb_slot_cy[0] = xywhr.y; }
                                else { fb_obj_b = fb_obj; obb_slot_valid[1] = 1; obb_slot_cx[1] = xywhr.x; obb_slot_cy[1] = xywhr.y; }
                            }
                        }
                    }

                    // Ensure both objects exist in message (use zero object if not filled)
                    if (!fb_obj_a.o) {
                        fb_obj_a = Obj::Createobb(*fb, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                        obb_slot_valid[0] = 0;
                    }
                    if (!fb_obj_b.o) {
                        fb_obj_b = Obj::Createobb(*fb, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
                        obb_slot_valid[1] = 0;
                    }

                    std::cout << "DEBUG: Creating obj_msg FlatBuffer" << std::endl;
                    auto obj_msg = Obj::Createobj_msg(*fb, fb_obj_a, fb_obj_b);
                    std::cout << "DEBUG: Finishing FlatBuffer" << std::endl;
                    fb->Finish(obj_msg);
                    
                    // Only send message to CBOT if connected
                    if (indigo_signal_builder->indigo_connection) {
                        std::cout << "DEBUG: Sending OBB message to CBOT" << std::endl;
                        send_cbot_obj_pos2d(indigo_signal_builder->server, fb, indigo_signal_builder->indigo_connection);
                        std::cout << "DEBUG: OBB message sent successfully" << std::endl;
                    } else {
                        std::cout << "DEBUG: CBOT not connected, skipping OBB message" << std::endl;
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
