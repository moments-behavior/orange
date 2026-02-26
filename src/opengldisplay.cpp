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
    
    // Static variables for OBB lock-in system tracking
    static std::vector<OBB> last_locked_detections;
    static bool last_detections_sent = false;
    
    // Static variables for persistent slot tracking
    static int persistent_slot_assignments[2] = {-1, -1};  // Object IDs assigned to each slot
    static bool slots_initialized = false;
    
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
                // Feed YOLO boxes to OBB detector for two-stage refinement
                if (!objs.empty()) {
                    obb_detector->set_yolo_boxes(objs);
                }
                // Notify OBB detector of new frame (async, non-blocking)
                obb_detector->notify_frame_ready(debayer.d_debayer, 0);
                
                // Get latest detections (non-blocking, returns immediately)
                // This now uses the lock-in system internally
                std::vector<OBB> obb_detections = obb_detector->get_latest_detections();
                                    
                // Only draw and send messages when we have stable locked detections
                if (obb_detections.size() > 0) {
                    static auto last_message_send_time = std::chrono::steady_clock::now();
                    
                    // Check if detections have changed significantly (lock-in system handles this)
                    bool detections_changed = (last_locked_detections.size() != obb_detections.size());
                    if (!detections_changed) {
                        // Check if positions changed significantly
                        for (size_t i = 0; i < obb_detections.size() && i < last_locked_detections.size(); i++) {
                            auto current_center = obb_detector->obb_to_xywhr(obb_detections[i]);
                            auto last_center = obb_detector->obb_to_xywhr(last_locked_detections[i]);
                            float distance = std::hypot(current_center.x - last_center.x, current_center.y - last_center.y);
                            if (distance > 50.0f) {  // Significant position change
                                detections_changed = true;
                                break;
                            }
                        }
                    }
                    
                    // Only process and send message when detections change
                    if (detections_changed || !last_detections_sent) {
                        // Update our tracking
                        last_locked_detections = obb_detections;
                        last_detections_sent = true;
                        
                        // Reset slot assignments when detections change significantly
                        if (detections_changed) {
                            slots_initialized = false;
                            persistent_slot_assignments[0] = -1;
                            persistent_slot_assignments[1] = -1;
                        }
                        
                        std::cout << "OBB: Updating " << obb_detections.size() << " locked detections (changed: " 
                                  << (detections_changed ? "yes" : "no") << ")" << std::endl;
                        
                        // Only prepare and send flatbuffer message when content changes
                        flatbuffers::FlatBufferBuilder* fb = indigo_signal_builder->builder;
                        fb->Clear();

                        // Hold up to two object offsets
                        ::flatbuffers::Offset<Obj::obb> fb_obj_a{};
                        ::flatbuffers::Offset<Obj::obb> fb_obj_b{};
                        // Build message strictly from stable (locked) detections
                        const auto& stable_detections = last_locked_detections;
                        for (size_t i = 0; i < stable_detections.size() && i < 10; i++) {
                            const OBB& obb = stable_detections[i];
                            
                            // Print detection coordinates in xywhr format (synchronized with drawing)
                            auto xywhr = obb_detector->obb_to_xywhr(obb);
                            // Only print every 30 frames to reduce noise
                            static int frame_counter = 0;
                            if (frame_counter % 30 == 0) {
                                std::cout << "OBB: Object " << obb.object_id << " detected - Class " << obb.class_id 
                                          << " at xywhr(" << xywhr.x << ", " << xywhr.y << ", " 
                                          << xywhr.w << ", " << xywhr.h << ", " << xywhr.r << ")" << std::endl;
                                std::cout << "OBB: Raw corners: (" << obb.x1 << "," << obb.y1 << ") (" 
                                          << obb.x2 << "," << obb.y2 << ") (" << obb.x3 << "," << obb.y3 
                                          << ") (" << obb.x4 << "," << obb.y4 << ")" << std::endl;
                            }
                            frame_counter++;
                            
                            // Use shape verification result from OBB detector
                            bool shape_verified = obb.shape_verified;
                            // Label: 0 = shape verified, 1 = shape not verified (but we keep the detection)
                            float fb_label = shape_verified ? 0.0f : 1.0f;
                            
                            // Always draw all detected objects, but still assign slots for FlatBuffer compatibility
                            auto center = obb_detector->obb_to_xywhr(obb);
                            float cx = center.x;
                            float cy = center.y;
                            int assigned_slot = -1;
                            
                            // Persistent slot assignment: only change when detections change significantly
                            if (!slots_initialized) {
                                // First time or after significant change - assign slots based on detection index
                                if (i == 0) {
                                    assigned_slot = 0;
                                    persistent_slot_assignments[0] = i;
                                } else if (i == 1) {
                                    assigned_slot = 1;
                                    persistent_slot_assignments[1] = i;
                                } else {
                                    assigned_slot = -1;  // More than 2 objects
                                }
                            } else {
                                // Use persistent slot assignments based on detection index
                                if (persistent_slot_assignments[0] == i) {
                                    assigned_slot = 0;
                                } else if (persistent_slot_assignments[1] == i) {
                                    assigned_slot = 1;
                                } else {
                                    assigned_slot = -1;  // Object not in persistent slots
                                }
                            }
                            
                            // Always draw the object regardless of slot assignment
                            bool should_draw = true;
                            std::cout << "DEBUG: Processing object " << i << " - Class " << obb.class_id 
                                      << ", Detection Index " << i 
                                      << ", Assigned slot " << assigned_slot 
                                      << ", Slots initialized: " << (slots_initialized ? "YES" : "NO")
                                      << ", Should draw: " << (should_draw ? "YES" : "NO") << std::endl;

                            if (should_draw) {
                                // Draw all objects as red oriented bounding boxes
                                float obb_points[8] = {
                                    obb.x1, obb.y1,  // Top-left
                                    obb.x2, obb.y2,  // Top-right
                                    obb.x3, obb.y3,  // Bottom-right
                                    obb.x4, obb.y4   // Bottom-left
                                };
                                
                                CHECK(cudaMemcpyAsync(d_obb_points + i * 8, obb_points, 
                                                     sizeof(float) * 8, cudaMemcpyHostToDevice, 0));
                                
                                // Draw oriented bounding box in red for all objects
                                std::cout << "DEBUG: Drawing OBB at corners: (" << obb.x1 << "," << obb.y1 << ") (" 
                                          << obb.x2 << "," << obb.y2 << ") (" << obb.x3 << "," << obb.y3 << ") (" 
                                          << obb.x4 << "," << obb.y4 << ")" << std::endl;
                                gpu_draw_obb(debayer.d_debayer, camera_params->width, 
                                            camera_params->height, d_obb_points + i * 8, 
                                            obb.class_id, 0, 255, 0, 0);  // Red for all oriented bounding boxes
                                
                                // Populate flatbuffer with oriented OBB (theta from xywhr)
                                if (assigned_slot != -1) {
                                    auto fb_obj = Obj::Createobb(*fb, xywhr.x, xywhr.y, xywhr.w, xywhr.h, xywhr.r, fb_label);
                                    if (assigned_slot == 0) { fb_obj_a = fb_obj; obb_slot_valid[0] = 1; obb_slot_cx[0] = xywhr.x; obb_slot_cy[0] = xywhr.y; }
                                    else { fb_obj_b = fb_obj; obb_slot_valid[1] = 1; obb_slot_cx[1] = xywhr.x; obb_slot_cy[1] = xywhr.y; }
                                }
                            }
                        }
                        
                        // Mark slots as initialized after processing all objects
                        if (!slots_initialized) {
                            slots_initialized = true;
                            std::cout << "DEBUG: Slots initialized - Slot 0: Detection Index " << persistent_slot_assignments[0] 
                                      << ", Slot 1: Detection Index " << persistent_slot_assignments[1] << std::endl;
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

                        auto obj_msg = Obj::Createobj_msg(*fb, fb_obj_a, fb_obj_b);
                        fb->Finish(obj_msg);
                        
                        // Save sample detection output to file for debugging (append mode)
                        if (obb_slot_valid[0] || obb_slot_valid[1]) {
                            // Try multiple locations for the output file
                            std::vector<std::string> possible_paths = {
                                "/tmp/sample_detection_output.txt",
                                "./sample_detection_output.txt",
                                "/home/ratan/sample_detection_output.txt",
                                "/home/ratan/src/lime/sample_detection_output.txt"
                            };
                            
                            bool file_saved = false;
                            for (const auto& sample_file_path : possible_paths) {
                                std::ofstream sample_file(sample_file_path, std::ios::app);
                                if (sample_file.is_open()) {
                                    sample_file << "=== Sample OBB Detection Output ===" << std::endl;
                                    sample_file << "Timestamp: " << std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::system_clock::now().time_since_epoch()).count() << std::endl;
                                    sample_file << "Stable detections count: " << stable_detections.size() << std::endl;
                                    sample_file << "Using locked detections: " << (detections_changed ? "NO (new detections)" : "YES (same as before)") << std::endl;
                                    sample_file << "Drawing colors: Red=Oriented Bounding Box (all objects)" << std::endl;
                                    sample_file << "Camera dimensions: " << camera_params->width << "x" << camera_params->height << std::endl;
                                    sample_file << "Slot A valid: " << obb_slot_valid[0] << std::endl;
                                    sample_file << "Slot B valid: " << obb_slot_valid[1] << std::endl;
                                    
                                    // Print individual detection details
                                    sample_file << "Individual Detections:" << std::endl;
                                    for (size_t i = 0; i < stable_detections.size(); i++) {
                                        const OBB& obb = stable_detections[i];
                                        auto xywhr = obb_detector->obb_to_xywhr(obb);
                                        sample_file << "  Detection " << i << ": Class " << obb.class_id 
                                                  << ", Object ID " << obb.object_id 
                                                  << ", Shape Verified: " << (obb.shape_verified ? "YES" : "NO")
                                                  << ", xywhr(" << xywhr.x << ", " << xywhr.y << ", " 
                                                  << xywhr.w << ", " << xywhr.h << ", " << xywhr.r << ")" << std::endl;
                                        sample_file << "    Raw corners: (" << obb.x1 << "," << obb.y1 << ") (" 
                                                  << obb.x2 << "," << obb.y2 << ") (" << obb.x3 << "," << obb.y3 
                                                  << ") (" << obb.x4 << "," << obb.y4 << ")" << std::endl;
                                    }
                                    
                                    // Print readable OBB structure content
                                    sample_file << "OBB Message Content:" << std::endl;
                                    
                                    // Access the finished flatbuffer message
                                    auto obj_msg = Obj::Getobj_msg(fb->GetBufferPointer());
                                    
                                    if (obj_msg->cylinder1()) {
                                        auto obj1 = obj_msg->cylinder1();
                                        sample_file << "Object 1 (cylinder1):" << std::endl;
                                        sample_file << "  cx: " << obj1->cx() << std::endl;
                                        sample_file << "  cy: " << obj1->cy() << std::endl;
                                        sample_file << "  w: " << obj1->w() << std::endl;
                                        sample_file << "  h: " << obj1->h() << std::endl;
                                        sample_file << "  theta: " << obj1->theta() << std::endl;
                                        sample_file << "  label: " << obj1->label() << std::endl;
                                    } else {
                                        sample_file << "Object 1 (cylinder1): null" << std::endl;
                                    }
                                    
                                    if (obj_msg->cylinder2()) {
                                        auto obj2 = obj_msg->cylinder2();
                                        sample_file << "Object 2 (cylinder2):" << std::endl;
                                        sample_file << "  cx: " << obj2->cx() << std::endl;
                                        sample_file << "  cy: " << obj2->cy() << std::endl;
                                        sample_file << "  w: " << obj2->w() << std::endl;
                                        sample_file << "  h: " << obj2->h() << std::endl;
                                        sample_file << "  theta: " << obj2->theta() << std::endl;
                                        sample_file << "  label: " << obj2->label() << std::endl;
                                    } else {
                                        sample_file << "Object 2 (cylinder2): null" << std::endl;
                                    }
                                    
                                    sample_file << std::endl;
                                    sample_file.close();
                                    std::cout << "Sample detection output appended to: " << sample_file_path << std::endl;
                                    file_saved = true;
                                    break;
                                }
                            }
                            
                            if (!file_saved) {
                                std::cerr << "Failed to create sample detection output file in any location. Tried:" << std::endl;
                                for (const auto& path : possible_paths) {
                                    std::cerr << "  - " << path << std::endl;
                                }
                            }
                        }
                        
                        // Send message to CBOT only when content changes
                        if (indigo_signal_builder->indigo_connection) {
                            std::cout << "DEBUG: Sending OBB message to CBOT (detections: " << obb_detections.size() << ")" << std::endl;
                            send_cbot_obj_pos2d(indigo_signal_builder->server, fb, indigo_signal_builder->indigo_connection);
                            std::cout << "DEBUG: OBB message sent successfully" << std::endl;
                        } else {
                            std::cout << "DEBUG: CBOT not connected, skipping OBB message" << std::endl;
                        }
                    } else {
                        // Use the same detections as before - send the same stable message every frame
                        const auto& stable_detections = last_locked_detections;
                        std::cout << "OBB: Using same " << stable_detections.size() << " locked detections (no change) - sending same message" << std::endl;

                        // Draw the objects
                        for (size_t i = 0; i < stable_detections.size() && i < 10; i++) {
                            const OBB& obb = stable_detections[i];

                            float obb_points[8] = {
                                obb.x1, obb.y1,  // Top-left
                                obb.x2, obb.y2,  // Top-right
                                obb.x3, obb.y3,  // Bottom-right
                                obb.x4, obb.y4   // Bottom-left
                            };

                            CHECK(cudaMemcpyAsync(d_obb_points + i * 8, obb_points,
                                                 sizeof(float) * 8, cudaMemcpyHostToDevice, 0));

                            gpu_draw_obb(debayer.d_debayer, camera_params->width,
                                        camera_params->height, d_obb_points + i * 8,
                                        obb.class_id, 0, 255, 0, 0);
                        }

                        // Rebuild and send message from the same stable detections
                        flatbuffers::FlatBufferBuilder* fb = indigo_signal_builder->builder;
                        fb->Clear();

                        ::flatbuffers::Offset<Obj::obb> fb_obj_a{};
                        ::flatbuffers::Offset<Obj::obb> fb_obj_b{};
                        for (size_t i = 0; i < stable_detections.size() && i < 2; i++) {
                            const OBB& obb = stable_detections[i];
                            auto xywhr = obb_detector->obb_to_xywhr(obb);
                            float fb_label = obb.shape_verified ? 0.0f : 1.0f;
                            auto fb_obj = Obj::Createobb(*fb, xywhr.x, xywhr.y, xywhr.w, xywhr.h, xywhr.r, fb_label);
                            if (i == 0) { fb_obj_a = fb_obj; obb_slot_valid[0] = 1; obb_slot_cx[0] = xywhr.x; obb_slot_cy[0] = xywhr.y; }
                            else if (i == 1) { fb_obj_b = fb_obj; obb_slot_valid[1] = 1; obb_slot_cx[1] = xywhr.x; obb_slot_cy[1] = xywhr.y; }
                        }

                        if (!fb_obj_a.o) { fb_obj_a = Obj::Createobb(*fb, 0,0,0,0,0,0); obb_slot_valid[0] = 0; }
                        if (!fb_obj_b.o) { fb_obj_b = Obj::Createobb(*fb, 0,0,0,0,0,0); obb_slot_valid[1] = 0; }

                        auto obj_msg = Obj::Createobj_msg(*fb, fb_obj_a, fb_obj_b);
                        fb->Finish(obj_msg);

                        if (indigo_signal_builder->indigo_connection) {
                            send_cbot_obj_pos2d(indigo_signal_builder->server, fb, indigo_signal_builder->indigo_connection);
                        }
                    }
                } else {
                    // No detections available, reset tracking
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
