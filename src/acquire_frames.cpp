#include "emergent_camera.h"
#include "camera_params.h"
#include "video_capture.h"
#include "ptp_manager.h"
#include "gpu_manager.h"
#include "gpu_video_encoder.h"
#include "opengldisplay.h"
#include "network_base.h"
#include <memory>
#include <thread>
#include <chrono>
#include <iostream>
#include <atomic>

// Helper function for PTP timestamp checking
static void PTP_timestamp_checking(evt::PTPState* ptp_state, evt::EmergentCamera& camera, CameraState* camera_state) {
    uint64_t current_time = camera.getCurrentPTPTime();
    
    if (camera_state->frame_count > 0) {
        ptp_state->ptp_time_delta = current_time - ptp_state->ptp_time_prev;
        ptp_state->ptp_time_delta_sum += ptp_state->ptp_time_delta;
    }
    
    ptp_state->ptp_time = current_time;
    ptp_state->ptp_time_prev = current_time;
}

// Helper function for frame handling
static void handleAcquiredFrame(CameraState& camera_state,
                              evt::EmergentCamera& camera,
                              CameraEachSelect* camera_select,
                              CameraControl* camera_control,
                              evt::CameraParams* camera_params,
                              evt::PTPState& ptp_state,
                              COpenGLDisplay* openGLDisplay,
                              std::unique_ptr<evt::GPUVideoEncoder>& gpu_encoder,
                              uint64_t real_time,
                              Emergent::CEmergentFrame& frame) {
    // Check for dropped frames via frame ID
    if ((frame.frame_id != camera_state.id_prev + 1) && 
        (camera_state.frame_count != 0)) {
        camera_state.dropped_frames++;
    } else {
        camera_state.frames_recd++;
    }

    camera_state.frame_count++;

    // Handle frame ID wraparound
    if (frame.frame_id == 65535) {
        camera_state.id_prev = 0;
    } else {
        camera_state.id_prev = frame.frame_id;
    }

    // Push frame to encoder if recording
    if (camera_control->record_video && gpu_encoder) {
        gpu_encoder->PushToDisplay(frame.imagePtr,
            frame.bufferSize,
            frame.size_x,
            frame.size_y,
            frame.pixel_type,
            frame.timestamp,
            camera_state.frame_count,
            real_time);
    }

    // Push frame to display if streaming
    if (camera_select->stream_on && openGLDisplay) {
        openGLDisplay->PushToDisplay(frame.imagePtr,
            frame.bufferSize,
            frame.size_x,
            frame.size_y,
            frame.pixel_type,
            frame.timestamp,
            camera_state.frame_count);
    }
}

void acquireFrames(evt::EmergentCamera& camera, 
                  evt::CameraParams* camera_params,
                  CameraEachSelect* camera_select,
                  CameraControl* camera_control,
                  unsigned char* display_buffer,
                  const std::string& encoder_setup,
                  const std::string& folder_name,
                  PTPParams* ptp_params,
                  INDIGOSignalBuilder* indigo_signal_builder) {
    try {
        CameraState camera_state{};
        evt::PTPState ptp_state{};
        
        // Initialize display if needed
        std::unique_ptr<COpenGLDisplay> openGLDisplay;
        if (camera_select->stream_on) {
            openGLDisplay = std::make_unique<COpenGLDisplay>("", 
                camera_params, camera_select, display_buffer, indigo_signal_builder);
            if (openGLDisplay) {
                openGLDisplay->StartThread();
            }
        }

        // Initialize encoder if recording
        std::unique_ptr<evt::GPUVideoEncoder> gpu_encoder;
        bool encoder_ready_signal = false;
        if (camera_control->record_video) {
            gpu_encoder = std::make_unique<evt::GPUVideoEncoder>("", 
                camera_params, encoder_setup, folder_name, &encoder_ready_signal);
            if (gpu_encoder) {
                gpu_encoder->StartThread();
                
                auto start = std::chrono::steady_clock::now();
                while (!encoder_ready_signal) {
                    if (std::chrono::steady_clock::now() - start > std::chrono::seconds(5)) {
                        throw std::runtime_error("Encoder initialization timeout");
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }
        }

        // Main acquisition loop
        while (camera_control->subscribe) {
            Emergent::CEmergentFrame frame;
            
            // Try to get a frame
            if (camera.getFrame(&frame, 1000)) {  // Pass pointer to frame
                // Get system timestamp
                struct timespec ts_rt1;
                clock_gettime(CLOCK_REALTIME, &ts_rt1);
                uint64_t real_time = (ts_rt1.tv_sec * 1000000000ULL) + ts_rt1.tv_nsec;

                // Check PTP timing if enabled
                if (camera_control->sync_camera) {
                    PTP_timestamp_checking(&ptp_state, camera, &camera_state);
                }

                // Handle the acquired frame
                handleAcquiredFrame(camera_state, camera, camera_select, 
                    camera_control, camera_params, ptp_state,
                    openGLDisplay.get(), gpu_encoder, real_time, frame);

                // Queue the frame back
                camera.queueFrame(&frame);

                // Check for PTP stop condition
                if (ptp_params->network_sync && ptp_params->network_set_stop_ptp) {
                    if (ptp_state.ptp_time > ptp_params->ptp_stop_time) {
                        uint64_t ptp_stop_counter = std::atomic_fetch_add(
                            reinterpret_cast<std::atomic<uint64_t>*>(&ptp_params->ptp_stop_counter), 
                            static_cast<uint64_t>(1));
                        printf("%lu\n", ptp_stop_counter);
                        
                        while (ptp_params->ptp_stop_counter != camera_params->num_cameras) {
                            std::this_thread::sleep_for(std::chrono::microseconds(10));
                        }
                        
                        ptp_params->ptp_stop_reached = true;
                        camera_control->subscribe = false;
                        break;
                    }
                }
            } else {
                // Handle frame acquisition failure
                camera_state.dropped_frames++;
                std::cerr << "Frame acquisition timeout for camera " 
                         << camera_params->camera_serial << std::endl;
            }
        }

        // Calculate timing statistics
        double elapsed_time = camera_state.frame_count > 0 ? 
            static_cast<double>(camera_state.frame_count) / camera_params->frame_rate : 0.0;

        // Report statistics
        // report_statistics(camera_params, &camera_state, elapsed_time);

    } catch (const std::exception& e) {
        std::cerr << "Error in acquire_frames: " << e.what() << std::endl;
        throw;
    }
}