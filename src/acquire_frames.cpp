#include "NvEncoder/NvCodecUtils.h"
#include "opengldisplay.h"
#include "gpu_video_encoder.h"
#include "acquire_frames.h"
#include "kernel.cuh"
#include <cuda_runtime_api.h>
#include "global.h"
#include "yolo_worker.h" // Ensure this is included for YOLOv8Worker
#include "image_processing.h" // For FrameGPU, Debayer, FrameCPU definitions

// Define FrameProcess struct here, before it's used
struct FrameProcess
{
    FrameGPU frame_original;
    Debayer debayer;
    unsigned char *d_convert;
    FrameCPU frame_cpu;
};

// Static inline PTP_timestamp_checking (no changes from your "our code")
static inline void PTP_timestamp_checking(PTPState *ptp_state, CameraEmergent *ecam, CameraState *camera_state)
{
    EVT_CameraExecuteCommand(&ecam->camera, "GevTimestampControlLatch");
    EVT_CameraGetUInt32Param(&ecam->camera, "GevTimestampValueHigh", &ptp_state->ptp_time_high);
    EVT_CameraGetUInt32Param(&ecam->camera, "GevTimestampValueLow", &ptp_state->ptp_time_low);

    ptp_state->ptp_time = (((unsigned long long)(ptp_state->ptp_time_high)) << 32) | ((unsigned long long)(ptp_state->ptp_time_low));
    ptp_state->frame_ts = ecam->frame_recv.timestamp;

    if (camera_state->frame_count != 0)
    {
        ptp_state->ptp_time_delta = ptp_state->ptp_time - ptp_state->ptp_time_prev;
        ptp_state->ptp_time_delta_sum += ptp_state->ptp_time_delta;

        ptp_state->frame_ts_delta = ptp_state->frame_ts - ptp_state->frame_ts_prev;
        ptp_state->frame_ts_delta_sum += ptp_state->frame_ts_delta;
    }

    ptp_state->ptp_time_prev = ptp_state->ptp_time;
    ptp_state->frame_ts_prev = ptp_state->frame_ts;
}

// Corrected get_one_frame signature and usage
static inline void get_one_frame(
    CameraState *camera_state,
    CameraEachSelect* camera_select,
    CameraControl *camera_control,
    CameraEmergent *ecam,
    CameraParams *camera_params,
    PTPState *ptp_state,
    COpenGLDisplay* openGLDisplay,
    GPUVideoEncoder* gpu_encoder,
    YOLOv8Worker* yolo_worker, // Parameter for YOLO worker
    FrameProcess* frame_process_for_calib // Parameter for calibration specific processing
)
{
    camera_state->camera_return = EVT_CameraGetFrame(&ecam->camera, &ecam->frame_recv, EVT_INFINITE);

    struct timespec ts_rt1;
    clock_gettime(CLOCK_REALTIME, &ts_rt1);
    uint64_t real_time = (ts_rt1.tv_sec * 1000000000LL) + ts_rt1.tv_nsec;

    if (camera_control->sync_camera)
    {
        PTP_timestamp_checking(ptp_state, ecam, camera_state);
    }

    if (!camera_state->camera_return)
    {
        if (((ecam->frame_recv.frame_id) != camera_state->id_prev + 1) && (camera_state->frame_count != 0))
            camera_state->dropped_frames++;
        else
            camera_state->frames_recd++;

        camera_state->frame_count++;

        if (ecam->frame_recv.frame_id == 65535)
            camera_state->id_prev = 0;
        else
            camera_state->id_prev = ecam->frame_recv.frame_id;

        WORKER_ENTRY entry;
        entry.imagePtr = ecam->frame_recv.imagePtr;
        entry.bufferSize = ecam->frame_recv.bufferSize;
        entry.width = ecam->frame_recv.size_x;
        entry.height = ecam->frame_recv.size_y;
        entry.pixelFormat = ecam->frame_recv.pixel_type;
        entry.timestamp = ecam->frame_recv.timestamp;
        entry.frame_id = camera_state->frame_count;
        entry.timestamp_sys = real_time;

        if (camera_control->record_video && camera_select->record && gpu_encoder != nullptr) {
            if (!gpu_encoder->PushToDisplay(entry.imagePtr, entry.bufferSize, entry.width, entry.height, entry.pixelFormat, entry.timestamp, entry.frame_id, entry.timestamp_sys)) {
                 std::cerr << "Error pushing frame to GPUVideoEncoder for camera " << camera_params->camera_serial << std::endl;
            }
        }

        if (camera_select->yolo && yolo_worker != nullptr) {
            yolo_worker->PutObjectToQueueIn(&entry);
        }
        else if (camera_select->stream_on && openGLDisplay != nullptr) {
             if (!openGLDisplay->PushToDisplay(entry.imagePtr, entry.bufferSize, entry.width, entry.height, entry.pixelFormat, entry.timestamp, entry.frame_id)) {
                std::cerr << "Error pushing raw frame to COpenGLDisplay for camera " << camera_params->camera_serial << std::endl;
            }
        }

        // temp changes to take images for calibration
        // Use frame_process_for_calib here
        if (camera_select->frame_save_state == State_Write_New_Frame && frame_process_for_calib != nullptr) {
            ck(cudaMemcpy2D(frame_process_for_calib->frame_original.d_orig, camera_params->width, ecam->frame_recv.imagePtr, camera_params->width, camera_params->width, camera_params->height, cudaMemcpyHostToDevice));
            if (camera_params->color){
                debayer_frame_gpu(camera_params, &frame_process_for_calib->frame_original, &frame_process_for_calib->debayer);
            } else {
                duplicate_channel_gpu(camera_params, &frame_process_for_calib->frame_original, &frame_process_for_calib->debayer);
            }
            rgba2bgr_convert(frame_process_for_calib->d_convert, frame_process_for_calib->debayer.d_debayer, camera_params->width, camera_params->height, 0);
            cudaMemcpy2D(frame_process_for_calib->frame_cpu.frame, camera_params->width*3, frame_process_for_calib->d_convert, camera_params->width*3, camera_params->width*3, camera_params->height, cudaMemcpyDeviceToHost);
            cv::Mat view = cv::Mat(camera_params->width * camera_params->height * 3, 1, CV_8U, frame_process_for_calib->frame_cpu.frame).reshape(3, camera_params->height);

            std::string image_name = camera_select->picture_save_folder + "/" + camera_params->camera_serial + "_" + camera_select->frame_save_name + "." + camera_select->frame_save_format;
            std::cout << image_name << std::endl;
            cv::imwrite(image_name, view);
            camera_select->pictures_counter++;
            camera_select->frame_save_state = State_Frame_Idle;
        }

        camera_state->camera_return = EVT_CameraQueueFrame(&ecam->camera, &ecam->frame_recv);
        if (camera_state->camera_return)
            std::cout << "EVT_CameraQueueFrame Error for camera " << camera_params->camera_serial << "!" << std::endl;

        if (camera_state->frame_count % 1000 == 99)
        {
            std::cout << "Camera " << camera_params->camera_serial << ": processed 1000 frames." << std::endl;
        }
    }
    else
    {
        camera_state->dropped_frames++;
        std::cout << "EVT_CameraGetFrame Error: " << camera_state->camera_return
                  << ", camera serial: " << camera_params->camera_serial << std::endl;
    }
}

// acquire_frames function signature matches acquire_frames.h
void acquire_frames(
    CameraEmergent *ecam,
    CameraParams *camera_params,
    CameraEachSelect* camera_select,
    CameraControl* camera_control,
    unsigned char *display_buffer, // Renamed in .h to display_buffer_cuda_pbo, but keep original name here if user prefers
    std::string encoder_setup,
    std::string folder_name,
    PTPParams* ptp_params,
    INDIGOSignalBuilder* indigo_signal_builder,
    YOLOv8Worker* yolo_worker_for_this_camera
)
{
    CameraState camera_state;
    PTPState ptp_state;
    StopWatch w;

    // Declare FrameProcess for calibration. It's now defined at the top of the file.
    FrameProcess frame_process_calib_data; // Renamed for clarity
    bool calib_resources_initialized = false;

    // Initialize calibration resources only if explicitly needed for this run
    // For example, if frame_save_state is already set or based on a camera_select flag.
    // For now, let's assume it's always initialized if acquire_frames is called,
    // or you can add a more specific condition.
    // if (camera_select->some_flag_indicating_calibration_mode) {
        ck(cudaSetDevice(camera_params->gpu_id));
        initalize_gpu_frame(&frame_process_calib_data.frame_original, camera_params);
        initialize_gpu_debayer(&frame_process_calib_data.debayer, camera_params);
        initialize_cpu_frame(&frame_process_calib_data.frame_cpu, camera_params);
        ck(cudaMalloc((void **)&frame_process_calib_data.d_convert, camera_params->width * camera_params->height * 3));
        calib_resources_initialized = true;
    // }


    COpenGLDisplay* openGLDisplay = nullptr;
    if (camera_select->stream_on && (!camera_select->yolo || yolo_worker_for_this_camera == nullptr) && display_buffer != nullptr) {
        std::string display_thread_name = "OpenGLDisplay_Cam_" + camera_params->camera_serial;
        openGLDisplay = new COpenGLDisplay(display_thread_name.c_str(), camera_params, camera_select, display_buffer, indigo_signal_builder);
        openGLDisplay->StartThread();
    }

    GPUVideoEncoder* gpu_encoder = nullptr;
    bool encoder_ready_signal = false;
    if (camera_control->record_video && camera_select->record) {
        std::string encoder_thread_name = "GPUEncoder_Cam_" + camera_params->camera_serial;
        gpu_encoder = new GPUVideoEncoder(encoder_thread_name.c_str(), camera_params, encoder_setup, folder_name, &encoder_ready_signal);
        gpu_encoder->StartThread();
        
        while(!encoder_ready_signal && camera_control->subscribe) {
            usleep(10);
        }
        if (encoder_ready_signal && camera_control->subscribe) {
             std::cout << "Encoder ready for camera: " << camera_params->camera_serial << std::endl;
        } else if (!camera_control->subscribe) {
            std::cout << "Subscription stopped while waiting for encoder on camera: " << camera_params->camera_serial << std::endl;
        } else {
            std::cerr << "Timeout or error waiting for encoder on camera: " << camera_params->camera_serial << std::endl;
        }
    }

    if (camera_control->sync_camera)
    {
        show_ptp_offset(&ptp_state, ecam);
        start_ptp_sync(&ptp_state, ptp_params, camera_params, ecam, 3);
    }

    check_camera_errors(EVT_CameraExecuteCommand(&ecam->camera, "AcquisitionStart"), camera_params->camera_serial.c_str());

    if (camera_control->sync_camera)
    {
        grab_frames_after_countdown(&ptp_state, ecam);
    } else {
        try_start_timer();
    }
    w.Start();

    while (camera_control->subscribe)
    {
        // Corrected call to get_one_frame: pass yolo_worker_for_this_camera and then the calib frame_process
        get_one_frame(&camera_state, camera_select, camera_control, ecam, camera_params, &ptp_state,
                      openGLDisplay, gpu_encoder,
                      yolo_worker_for_this_camera, // YOLO worker instance for this camera stream
                      (calib_resources_initialized ? &frame_process_calib_data : nullptr) // Pass the calib specific struct
                     );
        
        if (ptp_params->network_sync && ptp_params->network_set_stop_ptp) {
            if (ptp_state.ptp_time > ptp_params->ptp_stop_time) {                
                uint64_t ptp_stop_conuter = sync_fetch_and_add(&ptp_params->ptp_stop_counter, 1);
                // printf("Cam %s stopping, counter %lu\n", camera_params->camera_serial.c_str(), ptp_stop_conuter);
                while (ptp_params->ptp_stop_counter != camera_params->num_cameras && camera_control->subscribe)
                {
                    usleep(10);
                }
                if (camera_control->subscribe) {
                    ptp_params->ptp_stop_reached = true;
                    camera_control->subscribe = false;
                }
                break; 
            }
        }
    }

    check_camera_errors(EVT_CameraExecuteCommand(&ecam->camera, "AcquisitionStop"), camera_params->camera_serial.c_str());
    if (!ptp_params->network_sync) {
        try_stop_timer();
    }
    double time_diff = w.Stop();

    if (openGLDisplay) {
        openGLDisplay->StopThread();
        delete openGLDisplay;
        openGLDisplay = nullptr;
    }
    if (gpu_encoder) {
        gpu_encoder->StopThread();
        delete gpu_encoder;
        gpu_encoder = nullptr;
    }

    if (calib_resources_initialized) {
        // Ensure correct GPU context if it matters for these frees, though typically tied to creation context
        ck(cudaSetDevice(camera_params->gpu_id));
        cudaFree(frame_process_calib_data.frame_original.d_orig);
        cudaFree(frame_process_calib_data.debayer.d_debayer);
        cudaFree(frame_process_calib_data.d_convert);
        free(frame_process_calib_data.frame_cpu.frame); // Malloc'd in initialize_cpu_frame
    }

    report_statistics(camera_params, &camera_state, time_diff);
    std::cout << "Acquire frames thread finished for camera: " << camera_params->camera_serial << std::endl;
}