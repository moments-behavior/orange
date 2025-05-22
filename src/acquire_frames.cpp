#include "NvEncoder/NvCodecUtils.h"
#include "opengldisplay.h"
#include "gpu_video_encoder.h"
#include "acquire_frames.h"
#include "kernel.cuh"
#include <cuda_runtime_api.h>

static inline void PTP_timestamp_checking(PTPState *ptp_state, CameraEmergent *ecam, CameraState *camera_state)
{

    EVT_CameraExecuteCommand(&ecam->camera, "GevTimestampControlLatch");
    EVT_CameraGetUInt32Param(&ecam->camera, "GevTimestampValueHigh", &ptp_state->ptp_time_high);
    EVT_CameraGetUInt32Param(&ecam->camera, "GevTimestampValueLow", &ptp_state->ptp_time_low);

    ptp_state->ptp_time = (((unsigned long long)(ptp_state->ptp_time_high)) << 32) | ((unsigned long long)(ptp_state->ptp_time_low));
    ptp_state->frame_ts = ecam->frame_recv.timestamp;
    // printf("camera %d, framecount %d, timestamp %f ms \n", camera_params.camera_id, frame_count, frame_ts * 1e-6);

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

static inline void get_one_frame(CameraState *camera_state, 
    CameraEachSelect* camera_select, 
    CameraControl *camera_control, 
    CameraEmergent *ecam, 
    CameraParams *camera_params, 
    PTPState *ptp_state, 
    COpenGLDisplay* openGLDisplay, 
    GPUVideoEncoder* gpu_encoder, 
    FrameProcess* frame_process)
{
    camera_state->camera_return = EVT_CameraGetFrame(&ecam->camera, &ecam->frame_recv, EVT_INFINITE);
    
    // get the system clock
    struct timespec ts_rt1;
    clock_gettime(CLOCK_REALTIME, &ts_rt1);
    uint64_t real_time = (ts_rt1.tv_sec * 1000000000LL) + ts_rt1.tv_nsec;

    if (camera_control->sync_camera)
    {
        PTP_timestamp_checking(ptp_state, ecam, camera_state);
    }

    if (!camera_state->camera_return)
    {
        // Counting dropped frames through frame_id as redundant check.
        if (((ecam->frame_recv.frame_id) != camera_state->id_prev + 1) && (camera_state->frame_count != 0))
            camera_state->dropped_frames++;
        else
        {
            camera_state->frames_recd++;
        }

        camera_state->frame_count++;

        // In GVSP there is no id 0 so when 16 bit id counter in camera is max then the next id is 1 so set prev id to 0 for math above.
        if (ecam->frame_recv.frame_id == 65535)
            camera_state->id_prev = 0;
        else
            camera_state->id_prev = ecam->frame_recv.frame_id;

        // push the image data to encode, or display
        if (camera_control->record_video && camera_select->record) {
            gpu_encoder->PushToDisplay(ecam->frame_recv.imagePtr, 
                ecam->frame_recv.bufferSize, 
                ecam->frame_recv.size_x, 
                ecam->frame_recv.size_y, 
                ecam->frame_recv.pixel_type, 
                ecam->frame_recv.timestamp,
                camera_state->frame_count,
                real_time
                );
        }
        
        if (camera_select->stream_on) {
            openGLDisplay->PushToDisplay(ecam->frame_recv.imagePtr, 
                ecam->frame_recv.bufferSize, 
                ecam->frame_recv.size_x, 
                ecam->frame_recv.size_y, 
                ecam->frame_recv.pixel_type, 
                ecam->frame_recv.timestamp,
                camera_state->frame_count);
        }

        // temp changes to take images for calibration
        if (camera_select->frame_save_state==State_Write_New_Frame) {

            ck(cudaMemcpy2D(frame_process->frame_original.d_orig, camera_params->width, ecam->frame_recv.imagePtr, camera_params->width, camera_params->width, camera_params->height, cudaMemcpyHostToDevice));
            if (camera_params->color){
                debayer_frame_gpu(camera_params, &frame_process->frame_original, &frame_process->debayer);
            } else {
                duplicate_channel_gpu(camera_params, &frame_process->frame_original, &frame_process->debayer);
            }      
            rgba2bgr_convert(frame_process->d_convert, frame_process->debayer.d_debayer, camera_params->width, camera_params->height, 0);                
            cudaMemcpy2D(frame_process->frame_cpu.frame, camera_params->width*3, frame_process->d_convert, camera_params->width*3, camera_params->width*3, camera_params->height, cudaMemcpyDeviceToHost);
            cv::Mat view = cv::Mat(camera_params->width * camera_params->height * 3, 1, CV_8U, frame_process->frame_cpu.frame).reshape(3, camera_params->height);
                
            std::string image_name = camera_select->picture_save_folder + "/" + camera_params->camera_serial + "_" + camera_select->frame_save_name + "." + camera_select->frame_save_format;
            std::cout << image_name << std::endl;
            cv::imwrite(image_name, view);
            camera_select->pictures_counter++;
            camera_select->frame_save_state = State_Frame_Idle;
        }          


        camera_state->camera_return = EVT_CameraQueueFrame(&ecam->camera, &ecam->frame_recv); // Re-queue.
        if (camera_state->camera_return)
            std::cout << "EVT_CameraQueueFrame Error!" << std::endl;

        if (camera_state->frame_count % 500 == 99)
        {
            printf("\n");
            fflush(stdout);
        }

        if (camera_state->frame_count % 1000 == 99)
        {
            // printf(".");
            // fflush(stdout);
            std::cout << camera_params->camera_name << std::endl;
        }
        // if (camera_state->frame_count % 20000 == 9999)
            // printf("\n");
    }
    else
    {
        camera_state->dropped_frames++;
        std::cout << "EVT_CameraGetFrame Error, " << camera_state->camera_return << ", camera serial, " << camera_params->camera_serial << std::endl;
    }
}



void acquire_frames(CameraEmergent *ecam, CameraParams *camera_params, CameraEachSelect* camera_select, CameraControl *camera_control, unsigned char *display_buffer, std::string encoder_setup, std::string folder_name, PTPParams *ptp_params, INDIGOSignalBuilder* indigo_signal_builder)
{
    CameraState camera_state;
    PTPState ptp_state;
    StopWatch w;

    FrameProcess frame_process;
    ck(cudaSetDevice(camera_params->gpu_id));
    // innitialization
    initalize_gpu_frame(&frame_process.frame_original, camera_params);
    initialize_gpu_debayer(&frame_process.debayer, camera_params);
    initialize_cpu_frame(&frame_process.frame_cpu, camera_params);
    ck(cudaMalloc((void **)&frame_process.d_convert, camera_params->width * camera_params->height * 3));

    COpenGLDisplay* openGLDisplay;
    if (camera_select->stream_on) {
        openGLDisplay = new COpenGLDisplay("", camera_params, camera_select, display_buffer, indigo_signal_builder);
        openGLDisplay->StartThread();
    }

    GPUVideoEncoder* gpu_encoder;
    bool encoder_ready_signal = false;
    if (camera_control->record_video && camera_select->record) {
        gpu_encoder = new GPUVideoEncoder("", camera_params, encoder_setup, folder_name, &encoder_ready_signal);
        gpu_encoder->StartThread();
        
        // wait till encoder is ready
        while(!encoder_ready_signal) {
            usleep(10);
        }
        std::cout << "encoder ready\n" << std::endl;
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
    }
    ptp_params->ptp_start_reached = true;
    w.Start();

    // int OFFSET_X_VAL = 2848;
    // EVT_CameraSetUInt32Param(&ecam->camera, "OffsetX", OFFSET_X_VAL);
    // int offset = 0;
    // int phase = 1;
    while (camera_control->subscribe)
    {
        // int OFFSET_Y_VAL = 1300 + offset * 4;
        // EVT_CameraSetUInt32Param(&ecam->camera, "OffsetY", OFFSET_Y_VAL);
        get_one_frame(&camera_state, camera_select, camera_control, ecam, camera_params, &ptp_state, openGLDisplay, gpu_encoder, &frame_process);
        if (ptp_params->network_sync && ptp_params->network_set_stop_ptp) {
            if (ptp_state.ptp_time > ptp_params->ptp_stop_time) {                
                uint64_t ptp_stop_conuter = sync_fetch_and_add(&ptp_params->ptp_stop_counter, 1);
                printf("%lu\n", ptp_stop_conuter);
                while (ptp_params->ptp_stop_counter != camera_params->num_cameras)
                {
                    // printf(".");
                    // fflush(stdout);
                    usleep(10);
                }
                ptp_params->ptp_stop_reached = true;
                camera_control->subscribe = false;
                break;
            }
        }
        // if (offset == 200) {
        //     phase = -1;
        // }
        // if (offset == 0) {
        //     phase = 1;
        // }
        // if (phase == -1) {
        //     offset--;
        // } else { offset++; }
    }

    check_camera_errors(EVT_CameraExecuteCommand(&ecam->camera, "AcquisitionStop"), camera_params->camera_serial.c_str());
    try_stop_timer();
    double time_diff = w.Stop();

    if (camera_select->stream_on) {
        openGLDisplay->StopThread();
    }
    if (camera_control->record_video && camera_select->record) {
        gpu_encoder->StopThread();
    }
    report_statistics(camera_params, &camera_state, time_diff);
    cudaFree(frame_process.frame_original.d_orig);
    cudaFree(frame_process.debayer.d_debayer);
    free(frame_process.frame_cpu.frame);
}