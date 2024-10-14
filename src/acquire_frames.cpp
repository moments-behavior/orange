#include "NvEncoder/NvCodecUtils.h"
#include "opengldisplay.h"
#include "gpu_video_encoder.h"
#include "acquire_frames.h"
#include "realtime_tool.h"
#include "detection3D.h"

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


static inline void get_one_frame(CameraState *camera_state, CameraEachSelect* camera_select, CameraControl *camera_control, CameraEmergent *ecam, CameraParams *camera_params, PTPState *ptp_state, COpenGLDisplay* openGLDisplay, GPUVideoEncoder* gpu_encoder, SyncDetection* sync_detection)
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
        if (camera_control->record_video) {
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

        if (camera_select->sync_detect) {
            if (!sync_detection->frame_unread[camera_select->sync_id]) {
                // copy frame, maybe need multiple, need to make sure it is all copied
                sync_detection->frame_unread[camera_select->sync_id] = true;
            }
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

void acquire_frames(CameraEmergent *ecam, CameraParams *camera_params, CameraEachSelect* camera_select, CameraControl *camera_control, unsigned char *display_buffer, std::string encoder_setup, std::string folder_name, PTPParams *ptp_params, INDIGOSignalBuilder* indigo_signal_builder, DetectionData* detection_data, SyncDetection* sync_detection)
{
    CameraState camera_state;
    PTPState ptp_state;
    
    std::vector<cv::Point2d> points2d;

    COpenGLDisplay* openGLDisplay;
    if (camera_select->stream_on) {
        openGLDisplay = new COpenGLDisplay("", camera_params, camera_select, display_buffer, indigo_signal_builder, detection_data);
        openGLDisplay->StartThread();
    }

    GPUVideoEncoder* gpu_encoder;
    bool encoder_ready_signal = false;
    if (camera_control->record_video) {
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
    StopWatch w;
    w.Start();


    int skip_frame_counter = 0;
    while (camera_control->subscribe)
    {   
        
        get_one_frame(&camera_state, camera_select, camera_control, ecam, camera_params, &ptp_state, openGLDisplay, gpu_encoder, sync_detection);
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
    }

    check_camera_errors(EVT_CameraExecuteCommand(&ecam->camera, "AcquisitionStop"), camera_params->camera_serial.c_str());
    double time_diff = w.Stop();

    if (camera_select->stream_on) {
        openGLDisplay->StopThread();
    }
    if (camera_control->record_video) {
        gpu_encoder->StopThread();
    }
    report_statistics(camera_params, &camera_state, time_diff);
}