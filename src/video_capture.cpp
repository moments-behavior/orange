#include "video_capture.h"
#include "NvEncoder/NvCodecUtils.h"
#include "opengldisplay.h"
#include "gpu_video_encoder.h"

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

static inline void get_one_frame(CameraState *camera_state, CameraEachSelect* camera_select, CameraControl *camera_control, CameraEmergent *ecam, CameraParams *camera_params, PTPState *ptp_state, COpenGLDisplay* openGLDisplay, GPUVideoEncoder* gpu_encoder)
{
    camera_state->camera_return = EVT_CameraGetFrame(&ecam->camera, &ecam->frame_recv, EVT_INFINITE);
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
                camera_state->frame_count);
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

        camera_state->camera_return = EVT_CameraQueueFrame(&ecam->camera, &ecam->frame_recv); // Re-queue.
        if (camera_state->camera_return)
            std::cout << "EVT_CameraQueueFrame Error!" << std::endl;

        if (camera_state->frame_count % 500 == 99)
        {
            printf(".");
            fflush(stdout);
        }
        if (camera_state->frame_count % 20000 == 9999)
            printf("\n");
    }
    else
    {
        camera_state->dropped_frames++;
        std::cout << "EVT_CameraGetFrame Error" << camera_state->camera_return << std::endl;
    }
}

static inline void report_statistics(CameraParams *camera_params, CameraState *camera_state, double time_diff)
{
    std::string print_out;
    print_out += "\n" + camera_params->camera_serial;
    print_out += ", Frame count: " + std::to_string(camera_state->frame_count);
    print_out += ", Frame received: " + std::to_string(camera_state->frames_recd);
    print_out += ", Dropped Frames: " + std::to_string(camera_state->dropped_frames);
    float calc_frame_rate = camera_state->frames_recd / time_diff;
    print_out += ", Calculated Frame Rate: " + std::to_string(calc_frame_rate);
    std::cout << print_out << std::endl;
}

static inline void show_ptp_offset(PTPState *ptp_state, CameraEmergent *ecam)
{
    // Show raw offsets.
    for (unsigned int i = 0; i < 5;)
    {
        EVT_CameraGetInt32Param(&ecam->camera, "PtpOffset", &ptp_state->ptp_offset);
        if (ptp_state->ptp_offset != ptp_state->ptp_offset_prev)
        {
            ptp_state->ptp_offset_sum += ptp_state->ptp_offset;
            i++;
            // printf("Offset %d: %d\n", i, ptp_offset);
        }
        ptp_state->ptp_offset_prev = ptp_state->ptp_offset;
    }
    printf("Offset Average: %d\n", ptp_state->ptp_offset_sum / 5);
}

static inline void start_ptp_sync(PTPState *ptp_state, PTPParams *ptp_params, CameraParams *camera_params, CameraEmergent *ecam, unsigned int delay_in_second)
{


    if (ptp_params->network_sync) {
        std::cout << ptp_params->ptp_global_time << std::endl;
        while(!ptp_params->network_set_start_ptp) {
            usleep(10); // sleep 1ms
        }
        ptp_state->ptp_time = get_current_PTP_time(&ecam->camera);
    } else {
        if (ptp_params->ptp_counter == camera_params->num_cameras - 1)
        {
            ptp_state->ptp_time = get_current_PTP_time(&ecam->camera);
            ptp_params->ptp_global_time = ((unsigned long long)delay_in_second) * 1000000000 + ptp_state->ptp_time;
        }
    }

    uint64_t ptp_counter = sync_fetch_and_add(&ptp_params->ptp_counter, 1);
    printf("%lu\n", ptp_counter);
    while (ptp_params->ptp_counter != camera_params->num_cameras)
    {
        // printf(".");
        // fflush(stdout);
        usleep(10);
    }

    unsigned long long ptp_time_plus_delta_to_start = ptp_params->ptp_global_time;
    ptp_state->ptp_time_plus_delta_to_start_low = (unsigned int)(ptp_time_plus_delta_to_start & 0xFFFFFFFF);
    ptp_state->ptp_time_plus_delta_to_start_high = (unsigned int)(ptp_time_plus_delta_to_start >> 32);
    EVT_CameraSetUInt32Param(&ecam->camera, "PtpAcquisitionGateTimeHigh", ptp_state->ptp_time_plus_delta_to_start_high);
    EVT_CameraSetUInt32Param(&ecam->camera, "PtpAcquisitionGateTimeLow", ptp_state->ptp_time_plus_delta_to_start_low);
    ptp_state->ptp_time_plus_delta_to_start_uint = ptp_time_plus_delta_to_start;
    ptp_state->ptp_time_plus_delta_to_start = ptp_params->ptp_global_time;
    printf("PTP Gate time(ns): %llu\n", ptp_time_plus_delta_to_start);
}


static inline void grab_frames_after_countdown(PTPState *ptp_state, CameraEmergent *ecam)
{
    printf("Grabbing Frames after countdown...\n");
    ptp_state->ptp_time_countdown = 0;
    // Countdown code
    do
    {
        EVT_CameraExecuteCommand(&ecam->camera, "GevTimestampControlLatch");
        EVT_CameraGetUInt32Param(&ecam->camera, "GevTimestampValueHigh", &ptp_state->ptp_time_high);
        EVT_CameraGetUInt32Param(&ecam->camera, "GevTimestampValueLow", &ptp_state->ptp_time_low);
        ptp_state->ptp_time = (((unsigned long long)(ptp_state->ptp_time_high)) << 32) | ((unsigned long long)(ptp_state->ptp_time_low));

        if (ptp_state->ptp_time > ptp_state->ptp_time_countdown)
        {
            printf("%llu\n", (ptp_state->ptp_time_plus_delta_to_start - ptp_state->ptp_time) / 1000000000);
            ptp_state->ptp_time_countdown = ptp_state->ptp_time + 1000000000; // 1s
        }

    } while (ptp_state->ptp_time <= ptp_state->ptp_time_plus_delta_to_start);
    // Countdown done.
    printf("\n");
}

void aquire_frames(CameraEmergent *ecam, CameraParams *camera_params, CameraEachSelect* camera_select, CameraControl *camera_control, unsigned char *display_buffer, std::string encoder_setup, std::string folder_name, PTPParams *ptp_params, CBOTSignalBuilder* cbot_signal_builder)
{

    CameraState camera_state;
    PTPState ptp_state;
    
    COpenGLDisplay* openGLDisplay;
    if (camera_select->stream_on) {
        openGLDisplay = new COpenGLDisplay("", camera_params, camera_select, display_buffer, cbot_signal_builder);
        openGLDisplay->StartThread();
    }

    GPUVideoEncoder* gpu_encoder;
    
    if (camera_control->record_video) {
        gpu_encoder = new GPUVideoEncoder("", camera_params, encoder_setup, folder_name);
        gpu_encoder->StartThread();
    }

    // if (camera_control->sync_camera)
    // {
    //     show_ptp_offset(&ptp_state, ecam);
    //     start_ptp_sync(&ptp_state, ptp_params, camera_params, ecam, 3);
    // }

    check_camera_errors(EVT_CameraExecuteCommand(&ecam->camera, "AcquisitionStart"));

    if (camera_control->sync_camera)
    {
        grab_frames_after_countdown(&ptp_state, ecam);
    }

    StopWatch w;
    w.Start();

    // int OFFSET_X_VAL = 2848;
    // EVT_CameraSetUInt32Param(&ecam->camera, "OffsetX", OFFSET_X_VAL);
    // int offset = 0;
    // int phase = 1;
    while (camera_control->subscribe)
    {
        // int OFFSET_Y_VAL = 1300 + offset * 4;
        // EVT_CameraSetUInt32Param(&ecam->camera, "OffsetY", OFFSET_Y_VAL);
        get_one_frame(&camera_state, camera_select, camera_control, ecam, camera_params, &ptp_state, openGLDisplay, gpu_encoder);
        // if (ptp_params->network_sync && ptp_params->network_set_stop_ptp) {
        //     if (ptp_state.ptp_time > ptp_params->ptp_stop_time) {                
        //         uint64_t ptp_stop_conuter = sync_fetch_and_add(&ptp_params->ptp_stop_counter, 1);
        //         printf("%lu\n", ptp_stop_conuter);
        //         while (ptp_params->ptp_stop_counter != camera_params->num_cameras)
        //         {
        //             // printf(".");
        //             // fflush(stdout);
        //             usleep(10);
        //         }
        //         ptp_params->ptp_stop_reached = true;
        //         camera_control->subscribe = false;
        //         break;
        //     }
        // }
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

    check_camera_errors(EVT_CameraExecuteCommand(&ecam->camera, "AcquisitionStop"));
    double time_diff = w.Stop();

    if (camera_select->stream_on) {
        openGLDisplay->StopThread();
    }
    if (camera_control->record_video) {
        gpu_encoder->StopThread();
    }
    report_statistics(camera_params, &camera_state, time_diff);
}