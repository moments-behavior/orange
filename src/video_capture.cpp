#include "video_capture.h"

void PTP_timestamp_checking(PTPState *ptp_state, CameraEmergent *ecam, CameraState *camera_state)
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

void report_statistics(CameraParams *camera_params, CameraState *camera_state, double time_diff)
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

void show_ptp_offset(PTPState *ptp_state, CameraEmergent *ecam)
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

void start_ptp_sync(PTPState *ptp_state, PTPParams *ptp_params, CameraParams *camera_params, CameraEmergent *ecam, unsigned int delay_in_second)
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


void grab_frames_after_countdown(PTPState *ptp_state, CameraEmergent *ecam)
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