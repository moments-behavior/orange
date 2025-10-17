#include "video_capture.h"
#include "FrameSaver.h"
#include "NvEncoder/NvCodecUtils.h"
#include "global.h"
#include "gpu_video_encoder.h"
#include "utils.h"
#ifndef HEADLESS
#include "FrameDetector.h"
#include "opengldisplay.h"
#endif

void load_camera_json_config_files(std::string file_name,
                                   CameraParams *camera_params,
                                   CameraEachSelect *camera_select,
                                   int camera_id, int num_cameras) {

    std::ifstream f(file_name);
    json camera_config = json::parse(f);

    camera_params->camera_id = camera_id;
    camera_params->num_cameras = num_cameras;
    camera_params->need_reorder = false;

    camera_params->camera_name = camera_config["name"];
    camera_params->width = camera_config["width"];
    camera_params->height = camera_config["height"];
    camera_params->frame_rate = camera_config["frame_rate"];
    camera_params->gain = camera_config["gain"];
    camera_params->exposure = camera_config["exposure"];
    camera_params->pixel_format = camera_config["pixel_format"];
    camera_params->color_temp = camera_config["color_temp"];
    camera_params->gpu_id = camera_config["gpu_id"];
    camera_params->gpu_direct = camera_config["gpu_direct"];
    camera_params->color = camera_config["color"];
    camera_params->focus = camera_config["focus"];
    camera_params->iris = camera_config["iris"];
    if (camera_config.contains("gop")) {
        camera_params->gop = camera_config["gop"];
    } else {
        camera_params->gop = 1;
    }
    if (camera_config.contains("yolo")) {
        camera_select->yolo_model = camera_config["yolo"];
    }
    if (camera_config.contains("offsetx")) {
        camera_params->offsetx = camera_config["offsetx"];
    }
    if (camera_config.contains("offsety")) {
        camera_params->offsety = camera_config["offsety"];
    }
}

bool set_camera_params(CameraParams *camera_params,
                       CameraEachSelect *camera_select,
                       GigEVisionDeviceInfo *device_info,
                       std::vector<std::string> &camera_config_files,
                       int camera_idx, int num_cameras) {
    // first checkt to see if it is in the config files
    camera_params->camera_serial.append(device_info->serialNumber);
    camera_params->camera_name = camera_params->camera_serial;

    std::string sub_str = camera_params->camera_serial + ".json";
    auto it =
        std::find_if(camera_config_files.begin(), camera_config_files.end(),
                     [&](const std::string &str) {
                         return str.find(sub_str) != std::string::npos;
                     });

    if (it == camera_config_files.end()) {
        if (strcmp(device_info->modelName, "HB-65000GM") == 0) {
            int gpu_id = 0;
            init_65MP_camera_params_mono(camera_params, camera_idx, num_cameras,
                                         2000, 1000, gpu_id, 400); // 458
        } else if (strcmp(device_info->modelName, "HB-7000SC") == 0) {
            int gpu_id = 0;
            init_7MP_camera_params_color(camera_params, camera_idx, num_cameras,
                                         1500, 3000, gpu_id, 30); // 2000, 3000
        } else if (strcmp(device_info->modelName, "HB-65000GC") == 0) {
            int gpu_id = 0;
            init_65MP_camera_params_color(camera_params, camera_idx,
                                          num_cameras, 2000, 28000, gpu_id, 10);
        } else if (strcmp(device_info->modelName, "HB-7000SM") == 0) {
            int gpu_id = 0;
            init_7MP_camera_params_mono(camera_params, camera_idx, num_cameras,
                                        1000, 3000, gpu_id, 30); // 2000, 3000
        } else {
            printf("Use default parameters. \n");
            return false;
        }
    } else {
        auto config_idx = std::distance(camera_config_files.begin(), it);
        std::cout << "Load camera json file: "
                  << camera_config_files[config_idx] << std::endl;
        load_camera_json_config_files(camera_config_files[config_idx],
                                      camera_params, camera_select, camera_idx,
                                      num_cameras);
    }
    return true;
}

void report_statistics(CameraParams *camera_params, CameraState *camera_state,
                       double time_diff) {
    std::string print_out;
    print_out += "\n" + camera_params->camera_serial;
    print_out += ", Frame count: " + std::to_string(camera_state->frame_count);
    print_out +=
        ", Frame received: " + std::to_string(camera_state->frames_recd);
    print_out +=
        ", Dropped Frames: " + std::to_string(camera_state->dropped_frames);
    float calc_frame_rate = camera_state->frames_recd / time_diff;
    print_out += ", Calculated Frame Rate: " + std::to_string(calc_frame_rate);
    std::cout << print_out << std::endl;
}

void show_ptp_offset(PTPState *ptp_state, CameraEmergent *ecam) {
    // Show raw offsets.
    for (unsigned int i = 0; i < 5;) {
        EVT_CameraGetInt32Param(&ecam->camera, "PtpOffset",
                                &ptp_state->ptp_offset);
        if (ptp_state->ptp_offset != ptp_state->ptp_offset_prev) {
            ptp_state->ptp_offset_sum += ptp_state->ptp_offset;
            i++;
            // printf("Offset %d: %d\n", i, ptp_offset);
        }
        ptp_state->ptp_offset_prev = ptp_state->ptp_offset;
    }
    printf("Offset Average: %d\n", ptp_state->ptp_offset_sum / 5);
}

void start_ptp_sync(PTPState *ptp_state, PTPParams *ptp_params,
                    CameraParams *camera_params, CameraEmergent *ecam,
                    unsigned int delay_in_second) {
    if (ptp_params->network_sync) {
        uint64_t ptp_counter = sync_fetch_and_add(&ptp_params->ptp_counter, 1);
        printf("%lu\n", ptp_counter);
        std::cout << ptp_params->ptp_global_time << std::endl;
        while (!ptp_params->network_set_start_ptp) {
            usleep(10); // sleep 1ms
        }
        ptp_state->ptp_time = get_current_PTP_time(&ecam->camera);
    } else {
        if (ptp_params->ptp_counter == camera_params->num_cameras - 1) {
            ptp_state->ptp_time = get_current_PTP_time(&ecam->camera);
            ptp_params->ptp_global_time =
                ((unsigned long long)delay_in_second) * 1000000000 +
                ptp_state->ptp_time;
        }
        uint64_t ptp_counter = sync_fetch_and_add(&ptp_params->ptp_counter, 1);
        printf("%lu\n", ptp_counter);
        while (ptp_params->ptp_counter != camera_params->num_cameras) {
            // printf(".");
            // fflush(stdout);
            usleep(10);
        }
    }

    unsigned long long ptp_time_plus_delta_to_start =
        ptp_params->ptp_global_time;
    ptp_state->ptp_time_plus_delta_to_start_low =
        (unsigned int)(ptp_time_plus_delta_to_start & 0xFFFFFFFF);
    ptp_state->ptp_time_plus_delta_to_start_high =
        (unsigned int)(ptp_time_plus_delta_to_start >> 32);
    EVT_CameraSetUInt32Param(&ecam->camera, "PtpAcquisitionGateTimeHigh",
                             ptp_state->ptp_time_plus_delta_to_start_high);
    EVT_CameraSetUInt32Param(&ecam->camera, "PtpAcquisitionGateTimeLow",
                             ptp_state->ptp_time_plus_delta_to_start_low);
    ptp_state->ptp_time_plus_delta_to_start_uint = ptp_time_plus_delta_to_start;
    ptp_state->ptp_time_plus_delta_to_start = ptp_params->ptp_global_time;
    printf("PTP Gate time(ns): %llu\n", ptp_time_plus_delta_to_start);
}

void grab_frames_after_countdown(PTPState *ptp_state, CameraEmergent *ecam) {
    printf("Grabbing Frames after countdown...\n");
    ptp_state->ptp_time_countdown = 0;
    // Countdown code
    do {
        EVT_CameraExecuteCommand(&ecam->camera, "GevTimestampControlLatch");
        EVT_CameraGetUInt32Param(&ecam->camera, "GevTimestampValueHigh",
                                 &ptp_state->ptp_time_high);
        EVT_CameraGetUInt32Param(&ecam->camera, "GevTimestampValueLow",
                                 &ptp_state->ptp_time_low);
        ptp_state->ptp_time =
            (((unsigned long long)(ptp_state->ptp_time_high)) << 32) |
            ((unsigned long long)(ptp_state->ptp_time_low));

        if (ptp_state->ptp_time > ptp_state->ptp_time_countdown) {
            printf("%llu\n", (ptp_state->ptp_time_plus_delta_to_start -
                              ptp_state->ptp_time) /
                                 1000000000);
            ptp_state->ptp_time_countdown =
                ptp_state->ptp_time + 1000000000; // 1s
        }

    } while (ptp_state->ptp_time <= ptp_state->ptp_time_plus_delta_to_start);
}

inline void PTP_timestamp_checking(PTPState *ptp_state, CameraEmergent *ecam,
                                   CameraState *camera_state) {

    EVT_CameraExecuteCommand(&ecam->camera, "GevTimestampControlLatch");
    EVT_CameraGetUInt32Param(&ecam->camera, "GevTimestampValueHigh",
                             &ptp_state->ptp_time_high);
    EVT_CameraGetUInt32Param(&ecam->camera, "GevTimestampValueLow",
                             &ptp_state->ptp_time_low);

    ptp_state->ptp_time =
        (((unsigned long long)(ptp_state->ptp_time_high)) << 32) |
        ((unsigned long long)(ptp_state->ptp_time_low));
    ptp_state->frame_ts = ecam->frame_recv.timestamp;
    // printf("camera %d, framecount %d, timestamp %f ms \n",
    // camera_params.camera_id, frame_count, frame_ts * 1e-6);

    if (camera_state->frame_count != 0) {
        ptp_state->ptp_time_delta =
            ptp_state->ptp_time - ptp_state->ptp_time_prev;
        ptp_state->ptp_time_delta_sum += ptp_state->ptp_time_delta;

        ptp_state->frame_ts_delta =
            ptp_state->frame_ts - ptp_state->frame_ts_prev;
        ptp_state->frame_ts_delta_sum += ptp_state->frame_ts_delta;
    }

    ptp_state->ptp_time_prev = ptp_state->ptp_time;
    ptp_state->frame_ts_prev = ptp_state->frame_ts;
}

inline void get_one_frame(CameraState *camera_state,
                          CameraEachSelect *camera_select,
                          CameraControl *camera_control, CameraEmergent *ecam,
                          CameraParams *camera_params, PTPState *ptp_state,
                          void *openGLDisplay, GPUVideoEncoder *gpu_encoder,
                          FrameSaver *frame_saver, void *frame_detector) {
    if (camera_control->trigger_mode) {
        std::cout << "trigger" << std::endl;
        check_camera_errors(
            EVT_CameraExecuteCommand(&ecam->camera, "TriggerSoftware"),
            camera_params->camera_serial.c_str());
    }

    camera_state->camera_return =
        EVT_CameraGetFrame(&ecam->camera, &ecam->frame_recv, EVT_INFINITE);

    // get the system clock
    struct timespec ts_rt1;
    clock_gettime(CLOCK_REALTIME, &ts_rt1);
    uint64_t real_time = (ts_rt1.tv_sec * 1000000000LL) + ts_rt1.tv_nsec;

    if (camera_control->sync_camera) {
        PTP_timestamp_checking(ptp_state, ecam, camera_state);
    }

    if (!camera_state->camera_return) {
        // Counting dropped frames through frame_id as redundant check.
        if (((ecam->frame_recv.frame_id) != camera_state->id_prev + 1) &&
            (camera_state->frame_count != 0)) {
            camera_state->dropped_frames++;
            camera_select->dropped_frames++;
        } else {
            camera_state->frames_recd++;
            camera_select->capture_fps_estimator.update();
        }

        // In GVSP there is no id 0 so when 16 bit id counter in camera is max
        // then the next id is 1 so set prev id to 0 for math above.
        if (ecam->frame_recv.frame_id == 65535)
            camera_state->id_prev = 0;
        else
            camera_state->id_prev = ecam->frame_recv.frame_id;

        // push the image data to encode, or display
        if (camera_control->record_video && camera_select->record) {
            gpu_encoder->PushToDisplay(
                ecam->frame_recv.imagePtr, ecam->frame_recv.bufferSize,
                ecam->frame_recv.size_x, ecam->frame_recv.size_y,
                ecam->frame_recv.pixel_type, ecam->frame_recv.timestamp,
                camera_state->frame_count, real_time);
        }

#ifndef HEADLESS
        COpenGLDisplay *display = static_cast<COpenGLDisplay *>(openGLDisplay);
        if (display) {
            display->PushToDisplay(
                ecam->frame_recv.imagePtr, ecam->frame_recv.bufferSize,
                ecam->frame_recv.size_x, ecam->frame_recv.size_y,
                ecam->frame_recv.pixel_type, ecam->frame_recv.timestamp,
                camera_state->frame_count);
        }
        FrameDetector *detector = static_cast<FrameDetector *>(frame_detector);
        if (detector && camera_select->sigs->frame_detect_state.load() ==
                            State_Copy_New_Frame) {
            detector->notify_frame_ready(ecam->frame_recv.imagePtr, 0);
        }
#endif

        if (camera_select->sigs->frame_save_state.load() ==
            State_Copy_New_Frame) {
            frame_saver->notify_frame_ready(ecam->frame_recv.imagePtr);
        }

        camera_state->frame_count++;

        camera_state->camera_return =
            EVT_CameraQueueFrame(&ecam->camera, &ecam->frame_recv); // Re-queue.
        if (camera_state->camera_return) {
            std::cout << "EVT_CameraQueueFrame Error!" << std::endl;
        }

        /* if (camera_state->frame_count % 500 == 99) {
            printf("\n");
            fflush(stdout);
        }

        if (camera_state->frame_count % 1000 == 99) {
            // printf(".");
            // fflush(stdout);
            std::cout << camera_params->camera_name << std::endl;
        } */
        // if (camera_state->frame_count % 20000 == 9999)
        // printf("\n");
    } else {
        camera_state->dropped_frames++;
        std::cout << "EVT_CameraGetFrame Error, " << camera_state->camera_return
                  << ", camera serial, " << camera_params->camera_serial
                  << std::endl;
    }
}

void acquire_frames(CameraEmergent *ecam, CameraParams *camera_params,
                    CameraEachSelect *camera_select,
                    CameraControl *camera_control,
                    unsigned char *display_buffer, std::string encoder_setup,
                    std::string folder_name, PTPParams *ptp_params,
                    AppContext *ctx) {
    CHECK(cudaSetDevice(camera_params->gpu_id));
    CameraState camera_state;
    PTPState ptp_state;
    StopWatch w;

#ifndef HEADLESS
    FrameDetector *frame_detector = nullptr;
    if (camera_select->detect_mode == Detect3D_Standoff ||
        camera_select->detect_mode == Detect2D_Standoff) {
        frame_detector = new FrameDetector(camera_params, camera_select, ctx);
        frame_detector->start();

        while (detector_counter.load() !=
               camera_select->total_standoff_detector) {
            // printf(".");
            // fflush(stdout);
            usleep(10);
        }
        camera_select->sigs->frame_detect_state.store(State_Copy_New_Frame);
    }

    COpenGLDisplay *openGLDisplay = nullptr;
    if (camera_select->stream_on) {
        openGLDisplay = new COpenGLDisplay("gl", camera_params, camera_select,
                                           display_buffer, ctx);
        openGLDisplay->StartThread();
    }
#endif

    FrameSaver frame_saver(camera_params, camera_select);
    frame_saver.start();

    GPUVideoEncoder *gpu_encoder = nullptr;
    bool encoder_ready_signal = false;
    if (camera_control->record_video && camera_select->record) {
        gpu_encoder = new GPUVideoEncoder("encoder", camera_params,
                                          camera_select, encoder_setup,
                                          folder_name, &encoder_ready_signal);
        gpu_encoder->StartThread();

        // wait till encoder is ready
        while (!encoder_ready_signal) {
            usleep(10);
        }
        std::cout << "encoder ready\n" << std::endl;
    }

    if (camera_control->sync_camera) {
        show_ptp_offset(&ptp_state, ecam);
        start_ptp_sync(&ptp_state, ptp_params, camera_params, ecam, 3);
    }

    check_camera_errors(
        EVT_CameraExecuteCommand(&ecam->camera, "AcquisitionStart"),
        camera_params->camera_serial.c_str());

    if (camera_control->sync_camera) {
        grab_frames_after_countdown(&ptp_state, ecam);
        // Countdown done.
        try_start_timer();
    }
    ptp_params->ptp_start_reached = true;
    w.Start();

    // int OFFSET_X_VAL = 2848;
    // EVT_CameraSetUInt32Param(&ecam->camera, "OffsetX", OFFSET_X_VAL);
    // int offset = 0;
    // int phase = 1;
    while (camera_control->subscribe) {
        // int OFFSET_Y_VAL = 1300 + offset * 4;
        // EVT_CameraSetUInt32Param(&ecam->camera, "OffsetY", OFFSET_Y_VAL);
#ifndef HEADLESS
        get_one_frame(&camera_state, camera_select, camera_control, ecam,
                      camera_params, &ptp_state, openGLDisplay, gpu_encoder,
                      &frame_saver, frame_detector);
#else
        get_one_frame(&camera_state, camera_select, camera_control, ecam,
                      camera_params, &ptp_state, nullptr, gpu_encoder,
                      &frame_saver, nullptr);
#endif
        if (ptp_params->network_sync && ptp_params->network_set_stop_ptp) {
            if (ptp_state.ptp_time > ptp_params->ptp_stop_time) {
                uint64_t ptp_stop_conuter =
                    sync_fetch_and_add(&ptp_params->ptp_stop_counter, 1);
                printf("%lu\n", ptp_stop_conuter);
                while (ptp_params->ptp_stop_counter !=
                       camera_params->num_cameras) {
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

    check_camera_errors(
        EVT_CameraExecuteCommand(&ecam->camera, "AcquisitionStop"),
        camera_params->camera_serial.c_str());
    try_stop_timer();
    double time_diff = w.Stop();

#ifndef HEADLESS
    if (camera_select->stream_on) {
        openGLDisplay->StopThread();
        delete openGLDisplay;
    }

    if (camera_select->detect_mode == Detect3D_Standoff ||
        camera_select->detect_mode == Detect2D_Standoff) {
        frame_detector->stop();
        delete frame_detector;
    }
#endif

    frame_saver.stop();

    if (camera_control->record_video && camera_select->record) {
        gpu_encoder->StopThread();
        delete gpu_encoder;
    }
    report_statistics(camera_params, &camera_state, time_diff);
}
