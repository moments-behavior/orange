#include "video_capture.h"
#include "FrameSaver.h"
#include "NvEncoder/NvCodecUtils.h"
#include "global.h"
#include "gpu_video_encoder.h"
#include "mjpeg_stream.h"
#include "utils.h"
#include <opencv2/opencv.hpp>
#ifndef HEADLESS
#include "FrameDetector.h"
#include "opengldisplay.h"
#endif

// Laplacian variance below this is considered blurry (tune per camera)
static constexpr double BLUR_LAPLACIAN_THRESHOLD = 50.0;
static constexpr double BRIGHTNESS_TARGET = 52.0;
static constexpr double BRIGHTNESS_LOW    = 47.0;

static double laplacian_variance_of_crop(const unsigned char *image_ptr,
                                          unsigned int width, unsigned int height,
                                          int cx, int cy, bool gpu_direct) {
    constexpr int CROP_SIZE = 256;
    cx = std::max(0, std::min(cx, (int)width  - CROP_SIZE));
    cy = std::max(0, std::min(cy, (int)height - CROP_SIZE));
    cv::Mat crop(CROP_SIZE, CROP_SIZE, CV_8UC1);
    if (gpu_direct) {
        const unsigned char *src_row = image_ptr + cy * width + cx;
        cudaError_t err = cudaMemcpy2D(crop.data, CROP_SIZE, src_row, width,
                                       CROP_SIZE, CROP_SIZE,
                                       cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return -1.0;
    } else {
        cv::Mat full(height, width, CV_8UC1,
                     const_cast<unsigned char *>(image_ptr));
        full(cv::Rect(cx, cy, CROP_SIZE, CROP_SIZE)).copyTo(crop);
    }
    cv::Mat small;
    cv::resize(crop, small, cv::Size(128, 128));
    cv::Mat lap;
    cv::Laplacian(small, lap, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev);
    return stddev[0] * stddev[0];
}

static double compute_laplacian_variance(const unsigned char *image_ptr,
                                         unsigned int width,
                                         unsigned int height,
                                         bool gpu_direct) {
    // Sample 5 regions and return the max — avoids flat areas (e.g. table top).
    int w = (int)width, h = (int)height;
    int half = 256 / 2;
    int cx[5] = { w/2-half, w/4-half, 3*w/4-half, w/4-half, 3*w/4-half };
    int cy[5] = { h/2-half, h/4-half, h/4-half,   3*h/4-half, 3*h/4-half };
    double best = -1.0;
    for (int i = 0; i < 5; i++) {
        double v = laplacian_variance_of_crop(image_ptr, width, height,
                                              cx[i], cy[i], gpu_direct);
        if (v > best) best = v;
    }
    return best;
}

static double compute_mean_brightness(const unsigned char *image_ptr,
                                      unsigned int width, unsigned int height,
                                      bool gpu_direct) {
    constexpr int CROP_SIZE = 512;
    int cx = (int)width / 2 - CROP_SIZE / 2;
    int cy = (int)height / 2 - CROP_SIZE / 2;
    if (cx < 0) cx = 0;
    if (cy < 0) cy = 0;
    int crop_w = std::min(CROP_SIZE, (int)width - cx);
    int crop_h = std::min(CROP_SIZE, (int)height - cy);
    cv::Mat crop(crop_h, crop_w, CV_8UC1);
    if (gpu_direct) {
        const unsigned char *src_row = image_ptr + cy * width + cx;
        cudaError_t err = cudaMemcpy2D(crop.data, crop_w, src_row, width,
                                       crop_w, crop_h, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) return -1.0;
    } else {
        cv::Mat full(height, width, CV_8UC1,
                     const_cast<unsigned char *>(image_ptr));
        full(cv::Rect(cx, cy, crop_w, crop_h)).copyTo(crop);
    }
    cv::Scalar mean = cv::mean(crop);
    return mean[0];
}

// Build a JPEG preview using a center crop + cudaMemcpy2D (safe for
// GPU buffers that may have padding/pitch != width).
static std::vector<uint8_t> make_preview_jpeg(
    const unsigned char *image_ptr, unsigned int width, unsigned int height,
    bool gpu_direct) {
    std::vector<uint8_t> out;
    // Use a center crop that fits in 1024x1024 max
    int crop_w = std::min((int)width, 1024);
    int crop_h = std::min((int)height, 1024);
    int cx = ((int)width - crop_w) / 2;
    int cy = ((int)height - crop_h) / 2;

    cv::Mat crop(crop_h, crop_w, CV_8UC1);
    if (gpu_direct) {
        const unsigned char *src_row = image_ptr + cy * width + cx;
        cudaError_t err = cudaMemcpy2D(crop.data, crop_w, src_row, width,
                                       crop_w, crop_h,
                                       cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            printf("make_preview_jpeg: cudaMemcpy2D failed (%d)\n", (int)err);
            fflush(stdout);
            return out;
        }
    } else {
        cv::Mat full(height, width, CV_8UC1,
                     const_cast<unsigned char *>(image_ptr));
        full(cv::Rect(cx, cy, crop_w, crop_h)).copyTo(crop);
    }
    cv::imencode(".jpg", crop, out, {cv::IMWRITE_JPEG_QUALITY, 75});
    return out;
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
    print_out +=
        ", Encoder Drops: " + std::to_string(camera_state->encoder_drops);
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
                    unsigned int delay_in_second,
                    CameraControl *camera_control,
                    CameraEachSelect *camera_select,
                    MjpegServer *mjpeg_server) {
    if (ptp_params->network_sync) {
        uint64_t ptp_counter = sync_fetch_and_add(&ptp_params->ptp_counter, 1);
        printf("%lu\n", ptp_counter);
        std::cout << ptp_params->ptp_global_time << std::endl;

        while (!ptp_params->network_set_start_ptp) {
            // Handle Test Focus requests while waiting for recording to start.
            // Camera acquisition has NOT started yet here, so we can safely
            // enter free-run mode, sweep focus, then restore PTP settings.
            int focus_gen = camera_control
                                ? camera_control->focus_test_generation.load()
                                : 0;
            int gen_processed = (camera_select)
                                     ? camera_select->focus_test_gen_processed
                                     : 0;
            if (camera_control && focus_gen > gen_processed) {
                if (camera_select)
                    camera_select->focus_test_gen_processed = focus_gen;

                printf("FOCUS TEST cam %s: running sweep "
                       "(focus=%u gain=%u)\n",
                       camera_params->camera_serial.c_str(),
                       camera_params->focus, camera_params->gain);
                fflush(stdout);

                // Temporarily enter free-run mode to get real frames
                ptp_sync_off(&ecam->camera, camera_params);
                EVT_CameraExecuteCommand(&ecam->camera, "AcquisitionStart");

                // Brightness check from first live frame
                int bret = EVT_CameraGetFrame(&ecam->camera,
                                              &ecam->frame_recv, EVT_INFINITE);
                if (bret == 0) {
                    double brightness = compute_mean_brightness(
                        (const unsigned char *)ecam->frame_recv.imagePtr,
                        camera_params->width, camera_params->height,
                        camera_params->gpu_direct);
                    printf("FOCUS TEST cam %s: brightness=%.1f\n",
                           camera_params->camera_serial.c_str(), brightness);
                    fflush(stdout);
                    if (brightness > 1.0 && brightness < BRIGHTNESS_LOW) {
                        int new_gain = std::min(
                            (int)(camera_params->gain *
                                  BRIGHTNESS_TARGET / brightness),
                            2000);
                        printf("FOCUS TEST cam %s: adjusting gain %u -> %d\n",
                               camera_params->camera_serial.c_str(),
                               camera_params->gain, new_gain);
                        fflush(stdout);
                        update_gain_value(&ecam->camera, new_gain,
                                         camera_params);
                    }
                }

                // Focus sweep ±50 around current value, step 5
                int sweep_center = (int)camera_params->focus;
                int sweep_min = std::max((int)camera_params->focus_min,
                                         sweep_center - 50);
                int sweep_max = std::min((int)camera_params->focus_max,
                                         sweep_center + 50);
                int best_focus    = sweep_center;
                double best_sharp = -1.0;

                for (int f = sweep_min; f <= sweep_max; f += 5) {
                    EVT_CameraSetUInt32Param(&ecam->camera, "Focus",
                                             (unsigned int)f);
                    usleep(150000); // motor settle + fresh frame

                    int ret = EVT_CameraGetFrame(&ecam->camera,
                                                 &ecam->frame_recv, EVT_INFINITE);
                    if (ret != 0) {
                        printf("FOCUS TEST cam %s: GetFrame failed "
                               "focus=%d (ret=%d)\n",
                               camera_params->camera_serial.c_str(), f, ret);
                        fflush(stdout);
                        continue;
                    }

                    double sv = compute_laplacian_variance(
                        (const unsigned char *)ecam->frame_recv.imagePtr,
                        camera_params->width, camera_params->height,
                        camera_params->gpu_direct);

                    printf("FOCUS TEST cam %s: focus=%d sharpness=%.2f\n",
                           camera_params->camera_serial.c_str(), f, sv);
                    fflush(stdout);

                    if (sv > best_sharp) {
                        best_sharp = sv;
                        best_focus = f;
                    }
                }

                EVT_CameraExecuteCommand(&ecam->camera, "AcquisitionStop");
                update_focus_value(&ecam->camera, best_focus, camera_params);
                printf("FOCUS TEST cam %s: DONE best_focus=%d "
                       "sharpness=%.2f\n",
                       camera_params->camera_serial.c_str(),
                       best_focus, best_sharp);
                fflush(stdout);

                // Restore PTP mode — ready for recording
                ptp_camera_sync(&ecam->camera, camera_params);
            }

            // Handle SETFOCUS: apply focus + grab one frame for preview
            if (camera_control) {
                thread_local int sf_gen_seen = 0;
                int sf_gen = camera_control->setfocus.generation.load();
                if (sf_gen > sf_gen_seen &&
                    camera_control->setfocus.camera_serial ==
                        camera_params->camera_serial) {
                    sf_gen_seen = sf_gen;
                    int fv = camera_control->setfocus.focus_value;
                    printf("SETFOCUS cam %s focus=%d step1: update_focus\n",
                           camera_params->camera_serial.c_str(), fv);
                    fflush(stdout);
                    update_focus_value(&ecam->camera, fv, camera_params);
                    printf("SETFOCUS cam %s focus=%d applied "
                           "(preview will come from next recording)\n",
                           camera_params->camera_serial.c_str(), fv);
                    fflush(stdout);
                }
            }

            usleep(10);
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
                          FrameSaver *frame_saver, void *frame_detector,
                          MjpegServer *mjpeg_server = nullptr) {
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
            (camera_state->frame_count != 0))
            camera_state->dropped_frames++;
        else {
            camera_state->frames_recd++;
        }

        // In GVSP there is no id 0 so when 16 bit id counter in camera is max
        // then the next id is 1 so set prev id to 0 for math above.
        if (ecam->frame_recv.frame_id == 65535)
            camera_state->id_prev = 0;
        else
            camera_state->id_prev = ecam->frame_recv.frame_id;

        // push the image data to encode, or display
        if (camera_control->record_video && camera_select->record) {
            bool queued = gpu_encoder->PushToDisplay(
                ecam->frame_recv.imagePtr, ecam->frame_recv.bufferSize,
                ecam->frame_recv.size_x, ecam->frame_recv.size_y,
                ecam->frame_recv.pixel_type, ecam->frame_recv.timestamp,
                camera_state->frame_count, real_time);
            if (!queued) {
                camera_state->encoder_drops++;
                if (camera_state->encoder_drops % 100 == 1)
                    printf("WARNING: cam %s encoder queue full, frame %llu "
                           "dropped (total encoder drops: %u)\n",
                           camera_params->camera_serial.c_str(),
                           camera_state->frame_count,
                           camera_state->encoder_drops);
            }
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
        if (detector &&
            camera_select->frame_detect_state.load() == State_Copy_New_Frame) {
            detector->notify_frame_ready(ecam->frame_recv.imagePtr, 0);
        }
#endif

        if (camera_select->frame_save_state.load() == State_Copy_New_Frame) {
            frame_saver->notify_frame_ready(ecam->frame_recv.imagePtr);
        }

        // SETFOCUS: apply focus, wait for motor to settle, then grab preview
        // Skip during recording -- camera API calls and cudaMemcpy2D stall
        // the capture thread and cause frame drops.
        if (camera_control &&
            !(camera_control->record_video && camera_select->record)) {
            thread_local int sf_rec_gen = 0;
            thread_local int sf_wait_frames = 0;

            int sg = camera_control->setfocus.generation.load();
            if (sg > sf_rec_gen &&
                camera_control->setfocus.camera_serial ==
                    camera_params->camera_serial) {
                sf_rec_gen = sg;
                // Apply focus now, grab preview after motor settles
                int requested_focus = camera_control->setfocus.focus_value;
                update_focus_value(&ecam->camera, requested_focus,
                                   camera_params);
                // Read back actual focus to verify motor moved
                unsigned int actual_focus = 0;
                EVT_CameraGetUInt32Param(&ecam->camera, "Focus",
                                         &actual_focus);
                sf_wait_frames = 30; // wait ~30 frames for motor
                printf("SETFOCUS cam %s set=%d actual=%u waiting %d frames\n",
                       camera_params->camera_serial.c_str(),
                       requested_focus, actual_focus, sf_wait_frames);
                fflush(stdout);
            }

            if (sf_wait_frames > 0) {
                sf_wait_frames--;
                if (sf_wait_frames == 0) {
                    auto jpeg = make_preview_jpeg(
                        (const unsigned char *)ecam->frame_recv.imagePtr,
                        camera_params->width, camera_params->height,
                        camera_params->gpu_direct);
                    FILE *fp = fopen(
                        "/home/ratan/orange_data/remote_preview.jpg", "wb");
                    if (fp) {
                        fwrite(jpeg.data(), 1, jpeg.size(), fp);
                        fclose(fp);
                    }
                    std::lock_guard<std::mutex> lk(
                        camera_control->setfocus.reply_mu);
                    camera_control->setfocus.reply_jpeg = std::move(jpeg);
                    camera_control->setfocus.reply_ready = true;
                    printf("SETFOCUS cam %s preview captured\n",
                           camera_params->camera_serial.c_str());
                    fflush(stdout);
                }
            }
        }

        // Push preview JPEG to any connected MJPEG viewers (every 15 frames)
        // Only when NOT recording -- the synchronous cudaMemcpy2D stalls the
        // capture thread and causes frame drops on cross-NUMA GPUs.
        if (mjpeg_server && camera_state->frame_count % 15 == 0 &&
            !(camera_control->record_video && camera_select->record)) {
            auto jpeg = make_preview_jpeg(
                (const unsigned char *)ecam->frame_recv.imagePtr,
                camera_params->width, camera_params->height,
                camera_params->gpu_direct);
            mjpeg_server->push(jpeg);
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
                    INDIGOSignalBuilder *indigo_signal_builder) {
    CHECK(cudaSetDevice(camera_params->gpu_id));
    CameraState camera_state;
    PTPState ptp_state;
    StopWatch w;

    // MJPEG stream: port 8080 + camera_id. View with browser or ffplay.
    int mjpeg_port = 8080 + camera_params->camera_id;
    MjpegServer mjpeg_server(mjpeg_port);
    mjpeg_server.start();
    printf("MJPEG stream cam %s -> http://vlan-dosa0:%d\n",
           camera_params->camera_serial.c_str(), mjpeg_port);
    fflush(stdout);
    // Also write to file so the URL is easy to find
    {
        FILE *f = fopen("/tmp/mjpeg_streams.txt", "a");
        if (f) {
            fprintf(f, "cam %s -> http://vlan-dosa0:%d\n",
                    camera_params->camera_serial.c_str(), mjpeg_port);
            fclose(f);
        }
    }

#ifndef HEADLESS
    FrameDetector *frame_detector = nullptr;
    if (camera_select->detect_mode == Detect3D_Standoff ||
        camera_select->detect_mode == Detect2D_Standoff) {
        frame_detector = new FrameDetector(camera_params, camera_select);
        frame_detector->start();

        while (detector_counter.load() !=
               camera_select->total_standoff_detector) {
            // printf(".");
            // fflush(stdout);
            usleep(10);
        }
        camera_select->frame_detect_state.store(State_Copy_New_Frame);
    }

    COpenGLDisplay *openGLDisplay = nullptr;
    if (camera_select->stream_on) {
        openGLDisplay =
            new COpenGLDisplay("", camera_params, camera_select, display_buffer,
                               indigo_signal_builder);
        openGLDisplay->StartThread();
    }
#endif

    FrameSaver frame_saver(camera_params, camera_select);
    frame_saver.start();

    GPUVideoEncoder *gpu_encoder = nullptr;
    bool encoder_ready_signal = false;
    if (camera_control->record_video && camera_select->record) {
        gpu_encoder = new GPUVideoEncoder("", camera_params, encoder_setup,
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
        start_ptp_sync(&ptp_state, ptp_params, camera_params, ecam, 3,
                       camera_control, camera_select, &mjpeg_server);
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
                      &frame_saver, frame_detector, &mjpeg_server);
#else
        get_one_frame(&camera_state, camera_select, camera_control, ecam,
                      camera_params, &ptp_state, nullptr, gpu_encoder,
                      &frame_saver, nullptr, &mjpeg_server);
#endif
        if (ptp_params->network_sync && ptp_params->network_set_stop_ptp) {
            if (ptp_state.frame_ts > ptp_params->ptp_stop_time) {
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
    mjpeg_server.stop();
    report_statistics(camera_params, &camera_state, time_diff);

    // Post-acquisition focus/gain analysis triggered by "Test Focus" button.
    // Uses live frames in free-run mode after AcquisitionStop — no video needed.
    int current_gen = camera_control->focus_test_generation.load();
    if (current_gen > camera_select->focus_test_gen_processed) {
        camera_select->focus_test_gen_processed = current_gen;

        printf("POST-ACQUIRE cam %s: starting focus/gain analysis "
               "(focus=%u gain=%u)\n",
               camera_params->camera_serial.c_str(),
               camera_params->focus, camera_params->gain);
        fflush(stdout);

        // Enter free-run mode: after AcquisitionStop + ptp_sync_off the camera
        // captures normally — no PTP gate, no dark frames.
        ptp_sync_off(&ecam->camera, camera_params);
        EVT_CameraExecuteCommand(&ecam->camera, "AcquisitionStart");

        // Warm-up: grab one frame to get a real brightness reading
        int warm_ret = EVT_CameraGetFrame(&ecam->camera, &ecam->frame_recv, EVT_INFINITE);
        if (warm_ret == 0) {
            double brightness = compute_mean_brightness(
                (const unsigned char *)ecam->frame_recv.imagePtr,
                camera_params->width, camera_params->height,
                camera_params->gpu_direct);

            printf("POST-ACQUIRE cam %s: brightness=%.1f\n",
                   camera_params->camera_serial.c_str(), brightness);
            fflush(stdout);

            if (brightness > 1.0 && brightness < BRIGHTNESS_LOW) {
                int new_gain = std::min(
                    (int)(camera_params->gain * BRIGHTNESS_TARGET / brightness),
                    2000);
                printf("POST-ACQUIRE cam %s: brightness too low, "
                       "adjusting gain %u -> %d\n",
                       camera_params->camera_serial.c_str(),
                       camera_params->gain, new_gain);
                fflush(stdout);
                update_gain_value(&ecam->camera, new_gain, camera_params);
            }
        } else {
            printf("POST-ACQUIRE cam %s: warm-up GetFrame failed (ret=%d)\n",
                   camera_params->camera_serial.c_str(), warm_ret);
            fflush(stdout);
        }

        // Focus sweep: ±50 around current focus, step 5
        int sweep_center = (int)camera_params->focus;
        int sweep_min = std::max((int)camera_params->focus_min,
                                 sweep_center - 50);
        int sweep_max = std::min((int)camera_params->focus_max,
                                 sweep_center + 50);
        int best_focus    = sweep_center;
        double best_sharp = -1.0;

        for (int f = sweep_min; f <= sweep_max; f += 5) {
            EVT_CameraSetUInt32Param(&ecam->camera, "Focus", (unsigned int)f);
            usleep(150000); // 150 ms: motor settle + fresh frame

            int ret = EVT_CameraGetFrame(&ecam->camera, &ecam->frame_recv, EVT_INFINITE);
            if (ret != 0) {
                printf("POST-ACQUIRE cam %s: GetFrame failed at focus=%d "
                       "(ret=%d)\n",
                       camera_params->camera_serial.c_str(), f, ret);
                fflush(stdout);
                continue;
            }

            double sv = compute_laplacian_variance(
                (const unsigned char *)ecam->frame_recv.imagePtr,
                camera_params->width, camera_params->height,
                camera_params->gpu_direct);

            printf("POST-ACQUIRE cam %s: focus=%d sharpness=%.2f\n",
                   camera_params->camera_serial.c_str(), f, sv);
            fflush(stdout);

            if (sv > best_sharp) {
                best_sharp = sv;
                best_focus = f;
            }
        }

        EVT_CameraExecuteCommand(&ecam->camera, "AcquisitionStop");

        update_focus_value(&ecam->camera, best_focus, camera_params);
        printf("POST-ACQUIRE cam %s: DONE best_focus=%d best_sharpness=%.2f\n",
               camera_params->camera_serial.c_str(), best_focus, best_sharp);
        fflush(stdout);

        // Restore PTP camera settings for next recording
        ptp_camera_sync(&ecam->camera, camera_params);
    }
}
