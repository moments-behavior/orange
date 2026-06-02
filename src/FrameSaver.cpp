#include "FrameSaver.h"
#include "global.h"
#include "image_processing.h"
#include "kernel.cuh"
#include "utils.h"
#include <algorithm>
#include <npp.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

FrameSaver::FrameSaver(CameraParams *params, CameraEachSelect *select)
    : camera_params(params), camera_select(select), running(false) {}

FrameSaver::~FrameSaver() {
    stop();
    cudaFreeAsync(frame_process.frame_original.d_orig, stream);
    cudaFreeAsync(frame_process.debayer.d_debayer, stream);
    cudaFreeHost(frame_process.frame_cpu.frame);
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaStreamDestroy(stream));
}

void FrameSaver::start() {
    running.store(true);
    worker_thread = std::thread(&FrameSaver::thread_loop, this);
}

void FrameSaver::stop() {
    if (running.load()) {
        running.store(false);
        cv.notify_all();
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
    }
}

void FrameSaver::notify_frame_ready(void *device_image_ptr) {
    if (camera_params->gpu_direct) {
        CHECK(cudaMemcpy2D(frame_process.frame_original.d_orig,
                           camera_params->width, device_image_ptr,
                           camera_params->width, camera_params->width,
                           camera_params->height, cudaMemcpyDeviceToDevice));
    } else {
        CHECK(cudaMemcpy2D(frame_process.frame_original.d_orig,
                           camera_params->width, device_image_ptr,
                           camera_params->width, camera_params->width,
                           camera_params->height, cudaMemcpyHostToDevice));
    }
    camera_select->sigs->frame_save_state.store(State_Frame_Copy_Done);
    cv.notify_one();
}

void FrameSaver::thread_loop() {
    ck(cudaSetDevice(camera_params->gpu_id));
    CHECK(cudaStreamCreate(&stream));
    initalize_gpu_frame_async(&frame_process.frame_original, camera_params,
                              stream);
    initialize_gpu_debayer_async(&frame_process.debayer, camera_params, 4,
                                 stream);
    initialize_pinned_cpu_frame(&frame_process.frame_cpu, camera_params, 3);

    CHECK(cudaMallocAsync((void **)&frame_process.d_convert,
                          camera_params->width * camera_params->height * 3,
                          stream));

    while (running.load()) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] {
            return camera_select->sigs->frame_save_state.load() ==
                       State_Frame_Copy_Done ||
                   !running.load();
        });

        if (!running.load())
            break;
        lock.unlock();

        // GPU processing
        if (camera_params->color) {
            debayer_frame_gpu(camera_params, &frame_process.frame_original,
                              &frame_process.debayer);
        } else {
            duplicate_channel_gpu(camera_params, &frame_process.frame_original,
                                  &frame_process.debayer);
        }

        rgba2rgb_convert(frame_process.d_convert,
                         frame_process.debayer.d_debayer, camera_params->width,
                         camera_params->height, stream);

        ck(cudaMemcpy2DAsync(frame_process.frame_cpu.frame,
                             camera_params->width * 3, frame_process.d_convert,
                             camera_params->width * 3, camera_params->width * 3,
                             camera_params->height, cudaMemcpyDeviceToHost,
                             stream));

        ck(cudaStreamSynchronize(stream));

        // Save to disk (RGB8 interleaved, written without OpenCV via stb).
        std::string fmt = camera_select->frame_save_format;
        std::transform(fmt.begin(), fmt.end(), fmt.begin(), ::tolower);
        std::string image_name = camera_select->picture_save_folder + "/" +
                                 camera_params->camera_serial + "_" +
                                 camera_select->frame_save_name + "." + fmt;

        std::cout << "Saving " << image_name << std::endl;
        const int w = camera_params->width;
        const int h = camera_params->height;
        const unsigned char *rgb = frame_process.frame_cpu.frame;
        int ok = 0;
        if (fmt == "jpg" || fmt == "jpeg") {
            ok = stbi_write_jpg(image_name.c_str(), w, h, 3, rgb, 95);
        } else {
            // default to PNG (lossless); covers "png" and any other format.
            ok = stbi_write_png(image_name.c_str(), w, h, 3, rgb, w * 3);
        }
        if (!ok) {
            std::cerr << "FrameSaver: failed to write " << image_name
                      << std::endl;
        }

        camera_select->pictures_counter++;
        camera_select->sigs->frame_save_state.store(State_Frame_Idle);
    }
}
