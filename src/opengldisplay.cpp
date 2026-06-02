#include "image_processing.h"
#include "video_capture.h"
#if defined(__GNUC__)
#include <unistd.h>
#endif
#include "global.h"
#include "opengldisplay.h"
#include "utils.h"
#include <cuda_runtime_api.h>
#include <npp.h>
#include <nvToolsExt.h>
#include <stdio.h>
#include <string.h>

COpenGLDisplay::COpenGLDisplay(const char *name, CameraParams *camera_params,
                               CameraEachSelect *camera_select,
                               unsigned char *display_buffer)
    : CThreadWorker(name), camera_params(camera_params),
      camera_select(camera_select), display_buffer(display_buffer) {
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
}

COpenGLDisplay::~COpenGLDisplay() {
    cudaFree(frame_original.d_orig);
    cudaFree(debayer.d_debayer);
}

void COpenGLDisplay::ThreadRunning() {
    ck(cudaSetDevice(camera_params->gpu_id));
    CHECK(cudaStreamCreate(&stream));
    NppStreamContext npp_ctx =
        make_npp_stream_context(camera_params->gpu_id, stream);
    if (camera_select->downsample != 1) {
        ck(cudaMalloc((void **)&d_resize,
                      output_image_size.width * output_image_size.height * 4));
    }

    initalize_gpu_frame_async(&frame_original, camera_params, stream);
    initialize_gpu_debayer_async(&debayer, camera_params, 4, stream);

    using clock = std::chrono::steady_clock;

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

            nvtxRangePush("dgl_copy");
            if (camera_params->gpu_direct) {
                CHECK(cudaMemcpy2DAsync(
                    frame_original.d_orig, camera_params->width, entry.imagePtr,
                    camera_params->width, camera_params->width,
                    camera_params->height, cudaMemcpyDeviceToDevice, stream));

            } else {
                CHECK(cudaMemcpy2DAsync(
                    frame_original.d_orig, camera_params->width, entry.imagePtr,
                    camera_params->width, camera_params->width,
                    camera_params->height, cudaMemcpyHostToDevice, stream));
            }
            nvtxRangePop();

            CHECK(cudaStreamSynchronize(stream));
            nvtxRangePush("dgl_debayer");

            if (camera_params->color) {
                debayer_frame_gpu_rgba_ctx(camera_params, &frame_original,
                                           &debayer, npp_ctx);
            } else {
                duplicate_channel_gpu_4_ctx(camera_params, &frame_original,
                                            &debayer, npp_ctx);
            }
            nvtxRangePop();

            nvtxRangePush("dgl_copy_to_interop_buffer");
            if (camera_select->downsample != 1) {
                const NppStatus npp_result = nppiResize_8u_C4R_Ctx(
                    debayer.d_debayer, camera_params->width * sizeof(uchar4),
                    input_image_size, input_image_roi, (Npp8u *)d_resize,
                    output_image_size.width * sizeof(uchar4), output_image_size,
                    output_image_roi, NPPI_INTER_SUPER, npp_ctx);
                if (npp_result != NPP_SUCCESS) {
                    std::cerr << "Error executing resize in display -- code: "
                              << npp_result << std::endl;
                }
                CHECK(cudaMemcpy2DAsync(
                    display_buffer, output_image_size.width * 4, d_resize,
                    output_image_size.width * 4, output_image_size.width * 4,
                    output_image_size.height, cudaMemcpyDeviceToDevice,
                    stream));

            } else {
                CHECK(cudaMemcpy2DAsync(
                    display_buffer, output_image_size.width * 4,
                    debayer.d_debayer, output_image_size.width * 4,
                    output_image_size.width * 4, output_image_size.height,
                    cudaMemcpyDeviceToDevice, stream));
            }
            nvtxRangePop();
            CHECK(cudaStreamSynchronize(stream));
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
    WORKER_ENTRY *entriesOut[WORK_ENTRIES_MAX]; // entries got out from the
                                                // display thread; their frames
                                                // should be returned to the
                                                // driver queue.
    int entriesOutCount = WORK_ENTRIES_MAX;
    GetObjectsFromQueueOut((void **)entriesOut, &entriesOutCount);
    if (entriesOutCount) { // return the frames to driver, and put entries back
                           // to frameSaveEntriesFreeQueue
        for (int j = 0; j < entriesOutCount; j++) {
            workerEntriesFreeQueue[workerEntriesFreeQueueCount] = entriesOut[j];
            workerEntriesFreeQueueCount++;
        }
    }

    // get the free entry if there is one and put it into QueueIn
    if (workerEntriesFreeQueueCount) {
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
