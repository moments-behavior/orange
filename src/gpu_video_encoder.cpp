#include "video_capture.h"
#if defined(__GNUC__)
#include <unistd.h>
#endif
#include "gpu_video_encoder.h"
#include "global.h"   // g_cam_brightness
#include "kernel.cuh" // launch_brightness_sum
#include "utils.h"    // make_npp_stream_context
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <string.h>

template <class EncoderClass>
void InitializeEncoder(EncoderClass &pEnc, NvEncoderInitParam encodeCLIOptions,
                       NV_ENC_BUFFER_FORMAT eFormat) {
    NV_ENC_INITIALIZE_PARAMS initializeParams = {NV_ENC_INITIALIZE_PARAMS_VER};
    NV_ENC_CONFIG encodeConfig = {NV_ENC_CONFIG_VER};

    initializeParams.encodeConfig = &encodeConfig;
    pEnc->CreateDefaultEncoderParams(
        &initializeParams, encodeCLIOptions.GetEncodeGUID(),
        encodeCLIOptions.GetPresetGUID(), encodeCLIOptions.GetTuningInfo());
    encodeCLIOptions.SetInitParams(&initializeParams, eFormat);
    encodeCLIOptions.FullParamToString(&initializeParams);
    pEnc->CreateEncoder(&initializeParams);
}

static inline void initialize_encoder(EncoderContext *encoder,
                                      std::string encoder_str,
                                      CameraParams *camera_params) {
    encoder->eFormat = NV_ENC_BUFFER_FORMAT_ABGR;
    int gop_in_frames = camera_params->frame_rate * camera_params->gop;
    std::string encoder_str_with_fps =
        encoder_str + " -fps " + std::to_string(camera_params->frame_rate) +
        " -gop " + std::to_string(gop_in_frames);
    encoder->encodeCLIOptions =
        NvEncoderInitParam(encoder_str_with_fps.c_str());
    CUdevice cuDevice;
    ck(cuDeviceGet(&cuDevice, camera_params->gpu_id));
    encoder->cuContext = NULL;
    // cuCtxCreate gained a CUctxCreateParams* arg in CUDA 13 (_v4); CUDA 12.x is _v2.
#if CUDA_VERSION >= 13000
    ck(cuCtxCreate(&encoder->cuContext, NULL, 0, cuDevice));
#else
    ck(cuCtxCreate(&encoder->cuContext, 0, cuDevice));
#endif
    encoder->pEnc = new NvEncoderCuda(encoder->cuContext, camera_params->width,
                                      camera_params->height, encoder->eFormat);
    InitializeEncoder(encoder->pEnc, encoder->encodeCLIOptions,
                      encoder->eFormat);
}

static inline void open_metadata_file(std::ofstream *frame_metadata,
                                      std::string metadata_file) {
    frame_metadata->open(metadata_file.c_str());

    if (!(*frame_metadata)) {
        std::cout << "File did not open!";
        return;
    }
    *frame_metadata << "frame_id,timestamp,timestamp_sys,ptp_offset\n";
}

static inline void write_metadata(std::ofstream *metadata,
                                  unsigned long long frame_id,
                                  unsigned long long timestamp,
                                  uint64_t timestamp_sys, int ptp_offset) {
    *metadata << frame_id << "," << timestamp << "," << timestamp_sys << ","
              << ptp_offset << std::endl;
}

static inline void initialize_writer(Writer *writer,
                                     CameraParams *camera_params,
                                     std::string folder_name,
                                     std::string encoder_str) {
    // writer->video_file = folder_name + "/Cam" +
    // std::to_string(camera_params->camera_id) + ".mp4"; writer->metadata_file
    // = folder_name + "/Cam" + std::to_string(camera_params->camera_id) +
    // "_meta.csv";
    writer->video_file =
        folder_name + "/Cam" + camera_params->camera_serial + ".mp4";
    writer->metadata_file =
        folder_name + "/Cam" + camera_params->camera_serial + "_meta.csv";
    writer->keyframe_file =
        folder_name + "/Cam" + camera_params->camera_serial + "_keyframe.csv";

    if (encoder_str.find("h264") != std::string::npos) {
        std::cout << "h264 encoding" << '\n';
        writer->video = new FFmpegWriter(
            AV_CODEC_ID_H264, camera_params->width, camera_params->height,
            camera_params->frame_rate, writer->video_file.c_str(),
            writer->keyframe_file.c_str());
    } else if (encoder_str.find("hevc") != std::string::npos) {
        std::cout << "hevc encoding" << '\n';
        writer->video = new FFmpegWriter(
            AV_CODEC_ID_HEVC, camera_params->width, camera_params->height,
            camera_params->frame_rate, writer->video_file.c_str(),
            writer->keyframe_file.c_str());
    } else {
        std::cout << "codec not supported" << '\n';
    }
    // The .mp4 was created by avio_open in the FFmpegWriter ctor above; hand it
    // back to the invoking user so recordings aren't left root-owned (sudo).
    chown_to_invoking_user(writer->video_file);

    writer->metadata = new std::ofstream();
    open_metadata_file(writer->metadata, writer->metadata_file);
    chown_to_invoking_user(writer->metadata_file);
}

static inline void encode_frame(EncoderContext *encoder, FFmpegWriter *writer,
                                Debayer *debayer) {
    // encoding
    const NvEncInputFrame *encoderInputFrame =
        encoder->pEnc->GetNextInputFrame();
    NvEncoderCuda::CopyToDeviceFrame(
        encoder->cuContext, debayer->d_debayer, 0,
        (CUdeviceptr)encoderInputFrame->inputPtr, (int)encoderInputFrame->pitch,
        encoder->pEnc->GetEncodeWidth(), encoder->pEnc->GetEncodeHeight(),
        CU_MEMORYTYPE_DEVICE, encoderInputFrame->bufferFormat,
        encoderInputFrame->chromaOffsets, encoderInputFrame->numChromaPlanes);

    encoder->pEnc->EncodeFrame(encoder->vPacket);
    for (std::vector<uint8_t> &packet : encoder->vPacket) {
        // For each encoded packet
        writer->write_packet(packet.data(), (int)packet.size(),
                             encoder->num_frame_encode++);
    }
}

static inline void close_writer(EncoderContext *encoder, Writer *writer) {
    encoder->pEnc->EndEncode(encoder->vPacket);
    for (std::vector<uint8_t> &packet : encoder->vPacket) {
        writer->video->write_packet(packet.data(), (int)packet.size(),
                                    encoder->num_frame_encode++);
    }
    encoder->pEnc->DestroyEncoder();
    (*writer->metadata).close();
}

GPUVideoEncoder::GPUVideoEncoder(const char *name, CameraParams *camera_params,
                                 CameraEachSelect *camera_select,
                                 std::string encoder_setup,
                                 std::string folder_name,
                                 bool *encoder_ready_signal)
    : CThreadWorker(name), camera_params(camera_params),
      camera_select(camera_select), encoder_setup(encoder_setup),
      folder_name(folder_name), encoder_ready_signal(encoder_ready_signal) {
    memset(workerEntries, 0, sizeof(workerEntries));
    workerEntriesFreeQueueCount = ENCODER_ENTRIES_MAX;
    for (int i = 0; i < workerEntriesFreeQueueCount; i++) {
        workerEntriesFreeQueue[i] = &workerEntries[i];
    }
}

GPUVideoEncoder::~GPUVideoEncoder() {}

void GPUVideoEncoder::ProcessOneFrame(void *f) {
    WORKER_ENTRY entry = *(WORKER_ENTRY *)f;
    PutObjectToQueueOut(f);

    // copy frame from cpu to gpu
    ck(cudaMemcpy2D(frame_original.d_orig, camera_params->width, entry.imagePtr,
                    camera_params->width, camera_params->width,
                    camera_params->height, cudaMemcpyHostToDevice));

    // Average-brightness sample: throttled (~10 Hz) and subsampled, on its own
    // stream. d_orig is valid here (the copy above ran on the synchronous default
    // stream) and isn't overwritten until the next sampled frame, so the only
    // sync is a few-microsecond scalar readback that happens ~10x/second — it
    // does not gate the NVENC work that follows on the default stream.
    if (camera_params->camera_id >= 0 && camera_params->camera_id < kMaxCameras &&
        ++bright_frame_counter % bright_interval == 0) {
        launch_brightness_sum(frame_original.d_orig, camera_params->width,
                              camera_params->height, bright_stride, d_bright_sum,
                              bright_stream);
        ck(cudaMemcpyAsync(h_bright_sum, d_bright_sum,
                           sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                           bright_stream));
        ck(cudaStreamSynchronize(bright_stream));
        float mean = (float)((double)(*h_bright_sum) / (double)bright_sample_count);
        g_cam_brightness[camera_params->camera_id].store(
            mean, std::memory_order_relaxed);
    }

    if (camera_params->color) {
        debayer_frame_gpu_rgba_ctx(camera_params, &frame_original, &debayer,
                                   npp_ctx);
    } else {
        duplicate_channel_gpu_4_ctx(camera_params, &frame_original, &debayer,
                                    npp_ctx);
    }

    encode_frame(&encoder, writer.video, &debayer);
    write_metadata(writer.metadata, entry.frame_id, entry.timestamp,
                   entry.timestamp_sys, entry.ptp_offset);
}

void GPUVideoEncoder::ThreadRunning() {
    ck(cudaSetDevice(camera_params->gpu_id));
    // innitialization
    initalize_gpu_frame(&frame_original, camera_params);
    initialize_gpu_debayer(&debayer, camera_params, 4);
    // ProcessOneFrame uses the synchronous default stream (0) for its copies/NPP.
    npp_ctx = make_npp_stream_context(camera_params->gpu_id, 0);

    // Brightness sampling resources: a dedicated (non-default) stream so the
    // tiny reduction never serializes against the encoder's default-stream work,
    // plus a device accumulator and a pinned host slot for the scalar readback.
    ck(cudaStreamCreate(&bright_stream));
    ck(cudaMalloc((void **)&d_bright_sum, sizeof(unsigned long long)));
    ck(cudaMallocHost((void **)&h_bright_sum, sizeof(unsigned long long)));
    bright_stride = 8;
    {
        const int target_hz = 10;
        int fr = (int)camera_params->frame_rate;
        bright_interval = fr > target_hz ? fr / target_hz : 1;
    }
    {
        int nx = (camera_params->width + bright_stride - 1) / bright_stride;
        int ny = (camera_params->height + bright_stride - 1) / bright_stride;
        bright_sample_count = (long)nx * ny;
        if (bright_sample_count < 1)
            bright_sample_count = 1;
    }
    if (camera_params->camera_id >= 0 && camera_params->camera_id < kMaxCameras)
        g_cam_brightness[camera_params->camera_id].store(
            -1.0f, std::memory_order_relaxed); // -1 = no sample yet

    initialize_encoder(&encoder, encoder_setup, camera_params);
    initialize_writer(&writer, camera_params, folder_name, encoder_setup);

    *encoder_ready_signal = true;
    while (IsMachineOn()) {
        void *f = GetObjectFromQueueIn();
        if (f) {
            ProcessOneFrame(f);
            camera_select->encoder_fps_estimator.update();
        } else {
            std::this_thread::sleep_for(
                std::chrono::microseconds(200)); // or 1ms
        }
    }

    // empty queue
    while (GetCountQueueInSize()) {
        void *f = GetObjectFromQueueIn();
        if (f) {
            ProcessOneFrame(f);
        }
    }

    close_writer(&encoder, &writer);
    std::string print_out;
    print_out += "\n" + camera_params->camera_serial;
    print_out += ", Frame encoded: " + std::to_string(encoder.num_frame_encode);
    std::cout << print_out << std::endl;

    delete writer.video;
    writer.video = nullptr;
    delete writer.metadata;
    writer.metadata = nullptr;
    delete encoder.pEnc;
    cudaFree(frame_original.d_orig);
    cudaFree(debayer.d_debayer);
    cudaFree(d_bright_sum);
    cudaFreeHost(h_bright_sum);
    if (bright_stream)
        cudaStreamDestroy(bright_stream);
}

bool GPUVideoEncoder::PushToDisplay(void *imagePtr, size_t bufferSize,
                                    int width, int height, int pixelFormat,
                                    unsigned long long timestamp,
                                    unsigned long long frame_id,
                                    uint64_t timestamp_sys, int ptp_offset) {
    WORKER_ENTRY
    *entriesOut[ENCODER_ENTRIES_MAX]; // entris got out from saver thread,
                                      // their frames should be returned to
                                      // driver queue.
    int entriesOutCount = ENCODER_ENTRIES_MAX;
    GetObjectsFromQueueOut((void **)entriesOut, &entriesOutCount);
    if (entriesOutCount) { // return the frames to driver, and put entries back
                           // to frameSaveEntriesFreeQueue
        // printf("++++++++++++++++++++++++ %s %s %d get WORKER_ENTRY from out
        // entriesOutCount: %d\n", __FILE__, __FUNCTION__, __LINE__,
        // entriesOutCount);
        for (int j = 0; j < entriesOutCount; j++) {
            workerEntriesFreeQueue[workerEntriesFreeQueueCount] = entriesOut[j];
            workerEntriesFreeQueueCount++;
        }
    }

    // get the free entry if there is one and put in to QueueIn, otherwise
    // EVT_CameraQueueFrame.
    if (workerEntriesFreeQueueCount) {
        // printf("++++++++++++++++++++++++ %s %s %d put WORKER_ENTRY to in
        // workerEntriesFreeQueueCount: %d\n", __FILE__, __FUNCTION__, __LINE__,
        // workerEntriesFreeQueueCount);
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
        entry->timestamp_sys = timestamp_sys;
        entry->ptp_offset = ptp_offset;
        PutObjectToQueueIn(entry);
        return true;
    }
    return false;
}
