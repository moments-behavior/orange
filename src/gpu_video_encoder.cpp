#include "video_capture.h"
#if defined(__GNUC__)
#include <unistd.h>
#endif
#include "gpu_video_encoder.h"
#include "global.h"   // g_cam_brightness
#include "kernel.cuh" // launch_brightness_sum
#include "utils.h"    // make_npp_stream_context
#include <chrono>
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

static inline void encode_frame(EncoderContext *encoder, Debayer *debayer,
                                bool force_idr) {
    // encoding
    const NvEncInputFrame *encoderInputFrame =
        encoder->pEnc->GetNextInputFrame();
    NvEncoderCuda::CopyToDeviceFrame(
        encoder->cuContext, debayer->d_debayer, 0,
        (CUdeviceptr)encoderInputFrame->inputPtr, (int)encoderInputFrame->pitch,
        encoder->pEnc->GetEncodeWidth(), encoder->pEnc->GetEncodeHeight(),
        CU_MEMORYTYPE_DEVICE, encoderInputFrame->bufferFormat,
        encoderInputFrame->chromaOffsets, encoderInputFrame->numChromaPlanes);

    // The first frame of a new chunk must be decodable on its own: force an
    // IDR and repeat the sequence headers (SPS/PPS, and VPS for HEVC).
    NV_ENC_PIC_PARAMS pic_params = {};
    if (force_idr)
        pic_params.encodePicFlags =
            NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS;
    encoder->pEnc->EncodeFrame(encoder->vPacket,
                               force_idr ? &pic_params : nullptr);
}

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0)
        .count();
}

Writer *GPUVideoEncoder::open_writer(int chunk) {
    Writer *w = new Writer();
    w->chunk = chunk;
    std::string folder = folder_name;
    if (chunk_clock.enabled()) {
        folder = chunk_folder_name(folder_name, chunk);
        make_folder(folder); // every camera calls this; existing is fine
    }
    w->video_file = folder + "/Cam" + camera_params->camera_serial + ".mp4";
    w->metadata_file =
        folder + "/Cam" + camera_params->camera_serial + "_meta.csv";
    w->keyframe_file =
        folder + "/Cam" + camera_params->camera_serial + "_keyframe.csv";
    w->summary.camera_serial = camera_params->camera_serial;

    if (encoder_setup.find("h264") != std::string::npos) {
        w->video = new FFmpegWriter(
            AV_CODEC_ID_H264, camera_params->width, camera_params->height,
            camera_params->frame_rate, w->video_file.c_str(),
            w->keyframe_file.c_str());
    } else if (encoder_setup.find("hevc") != std::string::npos) {
        w->video = new FFmpegWriter(
            AV_CODEC_ID_HEVC, camera_params->width, camera_params->height,
            camera_params->frame_rate, w->video_file.c_str(),
            w->keyframe_file.c_str());
    } else {
        std::cout << "codec not supported" << '\n';
    }
    // The .mp4 was created by avio_open in the FFmpegWriter ctor above; hand it
    // back to the invoking user so recordings aren't left root-owned (sudo).
    chown_to_invoking_user(w->video_file);

    w->metadata = new std::ofstream();
    open_metadata_file(w->metadata, w->metadata_file);
    chown_to_invoking_user(w->metadata_file);
    return w;
}

// Mux the packets NVENC just returned. They come out in submission order, one
// per frame, so each one belongs to the oldest frame in pending_frames. The
// first packet of a new chunk retires the previous chunk's writer.
void GPUVideoEncoder::route_packets() {
    for (std::vector<uint8_t> &packet : encoder.vPacket) {
        Writer *w = packet_writer;
        if (!pending_frames.empty()) {
            w = pending_frames.front();
            pending_frames.pop_front();
        }
        if (w != packet_writer) {
            retire_writer(packet_writer);
            packet_writer = w;
        }
        bool key = false;
        w->video->write_packet(packet.data(), (int)packet.size(),
                               (int)w->packets, &key);
        if (w->packets == 0)
            w->summary.first_packet_keyframe = key;
        w->packets++;
        encoder.num_frame_encode++;
    }
}

// Close a writer's files on the finalizer thread: av_write_trailer writes the
// mp4 index (several MB for a 30 min file) and must not hold up encoding.
void GPUVideoEncoder::retire_writer(Writer *w) {
    if (!w)
        return;
    bool report = chunk_clock.enabled();
    std::string session = folder_name;
    double seconds = chunk_seconds;
    finalizer->post([w, report, session, seconds]() {
        auto t0 = std::chrono::steady_clock::now();
        delete w->video; // writes the trailer and closes the file
        w->video = nullptr;
        w->metadata->close();
        delete w->metadata;
        w->metadata = nullptr;
        w->summary.frames = w->packets;
        w->summary.finalize_ms = ms_since(t0);
        if (report)
            chunk_registry::camera_finished_chunk(session, w->chunk, seconds,
                                                  w->summary);
        delete w;
    });
}

GPUVideoEncoder::GPUVideoEncoder(const char *name, CameraParams *camera_params,
                                 CameraEachSelect *camera_select,
                                 std::string encoder_setup,
                                 std::string folder_name,
                                 bool *encoder_ready_signal,
                                 double chunk_seconds, PTPParams *ptp_params)
    : CThreadWorker(name), camera_params(camera_params),
      camera_select(camera_select), encoder_setup(encoder_setup),
      folder_name(folder_name), encoder_ready_signal(encoder_ready_signal),
      chunk_seconds(chunk_seconds), ptp_params(ptp_params) {
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

    // Average-brightness sample: wall-clock throttled (~10 Hz) and subsampled, on
    // its own stream. d_orig is valid here (the copy above ran on the synchronous
    // default stream) and isn't overwritten until the next sampled frame, so the
    // only sync is a few-microsecond scalar readback that happens ~10x/second —
    // it does not gate the NVENC work that follows on the default stream.
    double bnow = steady_now_seconds();
    if (camera_params->camera_id >= 0 && camera_params->camera_id < kMaxCameras &&
        bnow >= bright_next_sample_time) {
        bright_next_sample_time = bnow + kBrightSamplePeriodSec;
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

    // Which chunk does this exposure belong to? The first frame fixes the
    // PTP reference that every camera shares (see recording_chunks.h).
    bool force_idr = false;
    if (chunk_clock.enabled()) {
        if (!chunk_clock.has_reference()) {
            unsigned long long gate = ptp_params ? ptp_params->ptp_global_time : 0;
            chunk_clock.set_reference(entry.timestamp, gate);
            printf("[chunk] Cam%s first frame ts %llu, gate %llu, offset from "
                   "gate %lld ns, reference %llu\n",
                   camera_params->camera_serial.c_str(), entry.timestamp, gate,
                   chunk_clock.first_frame_offset_from_gate(),
                   chunk_clock.reference());
        }
        int chunk = chunk_clock.chunk_of(entry.timestamp);
        if (chunk != row_writer->chunk) {
            auto t0 = std::chrono::steady_clock::now();
            row_writer = open_writer(chunk);
            force_idr = true;
            printf("[chunk] Cam%s -> chunk %05d at frame_id %llu ts %llu "
                   "(open %.1f ms, encoder queue %d, finalizer backlog %zu)\n",
                   camera_params->camera_serial.c_str(), chunk, entry.frame_id,
                   entry.timestamp, ms_since(t0), GetCountQueueInSize(),
                   finalizer->pending());
        }
    }

    Writer *w = row_writer;
    pending_frames.push_back(w);
    encode_frame(&encoder, &debayer, force_idr);
    route_packets();

    write_metadata(w->metadata, entry.frame_id, entry.timestamp,
                   entry.timestamp_sys, entry.ptp_offset);
    if (w->summary.metadata_rows == 0) {
        w->summary.first_frame_id = entry.frame_id;
        w->summary.first_timestamp = entry.timestamp;
    }
    w->summary.last_frame_id = entry.frame_id;
    w->summary.last_timestamp = entry.timestamp;
    w->summary.metadata_rows++;
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
    chunk_clock.configure(chunk_seconds, camera_params->frame_rate);
    if (chunk_clock.enabled())
        chunk_registry::register_camera(folder_name);
    finalizer = new ChunkFinalizer();
    row_writer = packet_writer = open_writer(0);

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

    // Flush the frames still inside NVENC, then close the last chunk.
    encoder.pEnc->EndEncode(encoder.vPacket);
    route_packets();
    encoder.pEnc->DestroyEncoder();
    retire_writer(packet_writer);
    if (row_writer != packet_writer) // only if NVENC lost frames at EOS
        retire_writer(row_writer);
    row_writer = packet_writer = nullptr;
    pending_frames.clear();
    delete finalizer; // waits until every chunk's files are closed
    finalizer = nullptr;

    std::string print_out;
    print_out += "\n" + camera_params->camera_serial;
    print_out += ", Frame encoded: " + std::to_string(encoder.num_frame_encode);
    std::cout << print_out << std::endl;

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
