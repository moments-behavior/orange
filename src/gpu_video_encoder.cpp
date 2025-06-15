// src/gpu_video_encoder.cpp

#include "gpu_video_encoder.h"
#include "kernel.cuh"
#include <npp.h>
#include <nppi.h>
#include <nppi_color_conversion.h>
#include <iostream>
#include "global.h"
#include "NvEncoder/NvEncoder.h" // <--- FIX: Added this include for NV_ENC_BUFFER_FORMAT

// Helper to initialize the NvEncoder
template <class EncoderClass>
void InitializeEncoder(EncoderClass &pEnc, NvEncoderInitParam encodeCLIOptions, NV_ENC_BUFFER_FORMAT eFormat)
{
    NV_ENC_INITIALIZE_PARAMS initializeParams = {NV_ENC_INITIALIZE_PARAMS_VER};
    NV_ENC_CONFIG encodeConfig = {NV_ENC_CONFIG_VER};
    initializeParams.encodeConfig = &encodeConfig;
    pEnc.CreateDefaultEncoderParams(&initializeParams, encodeCLIOptions.GetEncodeGUID(), encodeCLIOptions.GetPresetGUID(), encodeCLIOptions.GetTuningInfo());
    encodeCLIOptions.SetInitParams(&initializeParams, eFormat);
    pEnc.CreateEncoder(&initializeParams);
}

// Helper to initialize the FFmpeg-based file writer
static inline void initialize_writer(Writer *writer, CameraParams *camera_params, std::string folder_name, std::string encoder_str)
{
    writer->video_file = folder_name + "/Cam" + camera_params->camera_serial + ".mp4";
    writer->metadata_file = folder_name + "/Cam" + camera_params->camera_serial + "_meta.csv";
    writer->keyframe_file = folder_name + "/Cam" + camera_params->camera_serial + "_keyframe.csv";

    if (encoder_str.find("h264") != std::string::npos) {
        writer->video = new FFmpegWriter(AV_CODEC_ID_H264, camera_params->width, camera_params->height, camera_params->frame_rate, writer->video_file.c_str(), writer->keyframe_file.c_str());
    } else if (encoder_str.find("hevc") != std::string::npos){
        writer->video = new FFmpegWriter(AV_CODEC_ID_HEVC, camera_params->width, camera_params->height, camera_params->frame_rate, writer->video_file.c_str(), writer->keyframe_file.c_str());
    } else {
        std::cout << "codec not supported" << '\n';
        writer->video = new FFmpegWriter(AV_CODEC_ID_H264, camera_params->width, camera_params->height, camera_params->frame_rate, writer->video_file.c_str(), writer->keyframe_file.c_str());
    }
    writer->metadata = new std::ofstream();
    writer->metadata->open(writer->metadata_file.c_str());
     if (!(*writer->metadata))
    {
        std::cout << "Metadata file did not open!";
        return;
    }
    *writer->metadata << "frame_id,timestamp,timestamp_sys\n";
}

static inline void write_metadata(std::ofstream *metadata, unsigned long long frame_id, unsigned long long timestamp, uint64_t timestamp_sys)
{
    *metadata << frame_id << "," << timestamp << "," << timestamp_sys << std::endl;
}


static inline void close_writer(EncoderContext *encoder, Writer *writer)
{
    if(encoder->pEnc) {
        encoder->pEnc->EndEncode(encoder->vPacket);
        for (std::vector<uint8_t> &packet : encoder->vPacket)
        {
            writer->video->push_packet(packet.data(), (int)packet.size(), encoder->num_frame_encode++);
        }
        encoder->pEnc->DestroyEncoder();
        delete encoder->pEnc;
        encoder->pEnc = nullptr;
    }

    if(writer->video) {
        writer->video->quit_thread();
        writer->video->join_thread();
        delete writer->video;
        writer->video = nullptr;
    }

    if(writer->metadata && writer->metadata->is_open()) {
        writer->metadata->close();
        delete writer->metadata;
        writer->metadata = nullptr;
    }
}

GPUVideoEncoder::GPUVideoEncoder(const char* name, CUcontext cuda_context, CameraParams *camera_params,
                                const std::string& codec, const std::string& preset, const std::string& tuning,
                                std::string folder_name, bool* encoder_ready_signal,
                                SafeQueue<WORKER_ENTRY*>& recycle_queue)
        : CThreadWorker<WORKER_ENTRY>(name),
        camera_params(camera_params),
        folder_name(folder_name),
        encoder_ready_signal(encoder_ready_signal),
        m_cuContext(cuda_context),
        m_recycle_queue(recycle_queue),
        m_stream(nullptr),
        d_rgb_temp_(nullptr),
        d_iyuv_temp_(nullptr),
        last_fps_update_time_(std::chrono::steady_clock::now()),
        frame_counter_(0),
        current_fps_(0.0),
        scaled_width_(3840),
        scaled_height_(3840),
        d_scaled_mono_buffer_(nullptr)
{
    std::cout << "[GPUVideoEncoder] Constructor for " << name << " on GPU " << camera_params->gpu_id << std::endl;

    ck(cuCtxPushCurrent(cuda_context));

    try
    {
        ck(cudaSetDevice(camera_params->gpu_id));
        ck(cudaStreamCreate(&m_stream)); // Create dedicated stream

        initalize_gpu_frame(&frame_original, camera_params);
        initialize_gpu_debayer(&debayer, camera_params);

        ck(cudaMalloc(&d_scaled_mono_buffer_, scaled_width_ * scaled_height_));
        ck(cudaMalloc(&d_rgb_temp_, scaled_width_ * scaled_height_ * 3));
        ck(cudaMalloc(&d_iyuv_temp_, (size_t)scaled_width_ * scaled_height_ * 3 / 2));

        encoder.cuContext = m_cuContext;
        encoder.eFormat = NV_ENC_BUFFER_FORMAT_IYUV;
        encoder.pEnc = new NvEncoderCuda(encoder.cuContext, scaled_width_, scaled_height_, encoder.eFormat, 0, false, false);

        NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
        NV_ENC_CONFIG encodeConfig = {NV_ENC_CONFIG_VER};
        initializeParams.encodeConfig = &encodeConfig;

        GUID codecGuid = (codec == "hevc") ? NV_ENC_CODEC_HEVC_GUID : NV_ENC_CODEC_H264_GUID;

        GUID presetGuid = NV_ENC_PRESET_P3_GUID; // Default
        if (preset == "p1") presetGuid = NV_ENC_PRESET_P1_GUID;
        else if (preset == "p5") presetGuid = NV_ENC_PRESET_P5_GUID;
        else if (preset == "p7") presetGuid = NV_ENC_PRESET_P7_GUID;

        NV_ENC_TUNING_INFO tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY; // Default
        if (tuning == "ll") tuningInfo = NV_ENC_TUNING_INFO_LOW_LATENCY;
        else if (tuning == "ull") tuningInfo = NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;
        else if (tuning == "lossless") tuningInfo = NV_ENC_TUNING_INFO_LOSSLESS;

        encoder.pEnc->CreateDefaultEncoderParams(&initializeParams, codecGuid, presetGuid, tuningInfo);

        initializeParams.encodeWidth = scaled_width_;
        initializeParams.encodeHeight = scaled_height_;
        initializeParams.frameRateNum = camera_params->frame_rate;
        initializeParams.frameRateDen = 1;
        initializeParams.enablePTD = 1;

        if (tuningInfo == NV_ENC_TUNING_INFO_LOW_LATENCY || tuningInfo == NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY)
        {
            encodeConfig.gopLength = NVENC_INFINITE_GOPLENGTH;
            encodeConfig.frameIntervalP = 1;
            encodeConfig.rcParams.lowDelayKeyFrameScale = 1;
        }

        encodeConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
        encodeConfig.rcParams.averageBitRate = 20000000;
        encodeConfig.rcParams.maxBitRate = 25000000;
        encodeConfig.rcParams.vbvBufferSize = encodeConfig.rcParams.averageBitRate;

        encoder.pEnc->CreateEncoder(&initializeParams);
        encoder.pEnc->SetIOCudaStreams((NV_ENC_CUSTREAM_PTR)&m_stream, (NV_ENC_CUSTREAM_PTR)&m_stream);

        initialize_writer(&writer, camera_params, folder_name, codec);
        writer.video->create_thread();
        *encoder_ready_signal = true;
    }
    catch (...)
    {
        std::cerr << "[GPUVideoEncoder] Error initializing encoder for " << name << std::endl;
        CUcontext popped_context;
        cuCtxPopCurrent(&popped_context);
    }
    CUcontext popped_context;
    ck(cuCtxPopCurrent(&popped_context));
}

GPUVideoEncoder::~GPUVideoEncoder()
{
    std::cout << "[GPUVideoEncoder] Destructor for " << this->threadName << std::endl;
    ck(cuCtxPushCurrent(m_cuContext));

    close_writer(&encoder, &writer);
    ck(cudaSetDevice(camera_params->gpu_id));
    if (m_stream) { cudaStreamDestroy(m_stream); }
    cudaFree(frame_original.d_orig);
    cudaFree(debayer.d_debayer);
    if (d_rgb_temp_) cudaFree(d_rgb_temp_);
    if (d_iyuv_temp_) cudaFree(d_iyuv_temp_);
    if (d_scaled_mono_buffer_) cudaFree(d_scaled_mono_buffer_);

    CUcontext popped_context;
    ck(cuCtxPopCurrent(&popped_context));
}

bool GPUVideoEncoder::WorkerFunction(WORKER_ENTRY* entry)
{
    if (!entry) return false;

    ck(cuCtxPushCurrent(m_cuContext));
    ck(cudaSetDevice(camera_params->gpu_id));
    nppSetStream(m_stream); // <--- FIX: Associate NPP calls with our stream

    const int width = camera_params->width;
    const int height = camera_params->height;

    frame_counter_++;
    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = now - last_fps_update_time_;
    if (elapsed.count() >= 1.0) {
        current_fps_ = frame_counter_ / elapsed.count();
        std::cout << "[" << this->threadName << "] Encoding FPS: " << current_fps_
                  << " (Queue depth: " << this->GetCountQueueInSize() << ")" << std::endl;
        frame_counter_ = 0;
        last_fps_update_time_ = now;
    }

    ck(cudaMemcpyAsync(frame_original.d_orig, entry->d_image, (size_t)width * height, cudaMemcpyDeviceToDevice, m_stream));

    NppiSize oSrcSize = { width, height };
    NppiRect oSrcRect = { 0, 0, width, height };
    NppiSize oDstSize = { scaled_width_, scaled_height_ };
    NppiRect oDstRect = { 0, 0, scaled_width_, scaled_height_ };

    if (camera_params->color){
        // <--- FIX: Use standard NPP functions, not _Ctx versions --->
        nppiResize_8u_C1R(frame_original.d_orig, width, oSrcSize, oSrcRect, d_scaled_mono_buffer_, scaled_width_, oDstSize, oDstRect, NPPI_INTER_LANCZOS);
        nppiCFAToRGB_8u_C1C3R(d_scaled_mono_buffer_, scaled_width_, oDstSize, oDstRect, d_rgb_temp_, scaled_width_ * 3, debayer.grid, NPPI_INTER_UNDEFINED);

        Npp8u* pYuvDst[] = { d_iyuv_temp_, d_iyuv_temp_ + (size_t)scaled_width_ * scaled_height_, d_iyuv_temp_ + ((size_t)scaled_width_ * scaled_height_ * 5 / 4) };
        int rYuvDstStep[] = { scaled_width_, scaled_width_ / 2, scaled_width_ / 2 };

        nppiRGBToYUV420_8u_C3P3R(d_rgb_temp_, scaled_width_ * 3, pYuvDst, rYuvDstStep, oDstSize);
    } else {
        // <--- FIX: Use standard NPP functions, not _Ctx versions --->
        nppiResize_8u_C1R(frame_original.d_orig, width, oSrcSize, oSrcRect, d_scaled_mono_buffer_, scaled_width_, oDstSize, oDstRect, NPPI_INTER_LANCZOS);

        unsigned char* d_y_plane_dst = d_iyuv_temp_;
        ck(cudaMemcpyAsync(d_y_plane_dst, d_scaled_mono_buffer_, (size_t)scaled_width_ * scaled_height_, cudaMemcpyDeviceToDevice, m_stream));

        unsigned char* d_u_plane_dst = d_iyuv_temp_ + ((size_t)scaled_width_ * scaled_height_);
        unsigned char* d_v_plane_dst = d_u_plane_dst + ((size_t)scaled_width_ * scaled_height_ / 4);
        ck(cudaMemsetAsync(d_u_plane_dst, 128, (size_t)scaled_width_ * scaled_height_ / 4, m_stream));
        ck(cudaMemsetAsync(d_v_plane_dst, 128, (size_t)scaled_width_ * scaled_height_ / 4, m_stream));
    }

    const NvEncInputFrame *encoderInputFrame = encoder.pEnc->GetNextInputFrame();
    NvEncoderCuda::CopyToDeviceFrame(m_cuContext, d_iyuv_temp_, scaled_width_,
                                     (CUdeviceptr)encoderInputFrame->inputPtr, encoderInputFrame->pitch,
                                     encoder.pEnc->GetEncodeWidth(), encoder.pEnc->GetEncodeHeight(),
                                     CU_MEMORYTYPE_DEVICE, encoderInputFrame->bufferFormat,
                                     encoderInputFrame->chromaOffsets, encoderInputFrame->numChromaPlanes, false, (CUstream)m_stream);

    encoder.pEnc->EncodeFrame(encoder.vPacket);

    for (std::vector<uint8_t> &packet : encoder.vPacket)
    {
        writer.video->push_packet(packet.data(), (int)packet.size(), encoder.num_frame_encode++);
    }
    write_metadata(writer.metadata, entry->frame_id, entry->timestamp, entry->timestamp_sys);

    ck(cudaStreamSynchronize(m_stream));

    if (entry->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        m_recycle_queue.push(entry);
    }

    CUcontext popped_context;
    ck(cuCtxPopCurrent(&popped_context));

    return false;
}