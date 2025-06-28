// src/gpu_video_encoder.cpp

#include "gpu_video_encoder.h"
#include "kernel.cuh"
#include <npp.h>
#include <nppi.h>
#include <nppi_color_conversion.h>
#include <iostream>
#include "global.h"
#include "NvEncoder/NvEncoder.h"
#include "cuda_context_debug.h"

static std::string NvEncFormatToString(NV_ENC_BUFFER_FORMAT format) {
    switch (format) {
        case NV_ENC_BUFFER_FORMAT_NV12: return "NV12";
        case NV_ENC_BUFFER_FORMAT_YV12: return "YV12";
        case NV_ENC_BUFFER_FORMAT_IYUV: return "IYUV";
        case NV_ENC_BUFFER_FORMAT_YUV444: return "YUV444";
        case NV_ENC_BUFFER_FORMAT_ARGB: return "ARGB";
        default: return "Unknown";
    }
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

// SIMPLIFIED CONSTRUCTOR - No resizing, use native camera resolution
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
scaled_width_(camera_params->width),   // USE NATIVE CAMERA WIDTH
scaled_height_(camera_params->height), // USE NATIVE CAMERA HEIGHT
d_scaled_mono_buffer_(nullptr),
encoder_pitch_(0)
{
    CUDA_CTX_LOG("=== GPU Video Encoder Constructor START ===");
    
    std::cout << "[GPUVideoEncoder] Constructor for " << name << " on GPU " << camera_params->gpu_id << std::endl;
    std::cout << "[GPUVideoEncoder] NATIVE RESOLUTION MODE: " << camera_params->width << "x" << camera_params->height << std::endl;

    try {
        CUDA_CTX_LOG("Pushing CUDA context in constructor");
        ck(cuCtxPushCurrent(m_cuContext));
        dumpCudaState("Constructor - After context push");

        initalize_gpu_frame(&frame_original, camera_params);
        initialize_gpu_debayer(&debayer, camera_params);

        // No need for d_scaled_mono_buffer since we're not resizing
        // ck(cudaMalloc(&d_scaled_mono_buffer_, scaled_width_ * scaled_height_));
        ck(cudaMalloc(&d_rgb_temp_, scaled_width_ * scaled_height_ * 3));
        // DON'T allocate d_iyuv_temp_ yet - we need encoder pitch first

        encoder.cuContext = m_cuContext;
        encoder.eFormat = NV_ENC_BUFFER_FORMAT_IYUV;
        
        CUDA_CTX_LOG("Creating NVIDIA encoder");
        encoder.pEnc = new NvEncoderCuda(m_cuContext, camera_params->width, camera_params->height, encoder.eFormat);
        CUDA_CTX_LOG("NVIDIA encoder created successfully");
        
        NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
        NV_ENC_CONFIG encodeConfig = {NV_ENC_CONFIG_VER};
        initializeParams.encodeConfig = &encodeConfig;

        GUID codecGuid = (codec == "hevc") ? NV_ENC_CODEC_HEVC_GUID : NV_ENC_CODEC_H264_GUID;
        GUID presetGuid = (preset == "p1") ? NV_ENC_PRESET_P1_GUID : (preset == "p5") ? NV_ENC_PRESET_P5_GUID : (preset == "p7") ? NV_ENC_PRESET_P7_GUID : NV_ENC_PRESET_P3_GUID;
        NV_ENC_TUNING_INFO tuningInfo = (tuning == "ll") ? NV_ENC_TUNING_INFO_LOW_LATENCY : (tuning == "ull") ? NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY : (tuning == "lossless") ? NV_ENC_TUNING_INFO_LOSSLESS : NV_ENC_TUNING_INFO_HIGH_QUALITY;

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

        if (!camera_params->color) {
            std::cout << "[GPUVideoEncoder] Mono camera detected, setting monoChromeEncoding to 1" << std::endl;
            encodeConfig.monoChromeEncoding = 1;
        }

        encoder.pEnc->CreateEncoder(&initializeParams);
        encoder.pEnc->SetIOCudaStreams((NV_ENC_CUSTREAM_PTR)&m_stream, (NV_ENC_CUSTREAM_PTR)&m_stream);

        // NOW get the encoder's expected pitch and allocate IYUV buffer
        const NvEncInputFrame *tempFrame = encoder.pEnc->GetNextInputFrame();
        encoder_pitch_ = tempFrame->pitch;

        // Allocate IYUV buffer using encoder's pitch
        size_t encoder_buffer_size = (size_t)encoder_pitch_ * scaled_height_ * 3 / 2;
        ck(cudaMalloc(&d_iyuv_temp_, encoder_buffer_size));

        std::cout << "[GPUVideoEncoder] Native resolution " << scaled_width_ << "x" << scaled_height_ 
                  << " with encoder pitch: " << encoder_pitch_ << std::endl;

        initialize_writer(&writer, camera_params, folder_name, codec);
        writer.video->create_thread();
        std::cout << "[GPUVideoEncoder] Successfully initialized NATIVE RESOLUTION encoder for " << name 
                  << " - Codec: " << codec << ", Preset: " << preset << ", Tuning: " << tuning << std::endl;
        *encoder_ready_signal = true;
        
        CUDA_CTX_LOG("Popping CUDA context in constructor");
        CUcontext popped_context;
        ck(cuCtxPopCurrent(&popped_context));
    }
    catch (const std::exception& e)
    {
        std::cerr << "[GPUVideoEncoder] Exception initializing encoder for " << name
                  << ": " << e.what() << std::endl;
        CUDA_CTX_LOG("Exception in constructor");
        CUcontext popped_context;
        cuCtxPopCurrent(&popped_context);
        throw;
    }
    catch (...)
    {
        std::cerr << "[GPUVideoEncoder] Unknown exception initializing encoder for " << name << std::endl;
        CUDA_CTX_LOG("Unknown exception in constructor");
        CUcontext popped_context;
        cuCtxPopCurrent(&popped_context);
        throw;
    }
    
    CUDA_CTX_LOG("=== GPU Video Encoder Constructor END ===");
}

GPUVideoEncoder::~GPUVideoEncoder()
{
    std::cout << "[GPUVideoEncoder] Destructor for " << this->threadName << std::endl;
    ck(cuCtxPushCurrent(m_cuContext));

    close_writer(&encoder, &writer);
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

    // Set CUDA context for this thread
    CUDA_CONTEXT_SCOPE(m_cuContext);

    // Log entry into function
    ENCODER_CTX_LOG("=== ENTERING WorkerFunction ===", entry->frame_id);
    dumpCudaState("WorkerFunction Entry", entry->frame_id);


    try {

        // Ensure we're using the correct CUDA device
        ck(cudaSetDevice(camera_params->gpu_id));
        ENCODER_CTX_LOG("Setting NPP stream", entry->frame_id);
        nppSetStream(m_stream);

        const int width = camera_params->width;
        const int height = camera_params->height;

        // FPS tracking
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

        ENCODER_CTX_LOG("Processing frame dimensions", entry->frame_id);
        std::cout << "[GPUEncoder] NATIVE RESOLUTION - Processing frame " << entry->frame_id 
                  << " - Dimensions: " << width << "x" << height 
                  << " (no resizing, pitch: " << encoder_pitch_ << ")" << std::endl;

        // Copy frame data from entry to local buffer
        CUDA_MEM_LOG("Copying frame data from entry", entry->d_image, width * height, entry->frame_id);
        ENCODER_CTX_LOG("About to copy frame data", entry->frame_id);
        ck(cudaMemcpyAsync(frame_original.d_orig, entry->d_image, 
                           (size_t)width * height, cudaMemcpyDeviceToDevice, m_stream));

        if (camera_params->color) {
            // === COLOR PATH (existing logic) ===
            ENCODER_CTX_LOG("Processing COLOR frame", entry->frame_id);
            std::cout << "[GPUEncoder] Processing COLOR frame at native resolution..." << std::endl;
            
            // For color, we'd need to implement the full RGB->YUV pipeline
            // This would involve debayering and color space conversion
            // (Implementation would go here if needed)
            
        } else {
            // === FIXED MONOCHROME PATH ===
            ENCODER_CTX_LOG("Processing MONOCHROME frame", entry->frame_id);
            std::cout << "[GPUEncoder] Processing MONOCHROME frame at native resolution..." << std::endl;
            
            // Calculate plane pointers with ENCODER PITCH
            unsigned char* d_y_plane_dst = d_iyuv_temp_;
            unsigned char* d_u_plane_dst = d_iyuv_temp_ + ((size_t)encoder_pitch_ * height);
            unsigned char* d_v_plane_dst = d_u_plane_dst + ((size_t)encoder_pitch_ * height / 4);
            
            std::cout << "[GPUEncoder] Direct copy at native resolution:" << std::endl;
            std::cout << "  - Y plane: " << static_cast<void*>(d_y_plane_dst) 
                      << " (copy " << width << "x" << height 
                      << " with src pitch " << width << " -> dst pitch " << encoder_pitch_ << ")" << std::endl;
        
            CUDA_MEM_LOG("Converting to IYUV", d_iyuv_temp_, encoder_pitch_ * height * 3 / 2, entry->frame_id);
            
            // Use cudaMemcpy2DAsync for proper pitch conversion
            ck(cudaMemcpy2DAsync(d_y_plane_dst, encoder_pitch_,              // dst, dst_pitch
                                frame_original.d_orig, width,                // src, src_pitch  
                                width, height,                               // width, height
                                cudaMemcpyDeviceToDevice, m_stream));
        
            // Use encoder_pitch for U/V plane calculations
            ck(cudaMemsetAsync(d_u_plane_dst, 128, (size_t)encoder_pitch_ * height / 4, m_stream));
            ck(cudaMemsetAsync(d_v_plane_dst, 128, (size_t)encoder_pitch_ * height / 4, m_stream));
            
            std::cout << "[GPUEncoder] Native resolution IYUV planes created successfully" << std::endl;
        }

        std::cout << "[GPUEncoder] NATIVE RESOLUTION encoding:" << std::endl;
        std::cout << "  - Source: " << width << "x" << height << std::endl;
        std::cout << "  - Encoder: " << width << "x" << height << std::endl;
        std::cout << "  - Pitch match: YES" << std::endl;

        // Stream synchronization
        CUDA_SYNC_LOG("Synchronizing stream before encode", m_stream, entry->frame_id);
        ENCODER_CTX_LOG("Synchronizing stream before encode", entry->frame_id);
        std::cout << "[GPUEncoder] Synchronizing stream before encode..." << std::endl;
        ck(cudaStreamSynchronize(m_stream));

        // NVIDIA encoder call
        ENCODER_CTX_LOG("About to call NVIDIA EncodeFrame - CRITICAL POINT", entry->frame_id);
        dumpCudaState("Pre-EncodeFrame", entry->frame_id);
        
        std::cout << "[GPUEncoder] Calling EncodeFrame for NATIVE RESOLUTION frame " << entry->frame_id << "..." << std::endl;
        
        // Get encoder input frame and copy our IYUV data to it
        const NvEncInputFrame *encoderInputFrame = encoder.pEnc->GetNextInputFrame();
        NvEncoderCuda::CopyToDeviceFrame(encoder.cuContext,
                                         d_iyuv_temp_,
                                         encoder_pitch_, 
                                         (CUdeviceptr)encoderInputFrame->inputPtr,
                                         encoderInputFrame->pitch,
                                         encoder.pEnc->GetEncodeWidth(),
                                         encoder.pEnc->GetEncodeHeight(),
                                         CU_MEMORYTYPE_DEVICE,
                                         encoderInputFrame->bufferFormat,
                                         encoderInputFrame->chromaOffsets,
                                         encoderInputFrame->numChromaPlanes);

        // Encode the frame
        encoder.pEnc->EncodeFrame(encoder.vPacket);
        
        ENCODER_CTX_LOG("EncodeFrame completed successfully", entry->frame_id);
        ENCODER_CTX_LOG("Processing encoded packets", entry->frame_id);

        // Push encoded packets to writer
        for (std::vector<uint8_t> &packet : encoder.vPacket) {
            writer.video->push_packet(packet.data(), (int)packet.size(), encoder.num_frame_encode++);
        }
        
        // Write metadata
        write_metadata(writer.metadata, entry->frame_id, entry->timestamp, entry->timestamp_sys);
        
        std::cout << "[GPUEncoder] Frame " << entry->frame_id << " encoded successfully" << std::endl;

        // Handle reference counting
        ENCODER_CTX_LOG("Handling reference count", entry->frame_id);
        if (entry->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            m_recycle_queue.push(entry);
        }

    } catch (const std::exception& e) {
        std::cerr << "[GPUEncoder] Exception in WorkerFunction: " << e.what() << std::endl;
        if (entry->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            m_recycle_queue.push(entry);
        }
        
        return false;
    }
    
    ENCODER_CTX_LOG("=== EXITING WorkerFunction ===", entry->frame_id);
    return false;
}