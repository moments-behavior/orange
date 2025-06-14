// src/gpu_video_encoder.cpp

#include "gpu_video_encoder.h"
#include "kernel.cuh"
#include <npp.h> 
#include <nppi.h>
#include <nppi_color_conversion.h> 
#include <iostream>

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

        initalize_gpu_frame(&frame_original, camera_params);
        initialize_gpu_debayer(&debayer, camera_params);

        ck(cudaMalloc(&d_scaled_mono_buffer_, scaled_width_ * scaled_height_));
        // --- START FIX ---
        // Allocate d_rgb_temp_ to hold the SCALED RGB data, not the full size.
        // The full-size debayered data will go into `debayer.d_debayer` first.
        ck(cudaMalloc(&d_rgb_temp_, scaled_width_ * scaled_height_ * 3));
        // --- END FIX ---
        ck(cudaMalloc(&d_iyuv_temp_, scaled_width_ * scaled_height_ * 3 / 2));

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

    // --- Common setup for both color and mono ---
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

    // --- START FIX ---
    // The processing pipeline is now re-ordered to be safe for both color and mono.
    
    // 1. Copy the full-resolution frame from the entry buffer to our local buffer
    ck(cudaMemcpy(frame_original.d_orig, entry->d_image, width * height, cudaMemcpyDeviceToDevice));
    
    // 2. Prepare for downscaling
    NppiSize oSrcSize = { width, height };
    NppiRect oSrcRect = { 0, 0, width, height };
    NppiSize oDstSize = { scaled_width_, scaled_height_ };
    NppiRect oDstRect = { 0, 0, scaled_width_, scaled_height_ };

    if (camera_params->color){
        // --- COLOR PATH ---
        // 2a. Debayer the full-size mono image to a full-size RGBA buffer
        debayer_frame_gpu(camera_params, &frame_original, &debayer);

        // 2b. Resize the full-size RGBA image down to the scaled RGBA buffer
        NppStatus nppStatResize = nppiResize_8u_AC4R(
            debayer.d_debayer,        // Source: full-res RGBA image from debayer
            width * 4,                // Source pitch
            oSrcSize,
            oSrcRect,
            d_rgb_temp_,              // Destination: correctly-sized SCALED buffer
            scaled_width_ * 4,        // Destination pitch
            oDstSize,
            oDstRect,
            NPPI_INTER_LANCZOS
        );
        if (nppStatResize != NPP_SUCCESS) {
            std::cerr << "Error: NPP Resize (Color) failed with status " << nppStatResize << std::endl;
            if (entry->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) { m_recycle_queue.push(entry); }
            return false;
        }

        // 2c. Convert the SCALED RGBA image to IYUV (YUV420p)
        const Npp8u* pRgbSrc = d_rgb_temp_;
        int nRgbSrcStep = scaled_width_ * 4;
        Npp8u* pYuvDst[] = {
            d_iyuv_temp_,
            d_iyuv_temp_ + (scaled_width_ * scaled_height_),
            d_iyuv_temp_ + (scaled_width_ * scaled_height_) + (scaled_width_ * scaled_height_ / 4)
        };
        int rYuvDstStep[] = { scaled_width_, scaled_width_ / 2, scaled_width_ / 2 };
        NppiSize oScaledSizeROI = { scaled_width_, scaled_height_ };

        NppStatus nppStat = nppiRGBToYUV420_8u_AC4P3R(pRgbSrc, nRgbSrcStep, pYuvDst, rYuvDstStep, oScaledSizeROI);
        if (nppStat != NPP_SUCCESS) {
            std::cerr << "Error: NPP RGBToYUV420 (3-plane) conversion failed with status " << nppStat << std::endl;
            if (entry->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) { m_recycle_queue.push(entry); }
            CUcontext popped_context; ck(cuCtxPopCurrent(&popped_context));
            return false;
        }
    } else {
        // --- MONOCHROME PATH ---
        // 2a. Downscale the full-res mono frame to our target encode size
        NppStatus nppStatResize = nppiResize_8u_C1R(
            frame_original.d_orig,
            width,
            oSrcSize,
            oSrcRect,
            d_scaled_mono_buffer_,
            scaled_width_,
            oDstSize,
            oDstRect,
            NPPI_INTER_LANCZOS
        );
        if (nppStatResize != NPP_SUCCESS) {
            std::cerr << "Error: NPP Resize (Mono) failed with status " << nppStatResize << std::endl;
            if (entry->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) { m_recycle_queue.push(entry); }
            return false;
        }    

        // 2b. Copy the scaled mono data to the Y-plane and set U/V planes to neutral gray (128)
        unsigned char* d_y_plane_dst = d_iyuv_temp_;
        ck(cudaMemcpy(d_y_plane_dst, d_scaled_mono_buffer_, scaled_width_ * scaled_height_, cudaMemcpyDeviceToDevice));
        
        unsigned char* d_u_plane_dst = d_iyuv_temp_ + (scaled_width_ * scaled_height_);
        unsigned char* d_v_plane_dst = d_u_plane_dst + (scaled_width_ * scaled_height_ / 4);
        ck(cudaMemset(d_u_plane_dst, 128, scaled_width_ * scaled_height_ / 4));
        ck(cudaMemset(d_v_plane_dst, 128, scaled_width_ * scaled_height_ / 4));
    }
    // --- END FIX ---

    // 3. Copy the final IYUV frame to the encoder's input surface
    const NvEncInputFrame *encoderInputFrame = encoder.pEnc->GetNextInputFrame();
    NvEncoderCuda::CopyToDeviceFrame(encoder.cuContext,
                                     d_iyuv_temp_,
                                     scaled_width_, 
                                     (CUdeviceptr)encoderInputFrame->inputPtr,
                                     encoderInputFrame->pitch,
                                     encoder.pEnc->GetEncodeWidth(),
                                     encoder.pEnc->GetEncodeHeight(),
                                     CU_MEMORYTYPE_DEVICE,
                                     encoderInputFrame->bufferFormat,
                                     encoderInputFrame->chromaOffsets,
                                     encoderInputFrame->numChromaPlanes);

    // 4. Encode the frame
    encoder.pEnc->EncodeFrame(encoder.vPacket);

    // 5. Push encoded packets to the writer thread and write metadata
    for (std::vector<uint8_t> &packet : encoder.vPacket)
    {
        writer.video->push_packet(packet.data(), (int)packet.size(), encoder.num_frame_encode++);
    }
    write_metadata(writer.metadata, entry->frame_id, entry->timestamp, entry->timestamp_sys);
    
    // 6. Decrement ref count and recycle the entry if we're the last owner
    if (entry->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        m_recycle_queue.push(entry);
    }
    
    CUcontext popped_context;
    ck(cuCtxPopCurrent(&popped_context));

    return false;
}
