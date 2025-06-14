// src/gpu_video_encoder.cpp

#include "gpu_video_encoder.h"
#include "kernel.cuh"
#include <npp.h> // Make sure this is included for NPP functions
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

// Helper to initialize the FFmpeg-based file writer (no changes needed)
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
        // Default to H264 if something goes wrong
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

// static inline void encode_frame(EncoderContext *encoder, Writer *writer, 
//     unsigned char* d_iyuv_frame, int width, int height, 
//     unsigned long long frame_pts)
// {
//     const NvEncInputFrame *encoderInputFrame = encoder->pEnc->GetNextInputFrame();

//     // Our camera data: Y=4512 bytes/row, U=2256 bytes/row, V=2256 bytes/row
//     // Encoder expects: Y=4608 bytes/row (aligned), U=2304 bytes/row, V=2304 bytes/row

//     unsigned char* src_y_plane = d_iyuv_frame;
//     unsigned char* src_u_plane = d_iyuv_frame + (width * height);
//     unsigned char* src_v_plane = src_u_plane + (width * height / 4);

//     unsigned char* dst_y_plane = (unsigned char*)encoderInputFrame->inputPtr;
//     unsigned char* dst_u_plane = dst_y_plane + (encoderInputFrame->pitch * height);
//     unsigned char* dst_v_plane = dst_u_plane + ((encoderInputFrame->pitch / 2) * (height / 2));

//     // Copy Y plane row by row (4512 → 4608 pitch conversion)
//     ck(cudaMemcpy2D(dst_y_plane, encoderInputFrame->pitch,          // dst, dst_pitch
//         src_y_plane, width,                             // src, src_pitch  
//         width, height,                                  // width, height
//         cudaMemcpyDeviceToDevice));

//     // Copy U plane row by row
//     int uv_width = width / 2;
//     int uv_height = height / 2;
//     int encoder_uv_pitch = encoderInputFrame->pitch / 2;

//     ck(cudaMemcpy2D(dst_u_plane, encoder_uv_pitch,
//         src_u_plane, uv_width,
//         uv_width, uv_height,
//         cudaMemcpyDeviceToDevice));

//     // Fast 2D copy for V plane
//     ck(cudaMemcpy2D(dst_v_plane, encoder_uv_pitch,
//             src_v_plane, uv_width,
//             uv_width, uv_height,
//             cudaMemcpyDeviceToDevice));
//     encoder->pEnc->EncodeFrame(encoder->vPacket);

//     for (std::vector<uint8_t> &packet : encoder->vPacket)
//     {
//         writer->video->push_packet(packet.data(), (int)packet.size(), frame_pts);
//     }
// }

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
        
            ck(cudaSetDevice(camera_params->gpu_id));
            const size_t original_width = camera_params->width;
            const size_t original_height = camera_params->height;
        
            // Allocate buffer for the original full-resolution frame
            initalize_gpu_frame(&frame_original, camera_params);
            initialize_gpu_debayer(&debayer, camera_params);
        
            // --- ALLOCATE NEW DOWNSCALING AND CONVERSION BUFFERS ---
            // Buffer for the smaller, resized monochrome frame
            ck(cudaMalloc(&d_scaled_mono_buffer_, scaled_width_ * scaled_height_));
            // YUV and RGB buffers are now sized for the SMALLER frame
            ck(cudaMalloc(&d_rgb_temp_, scaled_width_ * scaled_height_ * 3));
            ck(cudaMalloc(&d_iyuv_temp_, scaled_width_ * scaled_height_ * 3 / 2));
            // --- END ALLOCATE ---
        
        
            encoder.cuContext = cuda_context;

            encoder.eFormat = NV_ENC_BUFFER_FORMAT_IYUV;
            
            encoder.pEnc = new NvEncoderCuda(encoder.cuContext, scaled_width_, scaled_height_, encoder.eFormat, 0, false, false);
        
            NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
            NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };
            initializeParams.encodeConfig = &encodeConfig;
        
            // --- NEW: Select GUIDs and Tuning Info based on string parameters ---
            GUID codecGuid = (codec == "hevc") ? NV_ENC_CODEC_HEVC_GUID : NV_ENC_CODEC_H264_GUID;
            
            GUID presetGuid = NV_ENC_PRESET_P3_GUID; // Default
            if (preset == "p1") presetGuid = NV_ENC_PRESET_P1_GUID;
            else if (preset == "p5") presetGuid = NV_ENC_PRESET_P5_GUID;
            else if (preset == "p7") presetGuid = NV_ENC_PRESET_P7_GUID;

            NV_ENC_TUNING_INFO tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY; // Default
            if (tuning == "ll") tuningInfo = NV_ENC_TUNING_INFO_LOW_LATENCY;
            else if (tuning == "ull") tuningInfo = NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;
            else if (tuning == "lossless") tuningInfo = NV_ENC_TUNING_INFO_LOSSLESS;

            // 1. Start with the selected preset and tuning info. This is the key change!
            encoder.pEnc->CreateDefaultEncoderParams(&initializeParams, codecGuid, presetGuid, tuningInfo);

            // 2. Override specific parameters for our exact needs.
            initializeParams.encodeWidth = scaled_width_;
            initializeParams.encodeHeight = scaled_height_;
            initializeParams.frameRateNum = camera_params->frame_rate;
            initializeParams.frameRateDen = 1;
            initializeParams.enablePTD = 1;
        
            // 3. Fine-tune the configuration for low latency and quality.
            if (tuningInfo == NV_ENC_TUNING_INFO_LOW_LATENCY || tuningInfo == NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY)
            {
                encodeConfig.gopLength = NVENC_INFINITE_GOPLENGTH;
                encodeConfig.frameIntervalP = 1; // No B-Frames for low latency
                encodeConfig.rcParams.lowDelayKeyFrameScale = 1; 
            }
                // This is the C++ equivalent of "-split_enable 2". It forces the driver
            // to use both NVENC engines on the A6000 for a single large frame.
            // if (codecGuid == NV_ENC_CODEC_HEVC_GUID)
            // {
            //     encodeConfig.encodeCodecConfig.hevcConfig.sliceMode = 1; 
            //     encodeConfig.encodeCodecConfig.hevcConfig.sliceModeData = 2; // Use 2 slices for 2 engines
            // }
            
            encodeConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
            encodeConfig.rcParams.averageBitRate = 20000000; // 40 Mbps
            encodeConfig.rcParams.maxBitRate = 25000000;     // 50 Mbps
            encodeConfig.rcParams.vbvBufferSize = encodeConfig.rcParams.averageBitRate;
            
            // 4. Create the encoder with our customized parameters.
            encoder.pEnc->CreateEncoder(&initializeParams);
        
            // This is a key setting for low latency that the preset enables.
            encodeConfig.rcParams.lowDelayKeyFrameScale = 1; 
            
            // 5. Create the encoder with our customized parameters.
            encoder.pEnc->CreateEncoder(&initializeParams);
        
            initialize_writer(&writer, camera_params, folder_name, codec); // Pass codec to FFmpegWriter
            writer.video->create_thread();
            *encoder_ready_signal = true;
        }

GPUVideoEncoder::~GPUVideoEncoder()
{
    std::cout << "[GPUVideoEncoder] Destructor for " << this->threadName << std::endl;
    close_writer(&encoder, &writer);
    ck(cudaSetDevice(camera_params->gpu_id));
    cudaFree(frame_original.d_orig);
    cudaFree(debayer.d_debayer);
    if (d_rgb_temp_) cudaFree(d_rgb_temp_);
    if (d_iyuv_temp_) cudaFree(d_iyuv_temp_);
    if (d_scaled_mono_buffer_) cudaFree(d_scaled_mono_buffer_);
    cuCtxDestroy(encoder.cuContext);
}

void GPUVideoEncoder::TestPattern(unsigned char* d_iyuv, int width, int height) {
    // Create a test pattern: red color in YUV
    size_t y_size = width * height;
    size_t uv_size = width * height / 4;
    
    // For pure red in YUV: Y=76, U=84, V=255
    cudaMemset(d_iyuv, 76, y_size);                    // Y plane
    cudaMemset(d_iyuv + y_size, 84, uv_size);         // U plane  
    cudaMemset(d_iyuv + y_size + uv_size, 255, uv_size); // V plane
    
    std::cout << "[TestPattern] Filled buffer with red color (Y=76, U=84, V=255)" << std::endl;
}

bool GPUVideoEncoder::WorkerFunction(WORKER_ENTRY* entry)
{
    if (!entry) return false;

    ck(cudaSetDevice(camera_params->gpu_id));

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

        // 1. Copy the full-res frame from the entry buffer to our temporary buffer
        ck(cudaMemcpy(frame_original.d_orig, entry->d_image, camera_params->width * camera_params->height, cudaMemcpyDeviceToDevice));

        // 2. Downscale the full-res mono frame to our target encode size using NPP
        NppiSize oSrcSize = { (int)camera_params->width, (int)camera_params->height };
        NppiRect oSrcRect = { 0, 0, (int)camera_params->width, (int)camera_params->height };
        NppiSize oDstSize = { scaled_width_, scaled_height_ };
        NppiRect oDstRect = { 0, 0, scaled_width_, scaled_height_ };
    
        // nppiResize_8u_C1R is perfect for resizing a single-channel (monochrome) 8-bit image
        NppStatus nppStatResize = nppiResize_8u_C1R(
            frame_original.d_orig,        // Source: full-res mono image
            camera_params->width,         // Source pitch
            oSrcSize,
            oSrcRect,
            d_scaled_mono_buffer_,        // Destination: our new scaled buffer
            scaled_width_,                // Destination pitch
            oDstSize,
            oDstRect,
            NPPI_INTER_LANCZOS            // High-quality downsampling filter
        );
    
        if (nppStatResize != NPP_SUCCESS) {
            std::cerr << "Error: NPP Resize failed with status " << nppStatResize << std::endl;
            if (entry->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) { m_recycle_queue.push(entry); }
            return false;
        }    

    if (camera_params->color){
        debayer_frame_gpu(camera_params, &frame_original, &debayer);
        rgba2rgb_convert(d_rgb_temp_, debayer.d_debayer, width, height, 0);

        const Npp8u* pRgbSrc = d_rgb_temp_;
        int nRgbSrcStep = width * 3;
        Npp8u* pYuvDst[] = {
            d_iyuv_temp_,
            d_iyuv_temp_ + (width * height),
            d_iyuv_temp_ + (width * height) + (width * height / 4)
        };
        int rYuvDstStep[] = { width, width / 2, width / 2 };
        NppiSize oSizeROI = { width, height };

        NppStatus nppStat = nppiRGBToYUV420_8u_C3P3R(pRgbSrc, nRgbSrcStep, pYuvDst, rYuvDstStep, oSizeROI);
        if (nppStat != NPP_SUCCESS) {
            std::cerr << "Error: NPP RGBToYUV420 (3-plane) conversion failed with status " << nppStat << std::endl;
            // FIX: Removed the incorrect recycle call. The ref-counter at the end will handle it.
            return false;
        }
    } else {
        // MONOCHROME PATH - now uses the scaled buffer as its source
        unsigned char* d_y_plane_dst = d_iyuv_temp_;
        ck(cudaMemcpy(d_y_plane_dst, d_scaled_mono_buffer_, scaled_width_ * scaled_height_, cudaMemcpyDeviceToDevice));
        
        unsigned char* d_u_plane_dst = d_iyuv_temp_ + (scaled_width_ * scaled_height_);
        unsigned char* d_v_plane_dst = d_u_plane_dst + (scaled_width_ * scaled_height_ / 4);
        ck(cudaMemset(d_u_plane_dst, 128, scaled_width_ * scaled_height_ / 4));
        ck(cudaMemset(d_v_plane_dst, 128, scaled_width_ * scaled_height_ / 4));
    }

    // The optimized copy to the encoder surface is now the only one.
    const NvEncInputFrame *encoderInputFrame = encoder.pEnc->GetNextInputFrame();
    NvEncoderCuda::CopyToDeviceFrame(encoder.cuContext,
                                     d_iyuv_temp_,
                                     scaled_width_, // Use scaled width for pitch
                                     (CUdeviceptr)encoderInputFrame->inputPtr,
                                     encoderInputFrame->pitch,
                                     encoder.pEnc->GetEncodeWidth(),
                                     encoder.pEnc->GetEncodeHeight(),
                                     CU_MEMORYTYPE_DEVICE,
                                     encoderInputFrame->bufferFormat,
                                     encoderInputFrame->chromaOffsets,
                                     encoderInputFrame->numChromaPlanes);

    // Encode the frame, this populates encoder.vPacket
    encoder.pEnc->EncodeFrame(encoder.vPacket);

    // This loop takes the encoded packets and sends them to the FFmpegWriter thread.
    for (std::vector<uint8_t> &packet : encoder.vPacket)
    {
        writer.video->push_packet(packet.data(), (int)packet.size(), encoder.num_frame_encode++);
    }
    
    write_metadata(writer.metadata, entry->frame_id, entry->timestamp, entry->timestamp_sys);
    
    // Correct, single point of recycling
    if (entry->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        m_recycle_queue.push(entry);
    }

    return false;
}


// Additional debugging function to dump a frame for analysis
void GPUVideoEncoder::DumpYUVFrame(const char* filename, unsigned char* d_yuv_data, int width, int height)
{
    size_t yuv_size = width * height * 3 / 2;
    unsigned char* h_yuv_data = new unsigned char[yuv_size];
    cudaMemcpy(h_yuv_data, d_yuv_data, yuv_size, cudaMemcpyDeviceToHost);
    
    FILE* fp = fopen(filename, "wb");
    if (fp) {
        fwrite(h_yuv_data, 1, yuv_size, fp);
        fclose(fp);
        std::cout << "[Debug] Dumped YUV frame to " << filename << std::endl;
    }
    
    delete[] h_yuv_data;
}