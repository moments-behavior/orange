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
#include "nvtx_profiling.h"

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
    NVTX_RANGE("Initialize_Writer");
    
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
    NVTX_RANGE("Write_Metadata");
    *metadata << frame_id << "," << timestamp << "," << timestamp_sys << std::endl;
}

static inline void close_writer(EncoderContext *encoder, Writer *writer)
{
    NVTX_RANGE("Close_Writer");
    
    if(encoder->pEnc) {
        NVTX_RANGE_PUSH("End_Encode");
        encoder->pEnc->EndEncode(encoder->vPacket);
        NVTX_RANGE_POP();
        
        NVTX_RANGE_PUSH("Flush_Remaining_Packets");
        for (std::vector<uint8_t> &packet : encoder->vPacket)
        {
            writer->video->push_packet(packet.data(), (int)packet.size(), encoder->num_frame_encode++);
        }
        NVTX_RANGE_POP();
        
        NVTX_RANGE_PUSH("Destroy_Encoder");
        encoder->pEnc->DestroyEncoder();
        delete encoder->pEnc;
        encoder->pEnc = nullptr;
        NVTX_RANGE_POP();
    }

    if(writer->video) {
        NVTX_RANGE_PUSH("Close_Video_Writer");
        writer->video->quit_thread();
        writer->video->join_thread();
        delete writer->video;
        writer->video = nullptr;
        NVTX_RANGE_POP();
    }

    if(writer->metadata && writer->metadata->is_open()) {
        NVTX_RANGE_PUSH("Close_Metadata");
        writer->metadata->close();
        delete writer->metadata;
        writer->metadata = nullptr;
        NVTX_RANGE_POP();
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
d_uv_default_plane_(nullptr), // Initialize new member
last_fps_update_time_(std::chrono::steady_clock::now()),
frame_counter_(0),
current_fps_(0.0),
scaled_width_(camera_params->width),
scaled_height_(camera_params->height),
d_scaled_mono_buffer_(nullptr),
encoder_pitch_(0)
{
    NVTX_ENCODE("GPUVideoEncoder_Constructor");
    CUDA_CTX_LOG("=== GPU Video Encoder Constructor START ===");
    
    std::cout << "[GPUVideoEncoder] Constructor for " << name << " on GPU " << camera_params->gpu_id << std::endl;
    std::cout << "[GPUVideoEncoder] NATIVE RESOLUTION MODE: " << camera_params->width << "x" << camera_params->height << std::endl;

    try {
        NVTX_RANGE_PUSH("Setup_CUDA_Context");
        CUDA_CTX_LOG("Pushing CUDA context in constructor");
        ck(cuCtxPushCurrent(m_cuContext));
        dumpCudaState("Constructor - After context push");
        NVTX_RANGE_POP();

        NVTX_RANGE_PUSH("Initialize_GPU_Frames");
        initalize_gpu_frame(&frame_original, camera_params);
        initialize_gpu_debayer(&debayer, camera_params);
        NVTX_RANGE_POP();

        NVTX_RANGE_PUSH("Allocate_GPU_Memory");
        ck(cudaMalloc(&d_rgb_temp_, scaled_width_ * scaled_height_ * 3));
        NVTX_RANGE_POP();

        NVTX_RANGE_PUSH("Setup_Encoder_Context");
        encoder.cuContext = m_cuContext;
        encoder.eFormat = NV_ENC_BUFFER_FORMAT_NV12;
        NVTX_RANGE_POP();
        
        NVTX_RANGE_PUSH("Create_NVIDIA_Encoder");
        CUDA_CTX_LOG("Creating NVIDIA encoder");
        encoder.pEnc = new NvEncoderCuda(m_cuContext, camera_params->width, camera_params->height, encoder.eFormat);
        CUDA_CTX_LOG("NVIDIA encoder created successfully");
        NVTX_RANGE_POP();
        
        NVTX_RANGE_PUSH("Configure_Encoder_Parameters");
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
        NVTX_RANGE_POP();

        std::cout << "===== NVENC Initialization Parameters =====" << std::endl;
        std::cout << "  Width: " << initializeParams.encodeWidth << std::endl;
        std::cout << "  Height: " << initializeParams.encodeHeight << std::endl;
        std::cout << "  Frame Rate: " << initializeParams.frameRateNum << "/" << initializeParams.frameRateDen << std::endl;
        std::cout << "  Rate Control: " << encodeConfig.rcParams.rateControlMode << std::endl;
        std::cout << "  Avg Bitrate: " << encodeConfig.rcParams.averageBitRate << std::endl;
        std::cout << "  Input Format: " << encoder.eFormat << std::endl;
        std::cout << "========================================" << std::endl;

        NVTX_RANGE_PUSH("Initialize_Encoder");
        encoder.pEnc->CreateEncoder(&initializeParams);
        encoder.pEnc->SetIOCudaStreams((NV_ENC_CUSTREAM_PTR)&m_stream, (NV_ENC_CUSTREAM_PTR)&m_stream);
        NVTX_RANGE_POP();

        NVTX_RANGE_PUSH("Allocate_YUV_And_Mono_Buffers");
        const NvEncInputFrame *tempFrame = encoder.pEnc->GetNextInputFrame();
        encoder_pitch_ = tempFrame->pitch;

        size_t encoder_buffer_size = (size_t)encoder_pitch_ * scaled_height_ * 3 / 2;
        ck(cudaMalloc(&d_iyuv_temp_, encoder_buffer_size));
        
        // --- NEW CODE: Allocate and pre-fill the monochrome UV plane ---
        if (!camera_params->color) {
            size_t uv_plane_size = (size_t)encoder_pitch_ * scaled_height_ / 4;
            ck(cudaMalloc(&d_uv_default_plane_, uv_plane_size));
            ck(cudaMemset(d_uv_default_plane_, 128, uv_plane_size));
            std::cout << "[GPUVideoEncoder] Pre-allocated monochrome UV plane (" << uv_plane_size << " bytes)." << std::endl;
        }
        // --- END NEW CODE ---
        NVTX_RANGE_POP();

        std::cout << "[GPUVideoEncoder] Native resolution " << scaled_width_ << "x" << scaled_height_ 
                  << " with encoder pitch: " << encoder_pitch_ << std::endl;

        NVTX_RANGE_PUSH("Initialize_File_Writer");
        initialize_writer(&writer, camera_params, folder_name, codec);
        writer.video->create_thread();
        NVTX_RANGE_POP();
        
        std::cout << "[GPUVideoEncoder] Successfully initialized NATIVE RESOLUTION encoder for " << name 
                  << " - Codec: " << codec << ", Preset: " << preset << ", Tuning: " << tuning << std::endl;
        *encoder_ready_signal = true;
        
        NVTX_RANGE_PUSH("Cleanup_Context");
        CUDA_CTX_LOG("Popping CUDA context in constructor");
        CUcontext popped_context;
        ck(cuCtxPopCurrent(&popped_context));
        NVTX_RANGE_POP();
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
    NVTX_ENCODE("GPUVideoEncoder_Destructor");
    
    std::cout << "[GPUVideoEncoder] Destructor for " << this->threadName << std::endl;
    ck(cuCtxPushCurrent(m_cuContext));

    NVTX_RANGE_PUSH("Close_Writer_And_Encoder");
    close_writer(&encoder, &writer);
    NVTX_RANGE_POP();
    
    NVTX_RANGE_PUSH("Cleanup_CUDA_Resources");
    if (m_stream) { cudaStreamDestroy(m_stream); }
    cudaFree(frame_original.d_orig);
    cudaFree(debayer.d_debayer);
    if (d_rgb_temp_) cudaFree(d_rgb_temp_);
    if (d_iyuv_temp_) cudaFree(d_iyuv_temp_);
    if (d_scaled_mono_buffer_) cudaFree(d_scaled_mono_buffer_);
    // Free the pre-filled monochrome plane
    if (d_uv_default_plane_) cudaFree(d_uv_default_plane_);
    NVTX_RANGE_POP();

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

    NVTX_ENCODE("GPUEncoder_WorkerFunction");

    try {
        NVTX_RANGE_PUSH("Setup_Device_And_Stream");
        ck(cudaSetDevice(camera_params->gpu_id));
        nppSetStream(m_stream);
        NVTX_RANGE_POP();
        
        {
            NVTX_SYNC("Stream_Wait_For_Data_Ready_Event");
            if (entry->data_ready) {
                ck(cudaStreamWaitEvent(m_stream, entry->data_ready, 0));
                ck(cudaEventDestroy(entry->data_ready));
                entry->data_ready = nullptr; 
            }
        }

        const int width = camera_params->width;
        const int height = camera_params->height;

        NVTX_RANGE_PUSH("FPS_Tracking");
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
        NVTX_RANGE_POP();

        ENCODER_CTX_LOG("Processing frame dimensions", entry->frame_id);
        
        NVTX_RANGE_PUSH("Input_Frame_Processing");
        // GPU DIRECT OPTIMIZATION: Check if we can skip the copy
        if (entry->gpu_direct_mode) {
            NVTX_RANGE_PUSH("GPU_Direct_Zero_Copy");
            std::cout << "[GPUEncoder] 🚀 GPU DIRECT Frame " << entry->frame_id 
                      << " - Using camera buffer directly (ZERO COPY!)" << std::endl;
            
            // Use the GPU Direct pointer directly - NO COPY!
            frame_original.d_orig = entry->d_image;
            
            ENCODER_CTX_LOG("GPU Direct path - no copy needed", entry->frame_id);
            NVTX_RANGE_POP();
        } else {
            NVTX_GPU_COPY("Traditional_Frame_Copy");
            std::cout << "[GPUEncoder] 🐌 COPY PATH Frame " << entry->frame_id 
                      << " - Copying frame data (" << (width * height / 1024 / 1024) << "MB)" << std::endl;
            
            // Traditional copy path
            CUDA_MEM_LOG("Copying frame data from entry", entry->d_image, width * height, entry->frame_id);
            ENCODER_CTX_LOG("About to copy frame data", entry->frame_id);
            ck(cudaMemcpyAsync(frame_original.d_orig, entry->d_image, 
                               (size_t)width * height, cudaMemcpyDeviceToDevice, m_stream));
        }
        NVTX_RANGE_POP();

        if (camera_params->color) {
            NVTX_RANGE_PUSH("Color_Frame_Processing");
            // === COLOR PATH ===
            ENCODER_CTX_LOG("Processing COLOR frame", entry->frame_id);
            std::cout << "[GPUEncoder] Processing COLOR frame at native resolution..." << std::endl;
            
            NVTX_RANGE_PUSH("Debayer_Operation");
            // Debayer the raw Bayer data to RGBA
            debayer_frame_gpu(camera_params, &frame_original, &debayer);
            NVTX_RANGE_POP();
            
            NVTX_RANGE_PUSH("RGBA_to_RGB_Conversion");
            // Convert RGBA to RGB first
            ck(cudaMalloc(&d_rgb_temp_, width * height * 3));
            
            // Convert RGBA to RGB (remove alpha channel)
            rgba2rgb_convert(d_rgb_temp_, debayer.d_debayer, width, height, m_stream);
            NVTX_RANGE_POP();
            
            NVTX_RANGE_PUSH("RGB_to_YUV420_Conversion");
            // Now convert RGB to IYUV using NPP
            NppiSize image_size = {width, height};
            
            // Calculate plane pointers with ENCODER PITCH
            unsigned char* d_y_plane = d_iyuv_temp_;
            unsigned char* d_u_plane = d_iyuv_temp_ + ((size_t)encoder_pitch_ * height);
            unsigned char* d_v_plane = d_u_plane + ((size_t)encoder_pitch_ * height / 4);
            
            // Convert RGB to YUV420 planar (IYUV format)
            const unsigned char* rgb_planes[3] = {
                d_rgb_temp_,                    // R plane
                d_rgb_temp_ + width * height,   // G plane  
                d_rgb_temp_ + 2 * width * height // B plane
            };
            
            unsigned char* yuv_planes[3] = {d_y_plane, d_u_plane, d_v_plane};
            int yuv_steps[3] = {encoder_pitch_, encoder_pitch_ / 2, encoder_pitch_ / 2};
            
            // Use NPP to convert RGB to YUV420
            NppStatus npp_status = nppiRGBToYUV420_8u_C3P3R(
                d_rgb_temp_, width * 3,     // RGB source
                yuv_planes, yuv_steps,      // YUV destination planes
                image_size                  // Image size
            );
            
            if (npp_status != NPP_SUCCESS) {
                std::cerr << "[GPUEncoder] NPP RGB to YUV420 conversion failed: " << npp_status << std::endl;
            }
            NVTX_RANGE_POP();
            
            NVTX_RANGE_PUSH("Cleanup_RGB_Buffer");
            cudaFree(d_rgb_temp_);
            d_rgb_temp_ = nullptr;
            NVTX_RANGE_POP();
            
            NVTX_RANGE_POP(); // Color_Frame_Processing
            
        } else {
            NVTX_RANGE_PUSH("Monochrome_Frame_Processing");
            // === MONOCHROME PATH ===
            ENCODER_CTX_LOG("Processing MONOCHROME frame", entry->frame_id);
            
            NVTX_RANGE_PUSH("Calculate_YUV_Planes");
            unsigned char* d_y_plane_dst = d_iyuv_temp_;
            unsigned char* d_u_plane_dst = d_iyuv_temp_ + ((size_t)encoder_pitch_ * height);
            unsigned char* d_v_plane_dst = d_u_plane_dst + ((size_t)encoder_pitch_ * height / 4);
            NVTX_RANGE_POP();
            
            CUDA_MEM_LOG("Converting to IYUV", d_iyuv_temp_, encoder_pitch_ * height * 3 / 2, entry->frame_id);
            
            NVTX_RANGE_PUSH("Copy_Y_Plane");
            ck(cudaMemcpy2DAsync(d_y_plane_dst, encoder_pitch_, frame_original.d_orig, width, width, height, cudaMemcpyDeviceToDevice, m_stream));
            NVTX_RANGE_POP();
        
            // For monochrome, we can use a pre-filled UV plane
            NVTX_RANGE_PUSH("Set_UV_Planes_Optimized");
            size_t uv_plane_size = (size_t)encoder_pitch_ * height / 4;
            ck(cudaMemcpyAsync(d_u_plane_dst, d_uv_default_plane_, uv_plane_size, cudaMemcpyDeviceToDevice, m_stream));
            ck(cudaMemcpyAsync(d_v_plane_dst, d_uv_default_plane_, uv_plane_size, cudaMemcpyDeviceToDevice, m_stream));
            NVTX_RANGE_POP();
            
            NVTX_RANGE_POP(); // Monochrome_Frame_Processing
        };

        NVTX_ENCODE("NVIDIA_Hardware_Encode");
        NVTX_RANGE_PUSH("Setup_Encoder_Input");
        // NVIDIA encoder call
        ENCODER_CTX_LOG("About to call NVIDIA EncodeFrame - CRITICAL POINT", entry->frame_id);
        dumpCudaState("Pre-EncodeFrame", entry->frame_id);
        
        // Get encoder input frame and copy our IYUV data to it
        const NvEncInputFrame *encoderInputFrame = encoder.pEnc->GetNextInputFrame();
        NVTX_RANGE_POP();
        
        NVTX_RANGE_PUSH("Copy_To_Encoder_Buffer");
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
        NVTX_RANGE_POP();

        NVTX_RANGE_PUSH("Hardware_Encode_Call");
        // Encode the frame
        encoder.pEnc->EncodeFrame(encoder.vPacket);
        NVTX_RANGE_POP();
        
        ENCODER_CTX_LOG("EncodeFrame completed successfully", entry->frame_id);
        ENCODER_CTX_LOG("Processing encoded packets", entry->frame_id);

        NVTX_RANGE_PUSH("Process_Encoded_Packets");
        // Push encoded packets to writer
        for (std::vector<uint8_t> &packet : encoder.vPacket) {
            writer.video->push_packet(packet.data(), (int)packet.size(), encoder.num_frame_encode++);
        }
        NVTX_RANGE_POP();
        
        NVTX_RANGE_PUSH("Write_Frame_Metadata");
        // Write metadata
        write_metadata(writer.metadata, entry->frame_id, entry->timestamp, entry->timestamp_sys);
        NVTX_RANGE_POP();
        
        std::cout << "[GPUEncoder] Frame " << entry->frame_id << " encoded successfully" << std::endl;

        NVTX_RANGE_PUSH("Reference_Count_Management");
        // Handle reference counting and GPU Direct camera buffer management
        ENCODER_CTX_LOG("Handling reference count", entry->frame_id);
        int remaining_refs = entry->ref_count.fetch_sub(1, std::memory_order_acq_rel);
        
        if (remaining_refs == 1) {
            NVTX_RANGE_PUSH("Last_Worker_Cleanup");
            // This is the last worker - handle cleanup
            if (entry->gpu_direct_mode && entry->camera_instance && entry->camera_frame_struct) {
                NVTX_CAMERA("GPU_Direct_Camera_Requeue");
                // GPU Direct: Requeue the camera buffer now that all workers are done
                EVT_CameraQueueFrame(entry->camera_instance, entry->camera_frame_struct);
                std::cout << "[GPUEncoder] 🔄 GPU DIRECT Frame " << entry->frame_id 
                          << " - Last worker requeued camera buffer" << std::endl;
                ENCODER_CTX_LOG("GPU Direct camera buffer requeued by last worker", entry->frame_id);
            }
            
            NVTX_RANGE_PUSH("Reset_Entry_Fields");
            // Reset GPU Direct fields for recycling
            entry->gpu_direct_mode = false;
            entry->owns_memory = true;
            entry->camera_buffer_ptr = nullptr;
            entry->camera_instance = nullptr;
            entry->camera_frame_struct = nullptr;
            NVTX_RANGE_POP();
            
            NVTX_QUEUE("Recycle_Worker_Entry");
            // Recycle the worker entry
            m_recycle_queue.push(entry);
            
            std::cout << "[GPUEncoder] Frame " << entry->frame_id 
                      << " - Last worker recycled entry" << std::endl;
            NVTX_RANGE_POP(); // Last_Worker_Cleanup
        } else {
            std::cout << "[GPUEncoder] Frame " << entry->frame_id 
                      << " - Worker finished, " << (remaining_refs - 1) << " workers remaining" << std::endl;
        }
        NVTX_RANGE_POP(); // Reference_Count_Management

    } catch (const std::exception& e) {
        NVTX_RANGE_PUSH("Exception_Handling");
        std::cerr << "[GPUEncoder] Exception in WorkerFunction: " << e.what() << std::endl;
        
        // Handle reference counting even on error
        int remaining_refs = entry->ref_count.fetch_sub(1, std::memory_order_acq_rel);
        if (remaining_refs == 1) {
            if (entry->gpu_direct_mode && entry->camera_instance && entry->camera_frame_struct) {
                EVT_CameraQueueFrame(entry->camera_instance, entry->camera_frame_struct);
            }
            entry->gpu_direct_mode = false;
            entry->owns_memory = true;
            entry->camera_buffer_ptr = nullptr;
            entry->camera_instance = nullptr;
            entry->camera_frame_struct = nullptr;
            m_recycle_queue.push(entry);
        }
        NVTX_RANGE_POP();
        
        return false;
    }
    
    ENCODER_CTX_LOG("=== EXITING WorkerFunction ===", entry->frame_id);
    return false;
}