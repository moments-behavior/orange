#include "crop_and_encode_worker.h"
#include "kernel.cuh"
#include <nppi.h>
#include <npp.h>
#include <nppi_color_conversion.h>
#include <nppi_geometry_transforms.h>
#include <algorithm> // For std::max_element

CropAndEncodeWorker::CropAndEncodeWorker(const char* name, CameraParams* camera_params, const std::string& folder_name, SafeQueue<WORKER_ENTRY*>& recycle_queue)
    : CThreadWorker(name), camera_params_(camera_params), folder_name_(folder_name), m_recycle_queue(recycle_queue) {

    std::cout << "[CropAndEncodeWorker] Initializing " << name << " on GPU " << camera_params_->gpu_id << std::endl;

    ck(cudaSetDevice(camera_params_->gpu_id));
    ck(cudaStreamCreate(&m_stream));

    initialize_gpu_debayer(&debayer_gpu_, camera_params_);

    writer_.video_file = folder_name_ + "/Cam" + camera_params_->camera_serial + "_crop.mp4";
    writer_.keyframe_file = folder_name_ + "/Cam" + camera_params_->camera_serial + "_crop_keyframe.csv";
    writer_.metadata_file = folder_name_ + "/Cam" + camera_params_->camera_serial + "_crop_meta.csv";

    writer_.video = new FFmpegWriter(AV_CODEC_ID_HEVC, 256, 256, camera_params_->frame_rate,
                                   writer_.video_file.c_str(), writer_.keyframe_file.c_str());
    writer_.video->create_thread();

    writer_.metadata = new std::ofstream();
    writer_.metadata->open(writer_.metadata_file.c_str());
    if (!(*writer_.metadata)) {
        std::cout << "[CropAndEncodeWorker] Warning: Could not open metadata file!" << std::endl;
    } else {
        *writer_.metadata << "frame_id,timestamp,timestamp_sys,detection_confidence,crop_x,crop_y,crop_w,crop_h\n";
    }

    ck(cudaMalloc(&d_cropped_bgr_, 256 * 256 * 3));
    ck(cudaMalloc(&d_yuv_buffer_, 256 * 256 * 3 / 2));

    try {
        CUcontext cuContext;
        ck(cuCtxGetCurrent(&cuContext));

        encoder_ = new NvEncoderCuda(cuContext, 256, 256, NV_ENC_BUFFER_FORMAT_NV12);

        // 1. All encoder parameters are now in one place.
        NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
        NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };
        initializeParams.encodeConfig = &encodeConfig;

        // 2. Set the desired GUIDs for lossless encoding.
        // Using HEVC is required for the LOSSLESS tuning preset.
        GUID codecGuid = NV_ENC_CODEC_HEVC_GUID;
        GUID presetGuid = NV_ENC_PRESET_P7_GUID;
        NV_ENC_TUNING_INFO tuningInfo = NV_ENC_TUNING_INFO_LOSSLESS;

        std::cout << "[CropAndEncodeWorker] CONFIGURING FOR LOSSLESS (HEVC), HIGH-QUALITY RECORDING." << std::endl;

        // 3. Create the default parameter set from the GUIDs.
        encoder_->CreateDefaultEncoderParams(&initializeParams, codecGuid, presetGuid, tuningInfo);

        // 4. Override any specific parameters you need.
        encodeConfig.gopLength = NVENC_INFINITE_GOPLENGTH; // Infinite GOP length for lossless
        encodeConfig.frameIntervalP = 1; // No B-frames, only I-
        initializeParams.frameRateNum = camera_params_->frame_rate;
        initializeParams.frameRateDen = 1;
        initializeParams.enablePTD = 1;
        
        // The lossless preset handles the quality settings automatically,
        // but explicitly setting a low QP (Quantization Parameter) can be good practice.
        encodeConfig.rcParams.constQP = { 0, 0, 0 };

        // 5. Create the encoder with the final parameters.
        encoder_->CreateEncoder(&initializeParams);

        // 6. Final setup steps.
        encoder_->SetIOCudaStreams((NV_ENC_CUSTREAM_PTR)&m_stream, (NV_ENC_CUSTREAM_PTR)&m_stream);
        const NvEncInputFrame *tempFrame = encoder_->GetNextInputFrame();
        encoder_pitch_ = tempFrame->pitch;

    } catch (const std::exception& e) {
        std::cerr << "[CropAndEncodeWorker] Failed to initialize encoder: " << e.what() << std::endl;
        if (encoder_) {
            delete encoder_;
            encoder_ = nullptr;
        }
        throw;
    }
}

CropAndEncodeWorker::~CropAndEncodeWorker() {
    std::cout << "[CropAndEncodeWorker] Destructor for " << threadName << std::endl;

    if (camera_params_) {
        ck(cudaSetDevice(camera_params_->gpu_id));
    }

    if (debayer_gpu_.d_debayer) cudaFree(debayer_gpu_.d_debayer);

    if (writer_.video) {
        writer_.video->quit_thread();
        writer_.video->join_thread();
        delete writer_.video;
    }

    if (writer_.metadata && writer_.metadata->is_open()) {
        writer_.metadata->close();
        delete writer_.metadata;
    }

    if (d_cropped_bgr_) cudaFree(d_cropped_bgr_);
    if (d_yuv_buffer_) cudaFree(d_yuv_buffer_);
    if (encoder_) delete encoder_;
    if (m_stream) cudaStreamDestroy(m_stream);
}

// // Enhanced WorkerFunction with comprehensive debugging
// // Enhanced WorkerFunction with comprehensive debugging
// bool CropAndEncodeWorker::WorkerFunction(WORKER_ENTRY* entry) {
//     if (!entry || entry->detections.empty()) {
//         if (entry && entry->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
//             m_recycle_queue.push(entry);
//         }
//         return false;
//     }

//     ck(cudaSetDevice(camera_params_->gpu_id));
//     nppSetStream(m_stream);

//     static bool first_debug_log = true;

//     try {
//         if (entry->event_ptr) ck(cudaStreamWaitEvent(m_stream, *entry->event_ptr, 0));

//         // ---------------- Best detection ----------------
//         const auto& best_detection = *std::max_element(
//             entry->detections.begin(), entry->detections.end(),
//             [](const pose::Object& a, const pose::Object& b) { return a.prob < b.prob; });

//         // ---------------- Debayer + RGBA→RGB -------------
//         FrameGPU frame_original_gpu;
//         frame_original_gpu.d_orig    = entry->d_image;
//         frame_original_gpu.size_pic  = entry->width * entry->height;
//         debayer_frame_gpu(camera_params_, &frame_original_gpu, &debayer_gpu_);

//         unsigned char* d_rgb_full;
//         ck(cudaMalloc(&d_rgb_full, static_cast<size_t>(entry->width) * entry->height * 3));
//         rgba2rgb_convert(d_rgb_full, debayer_gpu_.d_debayer,
//                          entry->width, entry->height, m_stream);

//         // ---------------- Crop to 256×256 ----------------
//         const int CROP = 256;
//         float cx = best_detection.rect.x + best_detection.rect.width  * 0.5f;
//         float cy = best_detection.rect.y + best_detection.rect.height * 0.5f;
//         int   ix = std::clamp(static_cast<int>(cx) - CROP / 2, 0, entry->width  - CROP);
//         int   iy = std::clamp(static_cast<int>(cy) - CROP / 2, 0, entry->height - CROP);

//         NppStatus copy_st = nppiCopy_8u_C3R(
//             d_rgb_full + (iy * entry->width + ix) * 3, entry->width * 3,
//             d_cropped_bgr_, CROP * 3,
//             {CROP, CROP});
//         ck(cudaFree(d_rgb_full));
//         if (copy_st != NPP_SUCCESS) {
//             std::cerr << "[CropAndEncodeWorker] NPP crop failed: " << copy_st << std::endl;
//             return false;
//         }

//         // ---------------- RGB → NV12 --------------------
//         if (encoder_) {
//             CUcontext cuContext; ck(cuCtxGetCurrent(&cuContext));
//             ck(cudaStreamSynchronize(m_stream));

//             // ============ ENHANCED DEBUGGING =============
//             std::cout << "[CropAndEncodeWorker] Frame " << entry->frame_id << " - Starting NV12 conversion" << std::endl;
//             std::cout << "  encoder_pitch_: " << encoder_pitch_ << std::endl;
//             std::cout << "  Expected Y plane size: " << static_cast<size_t>(encoder_pitch_) * 256 << " bytes" << std::endl;
//             std::cout << "  Expected UV plane size: " << static_cast<size_t>(encoder_pitch_) * 256 / 2 << " bytes" << std::endl;
//             std::cout << "  Total NV12 buffer size: " << static_cast<size_t>(encoder_pitch_) * 256 * 3 / 2 << " bytes" << std::endl;

//             // Use YUV420 planar format (IYUV) like gpu_video_encoder.cpp does
//             unsigned char* d_y_plane = d_yuv_buffer_;
//             unsigned char* d_u_plane = d_y_plane + static_cast<size_t>(encoder_pitch_) * 256;
//             unsigned char* d_v_plane = d_u_plane + static_cast<size_t>(encoder_pitch_) * 256 / 4;

//             std::cout << "  Y plane ptr: " << static_cast<void*>(d_y_plane) << std::endl;
//             std::cout << "  U plane ptr: " << static_cast<void*>(d_u_plane) << std::endl;
//             std::cout << "  V plane ptr: " << static_cast<void*>(d_v_plane) << std::endl;
//             std::cout << "  U plane offset: " << static_cast<size_t>(encoder_pitch_) * 256 << " bytes" << std::endl;
//             std::cout << "  V plane offset: " << static_cast<size_t>(encoder_pitch_) * 256 + static_cast<size_t>(encoder_pitch_) * 256 / 4 << " bytes" << std::endl;

//             unsigned char* yuv_planes[3] = {d_y_plane, d_u_plane, d_v_plane};
//             int yuv_steps[3] = {encoder_pitch_, encoder_pitch_ / 2, encoder_pitch_ / 2};

//             std::cout << "  NPP conversion steps: Y=" << yuv_steps[0] << ", U=" << yuv_steps[1] << ", V=" << yuv_steps[2] << std::endl;

//             // Use YUV420 planar conversion (IYUV format) like gpu_video_encoder.cpp
//             NppStatus conv_st = nppiRGBToYUV420_8u_C3P3R(
//                 d_cropped_bgr_, 256 * 3,        // RGB source
//                 yuv_planes, yuv_steps,          // YUV destination planes  
//                 {256, 256});
//             if (conv_st != NPP_SUCCESS) {
//                 std::cerr << "[CropAndEncodeWorker] RGB→NV12 conversion failed: "
//                           << conv_st << std::endl;
//                 return false;
//             }

//             std::cout << "  NPP conversion successful" << std::endl;

//             std::cout << "[CropAndEncodeWorker] Checking buffer layout:" << std::endl;
//             std::cout << "  d_yuv_buffer_ ptr: " << static_cast<void*>(d_yuv_buffer_) << std::endl;
//             std::cout << "  Y plane start: " << static_cast<void*>(d_y_plane) << std::endl;
//             std::cout << "  U plane start: " << static_cast<void*>(d_u_plane) << std::endl;
//             std::cout << "  V plane start: " << static_cast<void*>(d_v_plane) << std::endl;

//             // Sample first few bytes of each plane
//             unsigned char h_sample[16];
//             ck(cudaMemcpy(h_sample, d_y_plane, 16, cudaMemcpyDeviceToHost));
//             std::cout << "  Y plane first 16 bytes: ";
//             for(int i = 0; i < 16; i++) std::cout << (int)h_sample[i] << " ";
//             std::cout << std::endl;

//             ck(cudaMemcpy(h_sample, d_u_plane, 16, cudaMemcpyDeviceToHost));
//             std::cout << "  U plane first 16 bytes: ";
//             for(int i = 0; i < 16; i++) std::cout << (int)h_sample[i] << " ";
//             std::cout << std::endl;

//             ck(cudaMemcpy(h_sample, d_v_plane, 16, cudaMemcpyDeviceToHost));
//             std::cout << "  V plane first 16 bytes: ";
//             for(int i = 0; i < 16; i++) std::cout << (int)h_sample[i] << " ";
//             std::cout << std::endl;

//             const NvEncInputFrame* encIn = encoder_->GetNextInputFrame();
            
//             std::cout << "[CropAndEncodeWorker] Encoder input frame details:" << std::endl;
//             std::cout << "  encIn->inputPtr: " << static_cast<void*>(encIn->inputPtr) << std::endl;
//             std::cout << "  encIn->pitch: " << encIn->pitch << std::endl;
//             std::cout << "  encIn->bufferFormat: " << encIn->bufferFormat << std::endl;
//             std::cout << "  encIn->numChromaPlanes: " << encIn->numChromaPlanes << std::endl;
//             std::cout << "  Expected format NV_ENC_BUFFER_FORMAT_NV12: " << NV_ENC_BUFFER_FORMAT_NV12 << std::endl;
            
//             // Print chroma offsets
//             std::cout << "  Chroma offsets: [";
//             for (int i = 0; i < encIn->numChromaPlanes; i++) {
//                 std::cout << encIn->chromaOffsets[i];
//                 if (i < encIn->numChromaPlanes - 1) std::cout << ", ";
//             }
//             std::cout << "]" << std::endl;

//             if (encIn->pitch != encoder_pitch_) {
//                 std::cout << "[CropAndEncodeWorker] WARNING: Pitch mismatch!" << std::endl;
//                 std::cout << "  Cached pitch: " << encoder_pitch_ << std::endl;
//                 std::cout << "  Current input frame pitch: " << encIn->pitch << std::endl;
                
//                 // Update our cached pitch to match
//                 encoder_pitch_ = encIn->pitch;
//                 std::cout << "  Updated encoder_pitch_ to: " << encoder_pitch_ << std::endl;
//             }

//             // ============ VALIDATE PARAMETERS =============
//             std::cout << "[CropAndEncodeWorker] CopyToDeviceFrame parameters:" << std::endl;
//             std::cout << "  cuContext: " << cuContext << std::endl;
//             std::cout << "  src ptr (d_yuv_buffer_): " << static_cast<void*>(d_yuv_buffer_) << std::endl;
//             std::cout << "  src pitch: " << encoder_pitch_ << std::endl;
//             std::cout << "  dst ptr (encIn->inputPtr): " << static_cast<void*>(encIn->inputPtr) << std::endl;
//             std::cout << "  dst pitch: " << encIn->pitch << std::endl;
//             std::cout << "  width: 256" << std::endl;
//             std::cout << "  height: 256" << std::endl;
//             std::cout << "  memory type: CU_MEMORYTYPE_DEVICE (" << CU_MEMORYTYPE_DEVICE << ")" << std::endl;
//             std::cout << "  buffer format: " << encIn->bufferFormat << std::endl;

//             // ============ MEMORY VALIDATION =============
//             // Check if our buffer is valid
//             CUdeviceptr test_ptr = (CUdeviceptr)d_yuv_buffer_;
//             CUcontext ptr_context;
//             CUresult ctx_result = cuPointerGetAttribute(&ptr_context, CU_POINTER_ATTRIBUTE_CONTEXT, test_ptr);
//             std::cout << "  Source buffer context check: " << ctx_result << std::endl;
//             if (ctx_result == CUDA_SUCCESS) {
//                 std::cout << "  Source buffer context: " << ptr_context << std::endl;
//             }

//             // Check encoder input buffer
//             CUdeviceptr enc_ptr = (CUdeviceptr)encIn->inputPtr;
//             ctx_result = cuPointerGetAttribute(&ptr_context, CU_POINTER_ATTRIBUTE_CONTEXT, enc_ptr);
//             std::cout << "  Encoder buffer context check: " << ctx_result << std::endl;
//             if (ctx_result == CUDA_SUCCESS) {
//                 std::cout << "  Encoder buffer context: " << ptr_context << std::endl;
//             }

//             // ============ ATTEMPT COPY =============
//             std::cout << "[CropAndEncodeWorker] About to call CopyToDeviceFrame..." << std::endl;

//             NvEncoderCuda::CopyToDeviceFrame(cuContext,
//                                              d_yuv_buffer_, encoder_pitch_, // src (NV12)
//                                              (CUdeviceptr)encIn->inputPtr, encIn->pitch, // dst
//                                              256, 256,
//                                              CU_MEMORYTYPE_DEVICE,
//                                              encIn->bufferFormat,
//                                              encIn->chromaOffsets,
//                                              encIn->numChromaPlanes);

//             std::cout << "[CropAndEncodeWorker] CopyToDeviceFrame completed successfully!" << std::endl;

//             std::vector<std::vector<uint8_t>> packets;
//             encoder_->EncodeFrame(packets);
//             for (auto& p : packets) {
//                 writer_.video->push_packet(p.data(), static_cast<int>(p.size()), frame_counter_);
//             }
//         }

//         // ---------------- Metadata ----------------------
//         ++frame_counter_;
//         if (writer_.metadata && writer_.metadata->is_open()) {
//             *writer_.metadata << entry->frame_id << ',' << entry->timestamp << ','
//                               << entry->timestamp_sys << ',' << best_detection.prob << ','
//                               << best_detection.rect.x << ',' << best_detection.rect.y << ','
//                               << best_detection.rect.width << ',' << best_detection.rect.height << '\n';
//         }
//     } catch (const std::exception& e) {
//         std::cerr << "[CropAndEncodeWorker] Exception processing frame " << entry->frame_id
//                   << ": " << e.what() << std::endl;
        
//         // Add additional context in error case
//         if (encoder_) {
//             const NvEncInputFrame* encIn = encoder_->GetNextInputFrame();
//             std::cerr << "[CropAndEncodeWorker] Error context:" << std::endl;
//             std::cerr << "  encoder_pitch_: " << encoder_pitch_ << std::endl;
//             std::cerr << "  encIn->pitch: " << encIn->pitch << std::endl;
//             std::cerr << "  encIn->bufferFormat: " << encIn->bufferFormat << std::endl;
//             std::cerr << "  Expected NV12 format: " << NV_ENC_BUFFER_FORMAT_NV12 << std::endl;
//         }
//     }

//     // ---------------- Reference counting / recycle -----
//     if (entry->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
//         if (entry->gpu_direct_mode && entry->camera_instance && entry->camera_frame_struct) {
//             EVT_CameraQueueFrame(entry->camera_instance, entry->camera_frame_struct);
//         }
//         m_recycle_queue.push(entry);
//     }

//     return false;
// }

bool CropAndEncodeWorker::WorkerFunction(WORKER_ENTRY* entry) {
    if (!entry || entry->detections.empty()) {
        if (entry && entry->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            m_recycle_queue.push(entry);
        }
        return false;
    }

    ck(cudaSetDevice(camera_params_->gpu_id));
    nppSetStream(m_stream);

    try {
        if (entry->event_ptr) ck(cudaStreamWaitEvent(m_stream, *entry->event_ptr, 0));

        const auto& best_detection = *std::max_element(
            entry->detections.begin(), entry->detections.end(),
            [](const pose::Object& a, const pose::Object& b) { return a.prob < b.prob; });

        // --- START: NEW, EFFICIENT MONO-TO-NV12 PIPELINE ---

        // 1. Define crop region and size
        const int CROP_W = 256;
        const int CROP_H = 256;
        float cx = best_detection.rect.x + best_detection.rect.width  * 0.5f;
        float cy = best_detection.rect.y + best_detection.rect.height * 0.5f;
        int   ix = std::clamp(static_cast<int>(cx) - CROP_W / 2, 0, entry->width  - CROP_W);
        int   iy = std::clamp(static_cast<int>(cy) - CROP_H / 2, 0, entry->height - CROP_H);
        
        // This is the pointer to the original, full-resolution MONO8 image
        const unsigned char* d_mono_full = entry->d_image;

        if (encoder_) {
            const NvEncInputFrame* encIn = encoder_->GetNextInputFrame();
            unsigned char* d_nv12_dst = static_cast<unsigned char*>(encIn->inputPtr);

            // 2. Directly copy the cropped MONO region to the Y-plane of the encoder's buffer
            ck(cudaMemcpy2DAsync(d_nv12_dst,
                                 encIn->pitch, // Destination pitch
                                 d_mono_full + (iy * entry->width + ix), // Source pointer (offset to crop region)
                                 entry->width, // Source pitch
                                 CROP_W, CROP_H,
                                 cudaMemcpyDeviceToDevice,
                                 m_stream));

            // 3. Manually set the UV plane to a neutral gray (128)
            unsigned char* d_uv_plane_dst = d_nv12_dst + encIn->pitch * CROP_H;
            size_t uv_height = CROP_H / 2;
            ck(cudaMemset2DAsync(d_uv_plane_dst,
                                 encIn->pitch, // The UV plane has the same pitch as Y in NV12
                                 128,          // The neutral chroma value
                                 CROP_W,       // The width of the UV plane is the same as Y
                                 uv_height,
                                 m_stream));

            // 4. Encode the frame
            std::vector<std::vector<uint8_t>> packets;
            encoder_->EncodeFrame(packets);
            for (auto& p : packets) {
                writer_.video->push_packet(p.data(), static_cast<int>(p.size()), frame_counter_);
            }
        }

        // --- END: NEW PIPELINE ---

        ++frame_counter_;
        if (writer_.metadata && writer_.metadata->is_open()) {
            *writer_.metadata << entry->frame_id << ',' << entry->timestamp << ','
                              << entry->timestamp_sys << ',' << best_detection.prob << ','
                              << best_detection.rect.x << ',' << best_detection.rect.y << ','
                              << best_detection.rect.width << ',' << best_detection.rect.height << '\n';
        }
    } catch (const std::exception& e) {
        std::cerr << "[CropAndEncodeWorker] Exception processing frame " << entry->frame_id
                  << ": " << e.what() << std::endl;
    }

    if (entry->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        if (entry->gpu_direct_mode && entry->camera_instance && entry->camera_frame_struct) {
            EVT_CameraQueueFrame(entry->camera_instance, entry->camera_frame_struct);
        }
        m_recycle_queue.push(entry);
    }

    return false;
}