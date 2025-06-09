// src/gpu_video_encoder.cpp

#if defined(__GNUC__)
#include <unistd.h>
#endif
#include <stdio.h>
#include <string.h>
#include "kernel.cuh"
#include "gpu_video_encoder.h"
#include <cuda_runtime_api.h>

// Helper to initialize the NvEncoder (no changes needed)
template <class EncoderClass>
void InitializeEncoder(EncoderClass &pEnc, NvEncoderInitParam encodeCLIOptions, NV_ENC_BUFFER_FORMAT eFormat)
{
    NV_ENC_INITIALIZE_PARAMS initializeParams = {NV_ENC_INITIALIZE_PARAMS_VER};
    NV_ENC_CONFIG encodeConfig = {NV_ENC_CONFIG_VER};

    initializeParams.encodeConfig = &encodeConfig;
    pEnc->CreateDefaultEncoderParams(&initializeParams, encodeCLIOptions.GetEncodeGUID(), encodeCLIOptions.GetPresetGUID(), encodeCLIOptions.GetTuningInfo());
    encodeCLIOptions.SetInitParams(&initializeParams, eFormat);

    pEnc->CreateEncoder(&initializeParams);
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


static inline void encode_frame(EncoderContext *encoder, FFmpegWriter *writer, Debayer *debayer)
{
    const NvEncInputFrame *encoderInputFrame = encoder->pEnc->GetNextInputFrame();
    NvEncoderCuda::CopyToDeviceFrame(encoder->cuContext,
                                     debayer->d_debayer,
                                     0,
                                     (CUdeviceptr)encoderInputFrame->inputPtr,
                                     (int)encoderInputFrame->pitch,
                                     encoder->pEnc->GetEncodeWidth(),
                                     encoder->pEnc->GetEncodeHeight(),
                                     CU_MEMORYTYPE_DEVICE,
                                     encoderInputFrame->bufferFormat,
                                     encoderInputFrame->chromaOffsets,
                                     encoderInputFrame->numChromaPlanes);

    encoder->pEnc->EncodeFrame(encoder->vPacket);
    for (std::vector<uint8_t> &packet : encoder->vPacket)
    {
        writer->push_packet(packet.data(), (int)packet.size(), encoder->num_frame_encode++);
    }
}

static inline void write_meatadata(std::ofstream *metadata, unsigned long long frame_id, unsigned long long timestamp, uint64_t timestamp_sys)
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

// --- FIXED CONSTRUCTOR IMPLEMENTATION ---
GPUVideoEncoder::GPUVideoEncoder(const char* name, CameraParams *camera_params, std::string encoder_setup, std::string folder_name, bool* encoder_ready_signal, SafeQueue<WORKER_ENTRY*>& recycle_queue)
    : CThreadWorker<WORKER_ENTRY>(name),
      camera_params(camera_params),
      encoder_setup(encoder_setup),
      folder_name(folder_name),
      encoder_ready_signal(encoder_ready_signal),
      m_recycle_queue(recycle_queue)
{
    ck(cudaSetDevice(camera_params->gpu_id));
    initalize_gpu_frame(&frame_original, camera_params);
    initialize_gpu_debayer(&debayer, camera_params);
    ck(cuCtxCreate(&encoder.cuContext, 0, camera_params->gpu_id));
    encoder.pEnc = new NvEncoderCuda(encoder.cuContext, camera_params->width, camera_params->height, encoder.eFormat);
    encoder.encodeCLIOptions = NvEncoderInitParam(encoder_setup.c_str());
    InitializeEncoder(encoder.pEnc, encoder.encodeCLIOptions, encoder.eFormat);
    initialize_writer(&writer, camera_params, folder_name, encoder_setup);
    writer.video->create_thread();
    *encoder_ready_signal = true;
}

GPUVideoEncoder::~GPUVideoEncoder()
{
    close_writer(&encoder, &writer);
    ck(cudaSetDevice(camera_params->gpu_id));
    cudaFree(frame_original.d_orig);
    cudaFree(debayer.d_debayer);
    cuCtxDestroy(encoder.cuContext);
}

bool GPUVideoEncoder::WorkerFunction(WORKER_ENTRY* entry)
{
    if (!entry) return false;

    ck(cudaSetDevice(camera_params->gpu_id));

    // CHANGE: Copy from the entry's GPU buffer to this worker's internal GPU buffer
    // This is a fast Device-to-Device copy.
    size_t buffer_size = camera_params->width * camera_params->height;
    ck(cudaMemcpy(frame_original.d_orig, entry->d_image, buffer_size, cudaMemcpyDeviceToDevice));

    // Debayer/process the frame
    if (camera_params->color){
        debayer_frame_gpu(camera_params, &frame_original, &debayer);
    } else {
        duplicate_channel_gpu(camera_params, &frame_original, &debayer);
    }

    // Encode the frame and write metadata
    encode_frame(&encoder, writer.video, &debayer);
    write_meatadata(writer.metadata, entry->frame_id, entry->timestamp, entry->timestamp_sys);

    // This worker is a final consumer for this data path.
    // It is done with the entry, so return it to the central recycling queue.
    m_recycle_queue.push(entry);

    // Return false so CThreadWorker does not place it on its own output queue.
    return false;
}