#if defined(__GNUC__)
#include <unistd.h>
#endif
#include <stdio.h>
#include <string.h>
#include "kernel.cuh"
#include "gpu_video_encoder.h"
#include <cuda_runtime_api.h>

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

static inline void initialize_encoder(EncoderContext *encoder, std::string encoder_str, CameraParams *camera_params)
{
    try {
        std::cout << "Initializing encoder with setup string: " << encoder_str << std::endl;
        
        // Parse the encoder string to verify codec
        size_t codec_pos = encoder_str.find("-codec");
        if (codec_pos != std::string::npos) {
            size_t space_pos = encoder_str.find(" ", codec_pos + 6);
            std::string codec = encoder_str.substr(codec_pos + 6, space_pos - (codec_pos + 6));
            std::cout << "Using codec: " << codec << std::endl;
        }

        // Set up CUDA device
        CUdevice cuDevice;
        ck(cuDeviceGet(&cuDevice, camera_params->gpu_id));
        encoder->cuContext = NULL;
        ck(cuCtxCreate(&encoder->cuContext, 0, cuDevice));

        // Initialize encoder parameters
        encoder->eFormat = NV_ENC_BUFFER_FORMAT_ABGR;
        encoder->encodeCLIOptions = NvEncoderInitParam(encoder_str.c_str());
        encoder->pEnc = new NvEncoderCuda(encoder->cuContext, 
                                         camera_params->width,
                                         camera_params->height,
                                         encoder->eFormat);

        NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
        NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };
        initializeParams.encodeConfig = &encodeConfig;

        // Create default parameters
        encoder->pEnc->CreateDefaultEncoderParams(&initializeParams, 
            encoder_str.find("hevc") != std::string::npos ? NV_ENC_CODEC_HEVC_GUID : NV_ENC_CODEC_H264_GUID,
            encoder_str.find("p1") != std::string::npos ? NV_ENC_PRESET_P1_GUID : 
            encoder_str.find("p3") != std::string::npos ? NV_ENC_PRESET_P3_GUID : 
            encoder_str.find("p5") != std::string::npos ? NV_ENC_PRESET_P5_GUID : 
            NV_ENC_PRESET_P7_GUID,
            NV_ENC_TUNING_INFO_HIGH_QUALITY);

        // Modify parameters for high resolution encoding
        initializeParams.frameRateNum = camera_params->frame_rate;
        initializeParams.frameRateDen = 1;

        // Set reasonable bitrate for 4K resolution
        encodeConfig.rcParams.averageBitRate = 20000000;  // 20 Mbps
        encodeConfig.rcParams.maxBitRate = 25000000;      // 25 Mbps
        encodeConfig.rcParams.vbvBufferSize = 20000000;   // 20 Mb buffer
        encodeConfig.rcParams.vbvInitialDelay = 20000000; // 20 Mb delay

        // Use VBR mode for high quality
        encodeConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
        encodeConfig.rcParams.enableAQ = 1;
        encodeConfig.rcParams.enableTemporalAQ = 1;
        encodeConfig.rcParams.enableLookahead = 1;
        encodeConfig.rcParams.lookaheadDepth = 8;

        // Print detailed configuration
        std::cout << "Encoder configuration:" << std::endl
                  << "Width: " << camera_params->width << std::endl
                  << "Height: " << camera_params->height << std::endl
                  << "Frame Rate: " << camera_params->frame_rate << std::endl
                  << "Bitrate: " << encodeConfig.rcParams.averageBitRate / 1000000 << " Mbps" << std::endl
                  << "GPU ID: " << camera_params->gpu_id << std::endl
                  << "Codec: " << (encoder_str.find("hevc") != std::string::npos ? "HEVC" : "H264") << std::endl;

        // Initialize encoder
        try {
            encoder->pEnc->CreateEncoder(&initializeParams);
            std::cout << "Encoder initialization successful" << std::endl;
        } catch (const NVENCException& e) {
            std::cerr << "NVENC initialization failed: " << e.what() << std::endl;
            std::cerr << "Error code: " << e.getErrorCode() << std::endl;
            throw;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in initialize_encoder: " << e.what() << std::endl;
        throw;
    }
}

static inline void open_metadata_file(std::ofstream *frame_metadata, std::string metadata_file)
{
    frame_metadata->open(metadata_file.c_str());

    if (!(*frame_metadata))
    {
        std::cout << "File did not open!";
        return;
    }
    *frame_metadata << "frame_id,timestamp,timestamp_sys\n";
}

static inline void write_meatadata(std::ofstream *metadata, unsigned long long frame_id, unsigned long long timestamp, uint64_t timestamp_sys)
{
    *metadata << frame_id << "," << timestamp << "," << timestamp_sys << std::endl;
}


static inline void initialize_writer(Writer *writer, CameraParams *camera_params, std::string folder_name, std::string encoder_str)
{
    // writer->video_file = folder_name + "/Cam" + std::to_string(camera_params->camera_id) + ".mp4";
    // writer->metadata_file = folder_name + "/Cam" + std::to_string(camera_params->camera_id) + "_meta.csv";
    writer->video_file = folder_name + "/Cam" + camera_params->camera_serial + ".mp4";
    writer->metadata_file = folder_name + "/Cam" + camera_params->camera_serial + "_meta.csv";
    writer->keyframe_file = folder_name + "/Cam" + camera_params->camera_serial + "_keyframe.csv";

    if (encoder_str.find("h264") != std::string::npos) {
        std::cout << "h264 encoding" << '\n';
        writer->video = new FFmpegWriter(AV_CODEC_ID_H264, camera_params->width, camera_params->height, camera_params->frame_rate, writer->video_file.c_str(), writer->keyframe_file.c_str());
    } else if (encoder_str.find("hevc") != std::string::npos){
        std::cout << "hevc encoding" << '\n';
        writer->video = new FFmpegWriter(AV_CODEC_ID_HEVC, camera_params->width, camera_params->height, camera_params->frame_rate, writer->video_file.c_str(), writer->keyframe_file.c_str());
    } else {
        std::cout << "codec not supported" << '\n';
    }
    writer->metadata = new std::ofstream();
    open_metadata_file(writer->metadata, writer->metadata_file);
}


static inline void encode_frame(EncoderContext *encoder, FFmpegWriter *writer, Debayer *debayer)
{
    // encoding
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
        // For each encoded packet
        // writer->write_packet(packet.data(), (int)packet.size(), encoder->num_frame_encode++);
        writer->push_packet(packet.data(), (int)packet.size(), encoder->num_frame_encode++);
    }
}

static inline void close_writer(EncoderContext *encoder, Writer *writer)
{
    encoder->pEnc->EndEncode(encoder->vPacket);
    for (std::vector<uint8_t> &packet : encoder->vPacket)
    {
        // writer->video->write_packet(packet.data(), (int)packet.size(), encoder->num_frame_encode++);
        writer->video->push_packet(packet.data(), (int)packet.size(), encoder->num_frame_encode++);
    }
    encoder->pEnc->DestroyEncoder();
    (*writer->metadata).close();
}


GPUVideoEncoder::GPUVideoEncoder(const char *name, CameraParams *camera_params, std::string encoder_setup, std::string folder_name, bool* encoder_ready_signal)
    : CThreadWorker(name), camera_params(camera_params), display_buffer(display_buffer), encoder_setup(encoder_setup), folder_name(folder_name), encoder_ready_signal(encoder_ready_signal)
{
    memset(workerEntries, 0, sizeof(workerEntries));
    workerEntriesFreeQueueCount = ENCODER_ENTRIES_MAX;
    for (int i = 0; i < workerEntriesFreeQueueCount; i++)
    {
        workerEntriesFreeQueue[i] = &workerEntries[i];
    }
}

GPUVideoEncoder::~GPUVideoEncoder()
{
}

void GPUVideoEncoder::ProcessOneFrame(void* f)
{
    WORKER_ENTRY entry = *(WORKER_ENTRY*)f;
    PutObjectToQueueOut(f);
    
    // copy frame from cpu to gpu
    ck(cudaMemcpy2D(frame_original.d_orig, camera_params->width, entry.imagePtr, camera_params->width, camera_params->width, camera_params->height, cudaMemcpyHostToDevice));

    if (camera_params->color){
        debayer_frame_gpu(camera_params, &frame_original, &debayer);
    } else {
        duplicate_channel_gpu(camera_params, &frame_original, &debayer);
    }

    encode_frame(&encoder, writer.video, &debayer);
    write_meatadata(writer.metadata, entry.frame_id, entry.timestamp, entry.timestamp_sys);
}

void GPUVideoEncoder::ThreadRunning()
{
    ck(cudaSetDevice(camera_params->gpu_id));
    // innitialization
    initalize_gpu_frame(&frame_original, camera_params);
    initialize_gpu_debayer(&debayer, camera_params);

    initialize_encoder(&encoder, encoder_setup, camera_params);
    initialize_writer(&writer, camera_params, folder_name, encoder_setup);

    // start writing thread
    writer.video->create_thread();

    *encoder_ready_signal = true;
    while(IsMachineOn())
    {
        void* f = GetObjectFromQueueIn();
        if(f) {
            ProcessOneFrame(f);
        }
    }

    // empty queue
    while(GetCountQueueInSize()) 
    {
        void* f = GetObjectFromQueueIn();
        if(f) {
            ProcessOneFrame(f);
        }
    }
    
    close_writer(&encoder, &writer);
    std::string print_out;
    print_out += "\n" + camera_params->camera_serial;
    print_out += ", Frame encoded: " + std::to_string(encoder.num_frame_encode);
    std::cout << print_out << std::endl;

    writer.video->quit_thread();
    writer.video->join_thread();

    delete writer.video;
    delete writer.metadata;
    delete encoder.pEnc;
    cudaFree(frame_original.d_orig);
    cudaFree(debayer.d_debayer);
}


bool GPUVideoEncoder::PushToDisplay(void *imagePtr, size_t bufferSize, int width, int height, int pixelFormat, unsigned long long timestamp, unsigned long long frame_id, uint64_t timestamp_sys)
{
    WORKER_ENTRY *entriesOut[ENCODER_ENTRIES_MAX]; // entris got out from saver thread, their frames should be returned to driver queue.
    int entriesOutCount = ENCODER_ENTRIES_MAX;
    GetObjectsFromQueueOut((void **)entriesOut, &entriesOutCount);
    if (entriesOutCount)
    { // return the frames to driver, and put entries back to frameSaveEntriesFreeQueue
        // printf("++++++++++++++++++++++++ %s %s %d get WORKER_ENTRY from out entriesOutCount: %d\n", __FILE__, __FUNCTION__, __LINE__, entriesOutCount);
        for (int j = 0; j < entriesOutCount; j++)
        {
            workerEntriesFreeQueue[workerEntriesFreeQueueCount] = entriesOut[j];
            workerEntriesFreeQueueCount++;
        }
    }

    // get the free entry if there is one and put in to QueueIn, otherwise EVT_CameraQueueFrame.
    if (workerEntriesFreeQueueCount)
    {
        // printf("++++++++++++++++++++++++ %s %s %d put WORKER_ENTRY to in workerEntriesFreeQueueCount: %d\n", __FILE__, __FUNCTION__, __LINE__, workerEntriesFreeQueueCount);
        WORKER_ENTRY *entry = workerEntriesFreeQueue[workerEntriesFreeQueueCount - 1];
        workerEntriesFreeQueueCount--;
        entry->imagePtr = imagePtr;
        entry->bufferSize = bufferSize;
        entry->width = width;
        entry->height = height;
        entry->pixelFormat = pixelFormat;
        entry->timestamp = timestamp;
        entry->frame_id = frame_id;
        entry->timestamp_sys = timestamp_sys;
        PutObjectToQueueIn(entry);
        return true;
    }
    return false;
}