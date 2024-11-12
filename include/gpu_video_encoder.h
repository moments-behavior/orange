#ifndef ORANGE_GPU_VIDEO_ENCODER
#define ORANGE_GPU_VIDEO_ENCODER

#include "threadworker.h"
#include "video_capture.h"
#include "FFmpegWriter.h"
#include "NvEncoder/NvEncoderCuda.h"
#include "NvEncoder/NvEncoderCLIOptions.h"
#include "image_processing.h"
#include "camera_params.h"
#include <string>
#include <vector>
#include <fstream>

#define ENCODER_ENTRIES_MAX 20

namespace evt {

// Add this declaration with the other function declarations
bool is_h264_codec(const std::string& encoder_str);

// Define structs before using them in function declarations
struct Writer {
    std::string video_file;
    std::string keyframe_file;
    std::string metadata_file;
    FFmpegWriter* video;
    std::ofstream* metadata;
};

struct EncoderContext {
    NV_ENC_BUFFER_FORMAT eFormat;
    NvEncoderInitParam encodeCLIOptions;
    CUcontext cuContext;
    unsigned long long num_frame_encode = 0;
    std::vector<std::vector<uint8_t>> vPacket;
    NvEncoderCuda* pEnc;
};

// Function declarations
bool validate_encoder_parameters(const CameraParams* camera_params, const std::string& encoder_str);

static inline void initialize_gpu_frame(FrameGPU* frame_original, const evt::CameraParams* camera_params);
static inline void initialize_gpu_debayer(Debayer* debayer, const evt::CameraParams* camera_params);
static inline void debayer_frame_gpu(const evt::CameraParams* camera_params, FrameGPU* frame_original, Debayer* debayer);
static inline void duplicate_channel_gpu(const evt::CameraParams* camera_params, FrameGPU* frame_original, Debayer* debayer);
static inline void initialize_encoder(EncoderContext* encoder, std::string encoder_str, const evt::CameraParams* camera_params);
static inline void initialize_writer(Writer* writer, const evt::CameraParams* camera_params, std::string folder_name, std::string encoder_str);

class GPUVideoEncoder : public CThreadWorker {
public:
    GPUVideoEncoder(const char* name, 
                   const evt::CameraParams* camera_params,
                   std::string encoder_setup, 
                   std::string folder_name, 
                   bool* encoder_ready_signal);
    ~GPUVideoEncoder();

    bool PushToDisplay(void* imagePtr, size_t bufferSize, int width, int height, 
                      int pixelFormat, unsigned long long timestamp, 
                      unsigned long long frame_id, uint64_t timestamp_sys);
    void ProcessOneFrame(void* f);

    bool* encoder_ready_signal;
    const evt::CameraParams* camera_params;
    unsigned char* display_buffer;
    FrameGPU frame_original;
    Debayer debayer;

    EncoderContext encoder;
    Writer writer;
    std::string encoder_setup;
    std::string folder_name;

private: 
    virtual void ThreadRunning();

    WORKER_ENTRY workerEntries[ENCODER_ENTRIES_MAX];
    WORKER_ENTRY* workerEntriesFreeQueue[ENCODER_ENTRIES_MAX];
    int workerEntriesFreeQueueCount;
};

} // namespace evt

#endif