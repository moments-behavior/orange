// src/gpu_video_encoder.h

#ifndef ORANGE_GPU_VIDEO_ENCODER
#define ORANGE_GPU_VIDEO_ENCODER

#include "threadworker.h"
#include "video_capture.h"
#include "FFmpegWriter.h"
#include "NvEncoder/NvEncoderCuda.h"
#include "NvEncoder/NvEncoderCLIOptions.h"
#include "image_processing.h"
#include "thread.h" // For SafeQueue

struct Writer
{
    std::string video_file;
    std::string keyframe_file;
    std::string metadata_file;
    FFmpegWriter *video;
    std::ofstream* metadata;
};

struct EncoderContext
{
    NV_ENC_BUFFER_FORMAT eFormat;
    NvEncoderInitParam encodeCLIOptions;
    CUcontext cuContext;
    unsigned long long num_frame_encode = 0;
    std::vector<std::vector<uint8_t>> vPacket;
    NvEncoderCuda *pEnc;
};

class GPUVideoEncoder : public CThreadWorker<WORKER_ENTRY>
{
public:
    // --- Constructor now requires the recycle_queue ---
    GPUVideoEncoder(const char* name, CameraParams *camera_params, std::string encoder_setup, std::string folder_name, bool* encoder_ready_signal, SafeQueue<WORKER_ENTRY*>& recycle_queue);
    ~GPUVideoEncoder() override;

    // Public members
	bool* encoder_ready_signal;
	CameraParams* camera_params;
	FrameGPU frame_original;
	Debayer debayer;
	EncoderContext encoder;
    Writer writer;
	std::string encoder_setup;
	std::string folder_name;

protected:
    bool WorkerFunction(WORKER_ENTRY* f) override;

private:
    SafeQueue<WORKER_ENTRY*>& m_recycle_queue;
};


#endif