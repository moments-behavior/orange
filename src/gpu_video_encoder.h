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
#include <chrono> // For FPS tracking
#include <cuda.h> // For CUcontext

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
    GPUVideoEncoder(const char* name, CUcontext cuda_context, CameraParams *camera_params,
        const std::string& codec, const std::string& preset, const std::string& tuning,
        std::string folder_name, bool* encoder_ready_signal,
        SafeQueue<WORKER_ENTRY*>& recycle_queue);
    ~GPUVideoEncoder() override;

    double get_fps() const {
        return current_fps_;
    }

    bool* encoder_ready_signal;

protected:
    bool WorkerFunction(WORKER_ENTRY* f) override;

private:
    CameraParams* camera_params;
    std::string folder_name;

    FrameGPU frame_original;
    Debayer debayer;
    EncoderContext encoder;
    Writer writer;
	CUcontext m_cuContext; // Store the CUDA context
    cudaStream_t m_stream; // Dedicated stream for this worker
	int encoder_pitch_;
    int scaled_width_;
    int scaled_height_;
    unsigned char* d_scaled_mono_buffer_;

    // Intermediate GPU buffers for conversion
    unsigned char* d_rgb_temp_;
    unsigned char* d_iyuv_temp_; // Single buffer for the 3-plane IYUV data
    unsigned char* d_uv_default_plane_; // Pre-filled buffer for monochrome U/V planes

    SafeQueue<WORKER_ENTRY*>& m_recycle_queue;

    // FPS tracking
    std::chrono::steady_clock::time_point last_fps_update_time_;
    int frame_counter_;
    double current_fps_;

    // Debug helper functions
    void DumpYUVFrame(const char* filename, unsigned char* d_yuv_data, int width, int height);
    void TestPattern(unsigned char* d_iyuv, int width, int height);
};

#endif // ORANGE_GPU_VIDEO_ENCODER