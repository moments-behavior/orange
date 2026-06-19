#ifndef ORANGE_GPU_VIDEO_ENCODER
#define ORANGE_GPU_VIDEO_ENCODER
#include "FFmpegWriter.h"
#include "NvEncoder/NvEncoderCLIOptions.h"
#include "NvEncoder/NvEncoderCuda.h"
#include "image_processing.h"
#include "threadworker.h"
#include "video_capture.h"

#define ENCODER_ENTRIES_MAX 20

struct Writer {
    std::string video_file;
    std::string keyframe_file;
    std::string metadata_file;
    FFmpegWriter *video;
    std::ofstream *metadata;
};

struct EncoderContext {
    NV_ENC_BUFFER_FORMAT eFormat;
    NvEncoderInitParam encodeCLIOptions;
    CUcontext cuContext;
    unsigned long long num_frame_encode = 0;
    std::vector<std::vector<uint8_t>> vPacket;
    NvEncoderCuda *pEnc;
};

class GPUVideoEncoder : public CThreadWorker {
  public:
    GPUVideoEncoder(const char *name, CameraParams *camera_params,
                    CameraEachSelect *camera_select, std::string encoder_setup,
                    std::string folder_name,
                    bool *encoder_ready_signal); // name is the thread name
    ~GPUVideoEncoder();

    bool PushToDisplay(void *imagePtr, size_t bufferSize, int width, int height,
                       int pixelFormat, unsigned long long timestamp,
                       unsigned long long frame_id, uint64_t timestamp_sys,
                       int ptp_offset);
    void ProcessOneFrame(void *f);

    // open gl dimensions:
    CameraParams *camera_params;
    CameraEachSelect *camera_select;
    unsigned char *display_buffer;
    FrameGPU frame_original; // frame on gpu device
    Debayer debayer;
    NppStreamContext npp_ctx; // built once in ThreadRunning (CUDA 13 NPP needs a stream ctx)

    // Average-brightness sampling. Computed on a dedicated stream (never the
    // encoder's default stream), throttled to ~10 Hz and subsampled 1/stride^2,
    // so it can't disturb encode timing. Published to g_cam_brightness[].
    cudaStream_t bright_stream = nullptr;
    unsigned long long *d_bright_sum = nullptr; // device scalar accumulator
    unsigned long long *h_bright_sum = nullptr; // pinned host readback
    double bright_next_sample_time = 0.0; // wall-clock throttle (steady seconds)
    int bright_stride = 8;         // subsample every 8th pixel in x & y
    long bright_sample_count = 1;  // pixels summed per sample (for the mean)

    // encoding
    EncoderContext encoder;
    Writer writer;
    std::string encoder_setup;
    std::string folder_name;
    bool *encoder_ready_signal;

  private:
    virtual void
    ThreadRunning(); // overides of COffThreadMachine for worker thread
  private:
    WORKER_ENTRY workerEntries[ENCODER_ENTRIES_MAX];
    WORKER_ENTRY *workerEntriesFreeQueue[ENCODER_ENTRIES_MAX];
    int workerEntriesFreeQueueCount;
};

#endif
