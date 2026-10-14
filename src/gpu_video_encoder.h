#ifndef ORANGE_GPU_VIDEO_ENCODER
#define ORANGE_GPU_VIDEO_ENCODER
#include "FFmpegWriter.h"
#include "NvEncoder/NvEncoderCLIOptions.h"
#include "NvEncoder/NvEncoderCuda.h"
#include "image_processing.h"
#include "recording_chunks.h"
#include "threadworker.h"
#include "video_capture.h"
#include <deque>

#define ENCODER_ENTRIES_MAX 20

// One output file set (mp4 + metadata csv). Without chunking there is a
// single Writer for the whole recording, in the recording folder itself; with
// chunking there is one per chunk, in <recording folder>/chunk_NNNNN/.
struct Writer {
    int chunk = 0;
    std::string video_file;
    std::string keyframe_file;
    std::string metadata_file;
    FFmpegWriter *video = nullptr;
    std::ofstream *metadata = nullptr;
    unsigned long long packets = 0; // packets muxed so far (pts within file)
    CameraChunkSummary summary;
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
                    std::string folder_name, bool *encoder_ready_signal,
                    double chunk_seconds = 0.0,
                    PTPParams *ptp_params = nullptr); // name is the thread name
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
    std::string encoder_setup;
    std::string folder_name;
    bool *encoder_ready_signal;

    // Chunked recording (see recording_chunks.h). NVENC returns each frame's
    // packet a few frames after the frame is submitted, so metadata rows and
    // packets are routed separately: rows go to the chunk of the frame being
    // submitted (row_writer), packets to the chunk of the frame they encode
    // (packet_writer, via pending_frames). A finished chunk's files are closed
    // on the finalizer thread so the mp4 trailer never stalls encoding.
    double chunk_seconds = 0.0;
    PTPParams *ptp_params = nullptr;
    ChunkClock chunk_clock;
    bool chunk_first_frame_logged = false;
    Writer *row_writer = nullptr;
    Writer *packet_writer = nullptr;
    std::deque<Writer *> pending_frames;
    ChunkFinalizer *finalizer = nullptr;

  private:
    virtual void
    ThreadRunning(); // overides of COffThreadMachine for worker thread
    Writer *open_writer(int chunk);
    void route_packets();
    void retire_writer(Writer *w);
  private:
    WORKER_ENTRY workerEntries[ENCODER_ENTRIES_MAX];
    WORKER_ENTRY *workerEntriesFreeQueue[ENCODER_ENTRIES_MAX];
    int workerEntriesFreeQueueCount;
};

#endif
