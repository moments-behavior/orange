#ifndef CROP_AND_ENCODE_WORKER_H
#define CROP_AND_ENCODE_WORKER_H

#include "threadworker.h"
#include "video_capture.h"
#include "gpu_video_encoder.h" // For Writer struct
#include "FFmpegWriter.h"
#include "NvEncoder/NvEncoderCuda.h"
#include "image_processing.h"
#include <chrono>
#include <fstream>

class CropAndEncodeWorker : public CThreadWorker<WORKER_ENTRY> {
public:
    CropAndEncodeWorker(const char* name, CameraParams *camera_params, const std::string& folder_name, SafeQueue<WORKER_ENTRY*>& recycle_queue);
    ~CropAndEncodeWorker() override;

protected:
    bool WorkerFunction(WORKER_ENTRY* f) override;

private:
    CameraParams* camera_params_;
    std::string folder_name_;
    Writer writer_;
    NvEncoderCuda* encoder_ = nullptr;
    unsigned char* d_cropped_bgr_ = nullptr;
    unsigned char* d_yuv_buffer_ = nullptr;
    int encoder_pitch_ = 0;
    cudaStream_t m_stream = nullptr;
    int frame_counter_ = 0;
    SafeQueue<WORKER_ENTRY*>& m_recycle_queue;
    Debayer debayer_gpu_;
};

#endif // CROP_AND_ENCODE_WORKER_H