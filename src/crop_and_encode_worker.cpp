#include "crop_and_encode_worker.h"
#include "kernel.cuh"
#include <nppi.h>

CropAndEncodeWorker::CropAndEncodeWorker(const char* name, CameraParams* camera_params, const std::string& folder_name, SafeQueue<WORKER_ENTRY*>& recycle_queue)
    : CThreadWorker(name), camera_params_(camera_params), folder_name_(folder_name), m_recycle_queue(recycle_queue) {

    // Initialize the encoder and other resources here
    // For simplicity, we'll use H.264 with a fixed preset
    writer_.video_file = folder_name_ + "/Cam" + camera_params_->camera_serial + "_crop.mp4";
    writer_.video = new FFmpegWriter(AV_CODEC_ID_H264, 640, 640, camera_params_->frame_rate, writer_.video_file.c_str(), nullptr);
    writer_.video->create_thread();

    ck(cudaSetDevice(camera_params_->gpu_id));
    ck(cudaMalloc(&d_cropped_bgr_, 640 * 640 * 3)); // BGR buffer for the crop
    ck(cudaMalloc(&d_yuv_buffer_, 640 * 640 * 3 / 2)); // YUV buffer for encoding
}

CropAndEncodeWorker::~CropAndEncodeWorker() {
    if (writer_.video) {
        writer_.video->quit_thread();
        writer_.video->join_thread();
        delete writer_.video;
    }
    if (d_cropped_bgr_) cudaFree(d_cropped_bgr_);
    if (d_yuv_buffer_) cudaFree(d_yuv_buffer_);
    if (encoder_) delete encoder_;
}

bool CropAndEncodeWorker::WorkerFunction(WORKER_ENTRY* entry) {
    if (!entry || entry->detections.empty()) {
        if (entry) {
            if (entry->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                m_recycle_queue.push(entry);
            }
        }
        return false;
    }

    // Find the detection with the highest confidence
    const auto& best_detection = *std::max_element(entry->detections.begin(), entry->detections.end(),
        [](const pose::Object& a, const pose::Object& b) {
            return a.prob < b.prob;
        });

    // Perform the crop and resize using a new CUDA kernel
    gpu_crop_and_resize(entry->d_image, d_cropped_bgr_, entry->width, entry->height, best_detection.rect, 640, 640, nullptr);
    
    // Now, we would encode d_cropped_bgr_ to a video file.
    // This involves converting BGR to YUV and then using NvEncoder, which is a complex process.
    // For now, we will placeholder this logic.
    
    if (entry->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        m_recycle_queue.push(entry);
    }
    return false;
}