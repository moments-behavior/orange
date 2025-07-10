// optimized_yolo_preprocess.h
#ifndef OPTIMIZED_YOLO_PREPROCESS_H
#define OPTIMIZED_YOLO_PREPROCESS_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Optimized single-kernel YOLO preprocessing
 * Replaces the entire pipeline: mono/bayer -> resize -> letterbox -> normalize -> planar
 * 
 * @param d_src: Source image data (mono or bayer pattern)
 * @param d_dst_planar: Output planar float data (B,G,R channels sequentially)
 * @param src_width: Source image width (e.g. 4512)
 * @param src_height: Source image height (e.g. 4512)
 * @param dst_width: Target width (e.g. 640)
 * @param dst_height: Target height (e.g. 640)
 * @param is_color: true for bayer/color, false for mono
 * @param stream: CUDA stream to use
 */
void launch_optimized_yolo_preprocess(
    const unsigned char* d_src,
    float* d_dst_planar,
    int src_width, int src_height,
    int dst_width, int dst_height,
    bool is_color,
    cudaStream_t stream);

/**
 * Calculate output buffer size needed
 */
size_t get_optimized_output_size(int dst_width, int dst_height);

#ifdef __cplusplus
}
#endif

#endif // OPTIMIZED_YOLO_PREPROCESS_H