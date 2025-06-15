// src/yolov8_det.h
#ifndef DETECT_END2END_YOLOV8_HPP
#define DETECT_END2END_YOLOV8_HPP
#include "NvInferPlugin.h"
#include "common.hpp"
#include "fstream"
#include <nppi.h>

using namespace pose;

class YOLOv8
{
public:
    explicit YOLOv8(const std::string &engine_file_path, int width, int height);
    ~YOLOv8();

    void make_pipe(bool warmup = true);
    void preprocess_gpu(unsigned char *d_rgb, int source_width, int source_height);
    void infer();
    void postprocess(std::vector<Object> &objs);
    static void draw_objects(const cv::Mat&                                image,
                           cv::Mat&                                      res,
                           const std::vector<Object>&                    objs,
                           const std::vector<std::string>&               CLASS_NAMES,
                           const std::vector<std::vector<unsigned int>>& COLORS);

    void copy_keypoints_gpu(float* d_points, const std::vector<Object>& objs);
    
    int num_bindings;
    int num_inputs = 0;
    int num_outputs = 0;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void *> host_ptrs;
    std::vector<void *> device_ptrs;

    // --- MOVED from private to public ---
    int inp_h_int;
    int inp_w_int;
    // --- END MOVE ---

    PreParam pparam;
    cudaStream_t stream = nullptr;

private:
    unsigned char *d_temp = nullptr;
    unsigned char *d_boarder = nullptr;
    float *d_float = nullptr;
    float *d_planar = nullptr;
    int img_width;
    int img_height;
    int padw;
    int padh;
    // int inp_h_int; // Now public
    // int inp_w_int; // Now public

    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IRuntime *runtime = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    Logger gLogger{nvinfer1::ILogger::Severity::kERROR};
};

#endif // DETECT_END2END_YOLOV8_HPP