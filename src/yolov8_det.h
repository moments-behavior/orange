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
    void copy_from_Mat(const cv::Mat &image);
    void copy_from_Mat(const cv::Mat &image, cv::Size &size);
    void preprocess_gpu(unsigned char *d_rgb);
    void letterbox(const cv::Mat &image, cv::Mat &out, cv::Size &size);
    void infer();
    void postprocess(std::vector<Object> &objs);
    static void          draw_objects(const cv::Mat&                                image,
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

    PreParam pparam;
    cudaStream_t stream = nullptr;

private:
    // device pointer for gpu preprocessing
    unsigned char *d_temp;
    unsigned char *d_boarder;
    float *d_float;
    float *d_planar;
    int img_width;
    int img_height;

    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IRuntime *runtime = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    Logger gLogger{nvinfer1::ILogger::Severity::kERROR};
};

#endif // DETECT_END2END_YOLOV8_HPP
