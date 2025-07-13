#ifndef DETECT_END2END_YOLOV8_HPP
#define DETECT_END2END_YOLOV8_HPP
#include "NvInferPlugin.h"
#include "common.hpp"
#include "fstream"
#include <nppi.h>

class YOLOv8 {
  public:
    explicit YOLOv8(const std::string &engine_file_path, int width, int height,
                    unsigned char *d_input_image, bool use_external_stream,
                    cudaStream_t stream);
    ~YOLOv8();

    void make_pipe(bool warmup = true);
    void preprocess_gpu();
    void infer();
    void postprocess(std::vector<Bbox> &objs);
    static void
    draw_objects(const cv::Mat &image, cv::Mat &res,
                 const std::vector<Bbox> &objs,
                 const std::vector<std::string> &CLASS_NAMES,
                 const std::vector<std::vector<unsigned int>> &COLORS);

    void copy_keypoints_gpu(float *d_points, const std::vector<Bbox> &objs);
    void copy_keypoints_gpu(float *d_points, const Bbox &obj);
    void infer_capture_only();

    int num_bindings;
    int num_inputs = 0;
    int num_outputs = 0;
    std::vector<Binding> input_bindings;
    std::vector<Binding> output_bindings;
    std::vector<void *> host_ptrs;
    std::vector<void *> device_ptrs;

    PreParam pparam;
    cudaStream_t stream = nullptr;
    cudaGraph_t inference_graph = nullptr;
    cudaGraphExec_t inference_graph_exec = nullptr;
    bool graph_captured = false;

  private:
    // device pointer for gpu preprocessing
    unsigned char *d_input_image;
    unsigned char *d_temp;
    unsigned char *d_boarder;
    float *d_float;
    float *d_planar;
    int img_width;
    int img_height;
    int padw;
    int padh;
    int inp_h_int;
    int inp_w_int;
    bool use_external_stream;

    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IRuntime *runtime = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    Logger gLogger{nvinfer1::ILogger::Severity::kERROR};
};

#endif // DETECT_END2END_YOLOV8_HPP
