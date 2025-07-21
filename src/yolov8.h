#ifndef DETECT_END2END_YOLOV8_HPP
#define DETECT_END2END_YOLOV8_HPP
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "fstream"
#include <nppi.h>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>
#include <unistd.h>

class Logger : public nvinfer1::ILogger {
  public:
    nvinfer1::ILogger::Severity reportableSeverity;

    explicit Logger(nvinfer1::ILogger::Severity severity =
                        nvinfer1::ILogger::Severity::kINFO)
        : reportableSeverity(severity) {}

    void log(nvinfer1::ILogger::Severity severity,
             const char *msg) noexcept override {
        if (severity > reportableSeverity) {
            return;
        }
        switch (severity) {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case nvinfer1::ILogger::Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case nvinfer1::ILogger::Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        case nvinfer1::ILogger::Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "VERBOSE: ";
            break;
        }
        std::cerr << msg << std::endl;
    }
};

inline int get_size_by_dims(const nvinfer1::Dims &dims) {
    int size = 1;
    for (int i = 0; i < dims.nbDims; i++) {
        size *= dims.d[i];
    }
    return size;
}

inline int type_to_size(const nvinfer1::DataType &dataType) {
    switch (dataType) {
    case nvinfer1::DataType::kFLOAT:
        return 4;
    case nvinfer1::DataType::kHALF:
        return 2;
    case nvinfer1::DataType::kINT32:
        return 4;
    case nvinfer1::DataType::kINT8:
        return 1;
    case nvinfer1::DataType::kBOOL:
        return 1;
    default:
        return 4;
    }
}

inline static float clamp(float val, float min, float max) {
    return val > min ? (val < max ? val : max) : min;
}

inline bool IsPathExist(const std::string &path) {
    if (access(path.c_str(), 0) == F_OK) {
        return true;
    }
    return false;
}

inline bool IsFile(const std::string &path) {
    if (!IsPathExist(path)) {
        printf("%s:%d %s not exist\n", __FILE__, __LINE__, path.c_str());
        return false;
    }
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
}

inline bool IsFolder(const std::string &path) {
    if (!IsPathExist(path)) {
        return false;
    }
    struct stat buffer;
    return (stat(path.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode));
}

struct Binding {
    size_t size = 1;
    size_t dsize = 1;
    nvinfer1::Dims dims;
    std::string name;
};

struct Object {
    cv::Rect_<float> rect;
    int label = 0;
    float prob = 0.0;
    std::vector<float> kps;
};

struct PreParam {
    float ratio = 1.0f;
    float dw = 0.0f;
    float dh = 0.0f;
    float height = 0;
    float width = 0;
};

class YOLOv8 {
  public:
    explicit YOLOv8(const std::string &engine_file_path, int width, int height,
                    cudaStream_t stream, unsigned char *d_input_image,
                    const NppStreamContext &npp_ctx);
    ~YOLOv8();

    void make_pipe(bool graph_capture);
    void preprocess_gpu();
    void infer();
    void postprocess(std::vector<Object> &objs);
    void postprocess_kp(std::vector<Object> &objs, float score_thres,
                        float iou_thres, int topk);
    static void
    draw_objects(const cv::Mat &image, cv::Mat &res,
                 const std::vector<Object> &objs,
                 const std::vector<std::string> &CLASS_NAMES,
                 const std::vector<std::vector<unsigned int>> &COLORS);
    static void
    draw_objects_kp(const cv::Mat &image, cv::Mat &res,
                    const std::vector<Object> &objs,
                    const std::vector<std::vector<unsigned int>> &SKELETON,
                    const std::vector<std::vector<unsigned int>> &KPS_COLORS,
                    const std::vector<std::vector<unsigned int>> &LIMB_COLORS);

    void copy_keypoints_gpu(float *d_points, const std::vector<Object> &objs);
    void copy_keypoints_gpu(float *d_points, const Object &obj);
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
    NppStreamContext npp_ctx_;

    nvinfer1::ICudaEngine *engine = nullptr;
    nvinfer1::IRuntime *runtime = nullptr;
    nvinfer1::IExecutionContext *context = nullptr;
    Logger gLogger{nvinfer1::ILogger::Severity::kERROR};
};

#endif // DETECT_END2END_YOLOV8_HPP
