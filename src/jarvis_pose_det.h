#ifndef JARVIS_POSE_DET_H
#define JARVIS_POSE_DET_H

#include "NvInferPlugin.h"
#include "common.hpp"
#include "fstream"
#include <nppi.h>
#include <opencv2/opencv.hpp>

class JarvisPoseDetector {
public:
    explicit JarvisPoseDetector(const std::string &model_dir,
                               int num_cameras,
                               cudaStream_t stream);
    ~JarvisPoseDetector();

    // Three-stage pipeline
    void detect_centers(unsigned char **d_input_images, int img_width, int img_height);
    void detect_keypoints(unsigned char **d_input_images, int img_width, int img_height);
    void reconstruct_3d_pose();
    
    // Get results
    void get_center_results(std::vector<cv::Point2f> &centers, std::vector<float> &confidences);
    void get_keypoint_results(std::vector<cv::Point2f> &keypoints, std::vector<float> &confidences);
    void get_3d_center_result(cv::Point3f &center_3d, float &confidence);
    void get_3d_pose_results(std::vector<cv::Point3f> &keypoints_3d, std::vector<float> &confidences);

    // Model management
    void make_pipe(bool graph_capture);
    void load_models(const std::string &model_dir);

private:
    // Three separate TensorRT engines
    nvinfer1::ICudaEngine *center_engine = nullptr;
    nvinfer1::ICudaEngine *keypoint_engine = nullptr;
    nvinfer1::ICudaEngine *hybrid_net_engine = nullptr;
    
    nvinfer1::IRuntime *runtime = nullptr;
    nvinfer1::IExecutionContext *center_context = nullptr;
    nvinfer1::IExecutionContext *keypoint_context = nullptr;
    nvinfer1::IExecutionContext *hybrid_net_context = nullptr;
    
    Logger gLogger{nvinfer1::ILogger::Severity::kERROR};
    
    // Configuration
    int num_cameras;
    cudaStream_t stream;
    
    // Device memory for all three stages
    std::vector<void *> center_device_ptrs;
    std::vector<void *> center_host_ptrs;
    std::vector<void *> keypoint_device_ptrs;
    std::vector<void *> keypoint_host_ptrs;
    std::vector<void *> hybrid_net_device_ptrs;
    std::vector<void *> hybrid_net_host_ptrs;
    
    // Preprocessing buffers
    unsigned char *d_resized_images;
    unsigned char *d_cropped_images;
    float *d_float_images;
    
    // Results storage
    std::vector<cv::Point2f> center_results;
    std::vector<float> center_confidences;
    std::vector<cv::Point2f> keypoint_results;
    std::vector<float> keypoint_confidences;
    cv::Point3f center_3d_result;
    float center_3d_confidence;
    std::vector<cv::Point3f> pose_3d_results;
    std::vector<float> pose_3d_confidences;
    
    // Camera calibration data (loaded from model_info.json)
    std::vector<cv::Mat> camera_matrices;
    std::vector<cv::Mat> intrinsic_matrices;
    std::vector<cv::Mat> distortion_coefficients;
    
    // Model configuration
    int center_img_size;
    int keypoint_img_size;
    int num_keypoints;
    std::vector<float> mean_values;
    std::vector<float> std_values;
    
    // Helper functions
    void preprocess_center_detection(unsigned char **d_input_images, int img_width, int img_height);
    void preprocess_keypoint_detection(unsigned char **d_input_images, int img_width, int img_height);
    void postprocess_center_detection();
    void postprocess_keypoint_detection();
    void postprocess_3d_reconstruction();
    void triangulate_3d_center();
    void crop_keypoint_regions();
    void load_model_info(const std::string &model_dir);
    void project_3d_to_2d(const cv::Point3f &point_3d, int camera_id, cv::Point2f &point_2d);
};

#endif // JARVIS_POSE_DET_H
