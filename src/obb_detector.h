#ifndef ORANGE_OBB_DETECTOR
#define ORANGE_OBB_DETECTOR

#include "camera.h"
#include "types.h"
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>

// Structure to hold oriented bounding box corners
struct OBB {
    float x1, y1, x2, y2, x3, y3, x4, y4;
    int class_id;
    float confidence;
    
    OBB() : x1(0), y1(0), x2(0), y2(0), x3(0), y3(0), x4(0), y4(0), class_id(-1), confidence(0.0f) {}
    OBB(float x1_, float y1_, float x2_, float y2_, float x3_, float y3_, float x4_, float y4_, int cls, float conf)
        : x1(x1_), y1(y1_), x2(x2_), y2(y2_), x3(x3_), y3(y3_), x4(x4_), y4(y4_), class_id(cls), confidence(conf) {}
};

// Structure to hold prior statistics for each class
struct ClassPrior {
    float area_median, area_iqr, area_lo, area_hi;
    float aspect_median, aspect_iqr, aspect_lo, aspect_hi;
    float angle_median, angle_iqr;
    
    ClassPrior() : area_median(0), area_iqr(0), area_lo(0), area_hi(0),
                   aspect_median(0), aspect_iqr(0), aspect_lo(0), aspect_hi(0),
                   angle_median(0), angle_iqr(0) {}
};

// Structure to hold detection parameters
struct OBBDetectorParams {
    int threshold = 36;           // Binary threshold for motion detection
    int frame_stride = 1;         // Process every N-th frame
    bool keep_negatives = true;   // Keep frames with no detections
    std::string bg_mode = "first"; // Background mode: "first" or "median"
    int bg_frames = 30;           // Frames for median background
    
    // Classification gates (multipliers around IQR bounds)
    float area_lo_gate = 0.9f;    // Allow slightly smaller than lo
    float area_hi_gate = 1.1f;    // Allow slightly larger than hi
    float aspect_lo_gate = 0.9f;
    float aspect_hi_gate = 1.1f;
    float angle_k_gate = 2.0f;    // Allow up to k*IQR angular deviation
};

class OBBDetector {
public:
    OBBDetector(CameraParams* params, const std::vector<std::string>& csv_paths, 
                const OBBDetectorParams& detector_params = OBBDetectorParams());
    ~OBBDetector();

    // Initialize the detector (learn priors from CSV files)
    bool initialize();
    
    // Start/stop the detection thread
    void start();
    void stop();
    
    // Called from capture thread when a frame is ready
    void notify_frame_ready(void* device_image_ptr, cudaStream_t copy_stream);
    
    // Get the latest detection results
    std::vector<OBB> get_latest_detections();
    
    // Synchronous processing method (for direct use in display thread)
    std::vector<OBB> process_frame_sync(const cv::Mat& frame);
    
    // Check if detector is running
    bool is_running() const { return running.load(); }
    
    // Static function to draw oriented bounding boxes on image
    static void draw_obb_objects(const cv::Mat& image, cv::Mat& res,
                                const std::vector<OBB>& obbs,
                                const std::vector<std::string>& class_names,
                                const std::vector<std::vector<unsigned int>>& colors,
                                int line_thickness = 2);
    
    // Public utility functions for testing
    static std::vector<cv::Point2f> order_corners_clockwise_start_tl(const std::vector<cv::Point2f>& pts);
    static cv::Point2f compute_centroid(const std::vector<cv::Point2f>& points);
    
    // Public classification function for testing
    int classify_with_priors(const std::vector<cv::Point2f>& points);
    
    // Public geometry functions for testing
    std::pair<float, float> rect_wh_from_pts(const std::vector<cv::Point2f>& points);
    float long_axis_angle_deg(const std::vector<cv::Point2f>& points);
    
    // Public detection function for testing
    std::vector<std::vector<cv::Point2f>> detect_candidates(const cv::Mat& frame, const cv::Mat& background);
    
    // Debug function to print learned priors
    void print_priors();
    
    // Getter for priors (for testing)
    const std::map<int, ClassPrior>& get_priors() const { return priors; }

private:
    // Thread function
    void thread_loop();
    
    // CSV parsing and prior learning
    bool parse_labels_csv(const std::string& csv_path, std::vector<std::pair<int, std::vector<cv::Point2f>>>& labels);
    void learn_priors_from_csvs();
    ClassPrior compute_class_prior(const std::vector<std::vector<cv::Point2f>>& class_samples);
    
    // Geometry utilities
    float circular_diff(float a, float b);
    
    // Background modeling
    bool build_background(cv::VideoCapture& cap, cv::Mat& background);
    
    // Motion detection and OBB generation
    cv::Mat create_disk_kernel(int radius);
    
    // Classification
    float compute_classification_score(const std::vector<cv::Point2f>& points, int class_id);
    
    // CUDA utilities
    void copy_frame_to_cpu(void* device_ptr, cv::Mat& cpu_frame);
    
    // Member variables
    CameraParams* camera_params;
    OBBDetectorParams params;
    std::vector<std::string> csv_paths;
    
    // Prior data
    std::map<int, ClassPrior> priors;
    
    // Background model
    cv::Mat background_model;
    bool background_initialized;
    
    // Threading
    std::atomic<bool> running;
    std::atomic<bool> frame_ready;
    std::mutex mtx;
    std::condition_variable cv;
    std::thread worker_thread;
    
    // CUDA resources
    cudaStream_t stream;
    cudaEvent_t copy_done_event;
    
    // Frame processing (simplified)
    unsigned char* d_frame_original;
    unsigned char* d_debayer;
    unsigned char* h_frame_cpu;
    
    // Detection results
    std::vector<OBB> latest_detections;
    std::mutex detections_mtx;
    
    // Statistics
    std::atomic<uint64_t> frames_processed;
    std::atomic<uint64_t> detections_found;
};

#endif // ORANGE_OBB_DETECTOR
