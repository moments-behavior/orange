#ifndef ORANGE_OBB_DETECTOR
#define ORANGE_OBB_DETECTOR

#include "camera.h"
#include "common.hpp"
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

// Forward-declare YOLO bounding box (defined in common.hpp)
struct Bbox;

struct OBB {
    float x1, y1, x2, y2, x3, y3, x4, y4;
    int class_id;
    float confidence;
    int object_id;
    bool shape_verified;
    
    OBB() : x1(0), y1(0), x2(0), y2(0), x3(0), y3(0), x4(0), y4(0), class_id(-1), confidence(0.0f), object_id(-1), shape_verified(false) {}
    OBB(float x1_, float y1_, float x2_, float y2_, float x3_, float y3_, float x4_, float y4_, int cls, float conf, int obj_id = -1, bool verified = false)
        : x1(x1_), y1(y1_), x2(x2_), y2(y2_), x3(x3_), y3(y3_), x4(x4_), y4(y4_), class_id(cls), confidence(conf), object_id(obj_id), shape_verified(verified) {}
};

struct ClassPrior {
    float area_median, area_iqr, area_lo, area_hi;
    float aspect_median, aspect_iqr, aspect_lo, aspect_hi;
    float angle_median, angle_iqr;
    float width_median, height_median;
    
    ClassPrior() : area_median(0), area_iqr(0), area_lo(0), area_hi(0),
                   aspect_median(0), aspect_iqr(0), aspect_lo(0), aspect_hi(0),
                   angle_median(0), angle_iqr(0),
                   width_median(0), height_median(0) {}
};

struct OBBDetectorParams {
    // Classification gates (multipliers around IQR bounds)
    float area_lo_gate = 0.9f;
    float area_hi_gate = 1.1f;
    float aspect_lo_gate = 0.9f;
    float aspect_hi_gate = 1.1f;
    float angle_k_gate = 2.0f;
};

class OBBDetector {
public:
    OBBDetector(CameraParams* params, const std::vector<std::string>& csv_paths, 
                const OBBDetectorParams& detector_params = OBBDetectorParams());
    ~OBBDetector();

    bool initialize();
    
    void start();
    void stop();
    
    // Called from capture thread when a frame is ready on GPU
    void notify_frame_ready(void* device_image_ptr, cudaStream_t copy_stream);
    
    // Supply YOLO axis-aligned boxes for two-stage refinement.
    // Must be called every frame (even with empty vector) so the
    // detector knows YOLO ran.
    void set_yolo_boxes(const std::vector<Bbox>& boxes);
    
    std::vector<OBB> get_latest_detections();
    
    bool is_running() const { return running.load(); }
    
    static void draw_obb_objects(const cv::Mat& image, cv::Mat& res,
                                const std::vector<OBB>& obbs,
                                const std::vector<std::string>& class_names,
                                const std::vector<std::vector<unsigned int>>& colors,
                                int line_thickness = 2);
    
    static std::vector<cv::Point2f> order_corners_clockwise_start_tl(const std::vector<cv::Point2f>& pts);
    static cv::Point2f compute_centroid(const std::vector<cv::Point2f>& points);
    
    int classify_with_priors(const std::vector<cv::Point2f>& points);
    std::pair<float, float> rect_wh_from_pts(const std::vector<cv::Point2f>& points);
    float long_axis_angle_deg(const std::vector<cv::Point2f>& points);
    
    struct XYWHR {
        float x, y, w, h, r;
    };
    XYWHR obb_to_xywhr(const OBB& obb);
    
    // Two-stage: YOLO box center + CSV prior size + local mask angle → OBB
    std::vector<OBB> refine_yolo_detections(const cv::Mat& frame,
                                            const std::vector<Bbox>& yolo_boxes,
                                            int target_class_id = 2);

    // Seg-based: reconstruct mask from coefficients + prototypes → fit OBB
    // pparam carries the letterbox preprocessing info (ratio, dw, dh)
    std::vector<OBB> refine_from_seg_masks(const std::vector<Bbox>& yolo_boxes,
                                           const float* mask_protos,
                                           int proto_h, int proto_w, int num_protos,
                                           const PreParam& pparam,
                                           int target_class_id = 2);
    
    bool should_update_detections(const std::vector<OBB>& new_detections);
    
    void print_priors();
    const std::map<int, ClassPrior>& get_priors() const { return priors; }

private:
    void thread_loop();
    
    // CSV parsing and prior learning
    bool parse_labels_csv(const std::string& csv_path, std::vector<std::pair<int, std::vector<cv::Point2f>>>& labels);
    void learn_priors_from_csvs();
    ClassPrior compute_class_prior(const std::vector<std::vector<cv::Point2f>>& class_samples);
    
    float circular_diff(float a, float b);
    float compute_classification_score(const std::vector<cv::Point2f>& points, int class_id);
    
    // Local angle extraction for two-stage refinement
    float extract_angle_from_local_mask(const cv::Mat& frame,
                                        float cx, float cy,
                                        float crop_w, float crop_h,
                                        float pad_factor = 2.0f);
    float extract_angle_and_rect(const cv::Mat& frame,
                                 float cx, float cy,
                                 float crop_w, float crop_h,
                                 float pad_factor,
                                 cv::RotatedRect& out_rect);
    float extract_angle_hough(const cv::Mat& frame,
                              float cx, float cy,
                              float crop_w, float crop_h,
                              float pad_factor = 2.0f);

    // Angle smoothing helpers
    float smooth_angle(float cx, float cy, float raw_angle);
    int find_nearest_track(float cx, float cy);

    void copy_frame_to_cpu(void* device_ptr, cv::Mat& cpu_frame);
    
    CameraParams* camera_params;
    OBBDetectorParams params;
    std::vector<std::string> csv_paths;
    
    std::map<int, ClassPrior> priors;
    
    // Threading
    std::atomic<bool> running;
    std::atomic<bool> frame_ready;
    std::mutex mtx;
    std::condition_variable cv;
    std::thread worker_thread;
    
    // CUDA
    cudaStream_t stream;
    cudaEvent_t copy_done_event;
    void* d_frame_original;
    unsigned char* h_frame_cpu;
    
    // Detection results
    std::vector<OBB> latest_detections;
    std::mutex detections_mtx;
    
    std::atomic<uint64_t> frames_processed;
    std::atomic<uint64_t> detections_found;
    
    // Stable detection tracking
    std::vector<OBB> stable_detections;
    bool detections_stable;
    int frames_since_change;
    
    // YOLO boxes from external detector
    std::vector<Bbox> yolo_boxes_pending;
    bool yolo_has_update;
    std::mutex yolo_mtx;

    // Per-object angle smoothing (EMA)
    struct AngleTrack {
        float cx, cy;
        float smoothed_angle;
        int age;  // frames since last match
    };
    std::vector<AngleTrack> angle_tracks;
    static constexpr float ANGLE_EMA_ALPHA = 0.5f;   // weight of new measurement
    static constexpr float TRACK_MATCH_DIST = 80.0f;  // max pixels to match
    static constexpr int   TRACK_MAX_AGE = 30;         // drop after N unmatched frames
};

#endif // ORANGE_OBB_DETECTOR
