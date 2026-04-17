#include "obb_detector.h"
#include "common.hpp"
#include "kernel.cuh"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        throw std::runtime_error("CUDA error"); \
    } \
} while(0)

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

OBBDetector::OBBDetector(CameraParams* params, const std::vector<std::string>& csv_paths, 
                         const OBBDetectorParams& detector_params)
    : camera_params(params), csv_paths(csv_paths), params(detector_params),
      running(false), frame_ready(false), frames_processed(0), detections_found(0),
      d_frame_original(nullptr), h_frame_cpu(nullptr),
      detections_stable(false), frames_since_change(0),
      yolo_has_update(false) {
    
    CUDA_CHECK(cudaSetDevice(camera_params->gpu_id));
    stream = nullptr;
    cudaEventCreateWithFlags(&copy_done_event, cudaEventDisableTiming);
}

OBBDetector::~OBBDetector() {
    stop();
    if (h_frame_cpu) {
        cudaFreeHost(h_frame_cpu);
    }
    cudaEventDestroy(copy_done_event);
    if (stream) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
}

bool OBBDetector::initialize() {
    try {
        learn_priors_from_csvs();
        
        if (priors.empty()) {
            std::cerr << "No priors learned from CSV files" << std::endl;
            return false;
        }
        
        // Priors loaded successfully
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize OBBDetector: " << e.what() << std::endl;
        return false;
    }
}

void OBBDetector::start() {
    if (running.load()) return;
    running.store(true);
    worker_thread = std::thread(&OBBDetector::thread_loop, this);
}

void OBBDetector::stop() {
    if (running.load()) {
        running.store(false);
        cv.notify_all();
        if (worker_thread.joinable()) worker_thread.join();
    }
}

// ---------------------------------------------------------------------------
// Frame / YOLO interface
// ---------------------------------------------------------------------------

void OBBDetector::notify_frame_ready(void* device_image_ptr, cudaStream_t copy_stream) {
    if (!running.load()) return;
    d_frame_original = device_image_ptr;
    cudaEventRecord(copy_done_event, copy_stream);
    frame_ready = true;
    cv.notify_one();
}

void OBBDetector::set_yolo_boxes(const std::vector<Bbox>& boxes) {
    std::lock_guard<std::mutex> lock(yolo_mtx);
    yolo_boxes_pending = boxes;
    yolo_has_update = true;
}

// ---------------------------------------------------------------------------
// Worker thread -- pure YOLO+OBB refinement, no background subtraction
// ---------------------------------------------------------------------------

void OBBDetector::thread_loop() {
    CUDA_CHECK(cudaSetDevice(camera_params->gpu_id));
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    size_t cpu_frame_size = camera_params->width * camera_params->height * 3 * sizeof(unsigned char);
    CUDA_CHECK(cudaMallocHost(&h_frame_cpu, cpu_frame_size));
    
    
    
    while (running.load()) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] { return !running.load() || frame_ready; });
        if (!running.load()) break;
        
        frame_ready = false;
        lock.unlock();
        
        cudaStreamWaitEvent(stream, copy_done_event, 0);
        if (!running.load() || !d_frame_original) continue;
        
        // Consume YOLO boxes
        std::vector<Bbox> boxes;
        bool got_update = false;
        {
            std::lock_guard<std::mutex> ylock(yolo_mtx);
            if (yolo_has_update) {
                boxes = std::move(yolo_boxes_pending);
                yolo_boxes_pending.clear();
                yolo_has_update = false;
                got_update = true;
            }
        }
        
        if (!got_update) {
            frames_processed++;
            continue;
        }
        
        // Copy frame to CPU only when we have boxes to refine
        std::vector<OBB> detections;
        if (!boxes.empty()) {
            cv::Mat frame;
            copy_frame_to_cpu(d_frame_original, frame);
            if (!frame.empty()) {
                detections = refine_yolo_detections(frame, boxes);
            }
        }
        
        if (should_update_detections(detections)) {
            {
                std::lock_guard<std::mutex> dlock(detections_mtx);
                latest_detections = detections;
                stable_detections = detections;
                detections_stable = true;
                frames_since_change = 0;
            }
            
        } else {
            std::lock_guard<std::mutex> dlock(detections_mtx);
            latest_detections = stable_detections;
            frames_since_change++;
        }
        
        frames_processed++;
        detections_found += detections.size();
    }
    
    
}

// ---------------------------------------------------------------------------
// CSV parsing & prior learning
// ---------------------------------------------------------------------------

bool OBBDetector::parse_labels_csv(const std::string& csv_path, 
                                   std::vector<std::pair<int, std::vector<cv::Point2f>>>& labels) {
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open CSV file: " << csv_path << std::endl;
        return false;
    }
    
    std::string line;
    bool header_found = false;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        if (!header_found) {
            if (line.find("frame") != std::string::npos || 
                line.find("OrientedBoundingBox") != std::string::npos) {
                header_found = true;
                continue;
            }
        }
        
        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(iss, token, ',')) tokens.push_back(token);
        
        if (tokens.size() < 11) continue;
        
        try {
            int class_id = std::stoi(tokens[2]);
            std::vector<cv::Point2f> points;
            for (int i = 0; i < 4; i++) {
                float x = std::stof(tokens[3 + i * 2]);
                float y = std::stof(tokens[4 + i * 2]);
                points.emplace_back(x, y);
            }
            labels.emplace_back(class_id, points);
        } catch (const std::exception& e) {
            continue;
        }
    }
    return true;
}

void OBBDetector::learn_priors_from_csvs() {
    std::map<int, std::vector<std::vector<cv::Point2f>>> class_samples;
    
    for (const auto& csv_path : csv_paths) {
        std::vector<std::pair<int, std::vector<cv::Point2f>>> labels;
        if (parse_labels_csv(csv_path, labels)) {
            for (const auto& label : labels)
                class_samples[label.first].push_back(label.second);
        }
    }
    
    for (const auto& class_data : class_samples) {
        priors[class_data.first] = compute_class_prior(class_data.second);
    }
}

ClassPrior OBBDetector::compute_class_prior(const std::vector<std::vector<cv::Point2f>>& class_samples) {
    ClassPrior prior;
    if (class_samples.empty()) return prior;
    
    std::vector<float> areas, aspects, angles, widths, heights;
    
    for (const auto& sample : class_samples) {
        auto wh = rect_wh_from_pts(sample);
        float area = wh.first * wh.second;
        float aspect = wh.first / std::max(wh.second, 1e-6f);
        float angle = long_axis_angle_deg(sample);
        areas.push_back(area);
        aspects.push_back(aspect);
        angles.push_back(angle);
        widths.push_back(wh.first);
        heights.push_back(wh.second);
    }
    
    std::sort(areas.begin(), areas.end());
    std::sort(aspects.begin(), aspects.end());
    
    size_t n = areas.size();
    size_t q1 = (size_t)(n * 0.25f);
    size_t q3 = std::min((size_t)(n * 0.75f), n - 1);
    
    prior.area_median = areas[n / 2];
    prior.area_iqr = areas[q3] - areas[q1];
    prior.area_lo = std::max(0.0f, prior.area_median - 2.0f * prior.area_iqr);
    prior.area_hi = prior.area_median + 2.0f * prior.area_iqr;
    
    prior.aspect_median = aspects[n / 2];
    prior.aspect_iqr = aspects[q3] - aspects[q1];
    prior.aspect_lo = std::max(0.0f, prior.aspect_median - 2.0f * prior.aspect_iqr);
    prior.aspect_hi = prior.aspect_median + 2.0f * prior.aspect_iqr;
    
    std::sort(angles.begin(), angles.end());
    prior.angle_median = angles[n / 2];
    
    std::vector<float> angular_diffs;
    for (float a : angles) angular_diffs.push_back(circular_diff(a, prior.angle_median));
    std::sort(angular_diffs.begin(), angular_diffs.end());
    prior.angle_iqr = angular_diffs[q3] - angular_diffs[q1];
    
    std::sort(widths.begin(), widths.end());
    std::sort(heights.begin(), heights.end());
    prior.width_median = widths[n / 2];
    prior.height_median = heights[n / 2];
    
    return prior;
}

// ---------------------------------------------------------------------------
// Geometry utilities
// ---------------------------------------------------------------------------

cv::Point2f OBBDetector::compute_centroid(const std::vector<cv::Point2f>& points) {
    cv::Point2f c(0, 0);
    for (const auto& p : points) c += p;
    return c / static_cast<float>(points.size());
}

std::vector<cv::Point2f> OBBDetector::order_corners_clockwise_start_tl(const std::vector<cv::Point2f>& points) {
    if (points.size() != 4) return points;
    
    cv::Point2f centroid = compute_centroid(points);
    
    std::vector<std::pair<float, cv::Point2f>> angle_points;
    for (const auto& p : points) {
        float angle = std::atan2(p.y - centroid.y, p.x - centroid.x);
        angle_points.emplace_back(angle, p);
    }
    std::sort(angle_points.begin(), angle_points.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    
    std::vector<cv::Point2f> ordered;
    for (const auto& ap : angle_points) ordered.push_back(ap.second);
    
    int tl_idx = 0;
    for (int i = 1; i < 4; i++) {
        if (ordered[i].y < ordered[tl_idx].y || 
            (ordered[i].y == ordered[tl_idx].y && ordered[i].x < ordered[tl_idx].x))
            tl_idx = i;
    }
    std::rotate(ordered.begin(), ordered.begin() + tl_idx, ordered.end());
    return ordered;
}

float OBBDetector::long_axis_angle_deg(const std::vector<cv::Point2f>& points) {
    if (points.size() != 4) return 0.0f;
    auto ordered = order_corners_clockwise_start_tl(points);
    
    cv::Point2f e01 = ordered[1] - ordered[0];
    cv::Point2f e12 = ordered[2] - ordered[1];
    float len01 = std::sqrt(e01.x * e01.x + e01.y * e01.y);
    float len12 = std::sqrt(e12.x * e12.x + e12.y * e12.y);
    
    cv::Point2f v = (len01 >= len12) ? e01 : e12;
    float angle = std::atan2(v.y, v.x);
    return std::fmod(std::fabs(angle * 180.0f / M_PI), 180.0f);
}

std::pair<float, float> OBBDetector::rect_wh_from_pts(const std::vector<cv::Point2f>& points) {
    if (points.size() != 4) return {0.0f, 0.0f};
    auto ordered = order_corners_clockwise_start_tl(points);
    
    cv::Point2f e01 = ordered[1] - ordered[0];
    cv::Point2f e12 = ordered[2] - ordered[1];
    float len01 = std::sqrt(e01.x * e01.x + e01.y * e01.y);
    float len12 = std::sqrt(e12.x * e12.x + e12.y * e12.y);
    return (len01 >= len12) ? std::make_pair(len01, len12) : std::make_pair(len12, len01);
}

float OBBDetector::circular_diff(float a, float b) {
    float diff = std::fabs(a - b);
    return std::min(diff, 180.0f - diff);
}

OBBDetector::XYWHR OBBDetector::obb_to_xywhr(const OBB& obb) {
    std::vector<cv::Point2f> pts = {
        {obb.x1, obb.y1}, {obb.x2, obb.y2}, {obb.x3, obb.y3}, {obb.x4, obb.y4}
    };
    cv::Point2f center = compute_centroid(pts);
    auto wh = rect_wh_from_pts(pts);
    float angle = long_axis_angle_deg(pts);
    return {center.x, center.y, wh.first, wh.second, angle};
}

// ---------------------------------------------------------------------------
// Classification with priors
// ---------------------------------------------------------------------------

int OBBDetector::classify_with_priors(const std::vector<cv::Point2f>& points) {
    if (priors.empty() || points.size() != 4) return -1;
    
    auto wh = rect_wh_from_pts(points);
    float area = wh.first * wh.second;
    float aspect = wh.first / std::max(wh.second, 1e-6f);
    float angle = long_axis_angle_deg(points);
    
    int best_class = -1;
    float best_score = std::numeric_limits<float>::max();
    
    for (const auto& [class_id, prior] : priors) {
        if (area < prior.area_lo * params.area_lo_gate || area > prior.area_hi * params.area_hi_gate)
            continue;
        if (aspect < prior.aspect_lo * params.aspect_lo_gate || aspect > prior.aspect_hi * params.aspect_hi_gate)
            continue;
        float ang_dev = circular_diff(angle, prior.angle_median);
        if (ang_dev > params.angle_k_gate * std::max(prior.angle_iqr, 5.0f))
            continue;
        
        float score = compute_classification_score(points, class_id);
        if (score < best_score) { best_score = score; best_class = class_id; }
    }
    return best_class;
}

float OBBDetector::compute_classification_score(const std::vector<cv::Point2f>& points, int class_id) {
    if (priors.find(class_id) == priors.end()) return std::numeric_limits<float>::max();
    const ClassPrior& prior = priors[class_id];
    
    auto wh = rect_wh_from_pts(points);
    float area = wh.first * wh.second;
    float aspect = wh.first / std::max(wh.second, 1e-6f);
    float angle = long_axis_angle_deg(points);
    
    float s_area = std::abs(area - prior.area_median) / std::max(prior.area_iqr, 1e-6f);
    float s_aspect = std::abs(aspect - prior.aspect_median) / std::max(prior.aspect_iqr, 1e-6f);
    float s_angle = circular_diff(angle, prior.angle_median) / std::max(prior.angle_iqr, 5.0f);
    return s_area + s_aspect + s_angle;
}

void OBBDetector::print_priors() {
    // Intentionally empty; kept for API compatibility with test code.
}

// ---------------------------------------------------------------------------
// CUDA frame copy
// ---------------------------------------------------------------------------

void OBBDetector::copy_frame_to_cpu(void* device_ptr, cv::Mat& cpu_frame) {
    if (!running.load() || !device_ptr || !h_frame_cpu) return;
    
    unsigned char* d_convert;
    cudaError_t err = cudaMalloc(&d_convert, camera_params->width * camera_params->height * 3);
    if (err != cudaSuccess) {
        std::cerr << "OBBDetector: Failed to allocate GPU memory: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    rgba2bgr_convert(d_convert, (unsigned char*)device_ptr, camera_params->width, camera_params->height, 0);
    
    err = cudaMemcpy2DAsync(
        h_frame_cpu, camera_params->width * 3,
        d_convert, camera_params->width * 3,
        camera_params->width * 3, camera_params->height,
        cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        std::cerr << "OBBDetector: Failed to copy frame to CPU: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_convert);
        return;
    }
    
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << "OBBDetector: Failed to synchronize stream: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_convert);
        return;
    }
    
    cpu_frame = cv::Mat(camera_params->width * camera_params->height * 3, 1, CV_8U, h_frame_cpu)
                .reshape(3, camera_params->height).clone();
    cudaFree(d_convert);
}

std::vector<OBB> OBBDetector::get_latest_detections() {
    std::lock_guard<std::mutex> lock(detections_mtx);
    return latest_detections;
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

void OBBDetector::draw_obb_objects(const cv::Mat& image, cv::Mat& res,
                                   const std::vector<OBB>& obbs,
                                   const std::vector<std::string>& class_names,
                                   const std::vector<std::vector<unsigned int>>& colors,
                                   int line_thickness) {
    res = image.clone();
    
    for (const auto& obb : obbs) {
        std::vector<cv::Point> int_points = {
            {(int)std::round(obb.x1), (int)std::round(obb.y1)},
            {(int)std::round(obb.x2), (int)std::round(obb.y2)},
            {(int)std::round(obb.x3), (int)std::round(obb.y3)},
            {(int)std::round(obb.x4), (int)std::round(obb.y4)}
        };
        
        cv::Scalar color = (obb.class_id >= 0 && obb.class_id < (int)colors.size())
            ? cv::Scalar(colors[obb.class_id][0], colors[obb.class_id][1], colors[obb.class_id][2])
            : cv::Scalar(0, 255, 0);
        
        cv::polylines(res, int_points, true, color, line_thickness);
        
        if (obb.class_id >= 0 && obb.class_id < (int)class_names.size()) {
            char text[256];
            sprintf(text, "%s %.1f%%", class_names[obb.class_id].c_str(), obb.confidence * 100);
            
            int x = (int)std::round(obb.x1);
            int y = (int)std::round(obb.y1) - 5;
            if (y < 0) y = (int)std::round(obb.y1) + 15;
            x = std::max(0, std::min(x, res.cols - 100));
            
            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);
            cv::rectangle(res, cv::Rect(x, y - label_size.height - baseLine, 
                         label_size.width, label_size.height + baseLine), color, -1);
            cv::putText(res, text, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
        }
        
        for (const auto& pt : int_points) cv::circle(res, pt, 3, color, -1);
    }
}

// ---------------------------------------------------------------------------
// Two-stage: local angle extraction
// ---------------------------------------------------------------------------

// Segment the bright cylinder using a percentile-based threshold
// (adapts to any lighting), then compute orientation from image
// moments (PCA).  Moments are far more stable than minAreaRect
// for near-square objects because they weight ALL pixels in the
// blob, not just the boundary.
float OBBDetector::extract_angle_from_local_mask(
    const cv::Mat& frame, float cx, float cy,
    float crop_w, float crop_h, float pad_factor) {

    cv::RotatedRect dummy;
    return extract_angle_and_rect(frame, cx, cy, crop_w, crop_h, pad_factor, dummy);
}

float OBBDetector::extract_angle_and_rect(
    const cv::Mat& frame, float cx, float cy,
    float crop_w, float crop_h, float pad_factor,
    cv::RotatedRect& out_rect) {

    int h = frame.rows, w = frame.cols;
    float pw = crop_w * pad_factor;
    float ph = crop_h * pad_factor;
    int x1p = std::max(0, (int)(cx - pw / 2));
    int y1p = std::max(0, (int)(cy - ph / 2));
    int x2p = std::min(w - 1, (int)(cx + pw / 2));
    int y2p = std::min(h - 1, (int)(cy + ph / 2));
    if (x2p <= x1p || y2p <= y1p) return 0.0f;

    cv::Mat patch = frame(cv::Rect(x1p, y1p, x2p - x1p + 1, y2p - y1p + 1));
    cv::Mat gray, blur, mask;
    cv::cvtColor(patch, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blur, cv::Size(3, 3), 0);

    // Percentile threshold: isolate the brightest ~15% (the cylinder)
    std::vector<uchar> pixels(blur.begin<uchar>(), blur.end<uchar>());
    std::nth_element(pixels.begin(), pixels.begin() + (int)(pixels.size() * 0.85f), pixels.end());
    int thresh = pixels[(int)(pixels.size() * 0.85f)];
    cv::threshold(blur, mask, thresh, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return 0.0f;

    // Pick contour closest to patch center
    cv::Point2f patch_center(patch.cols / 2.0f, patch.rows / 2.0f);
    int best_idx = -1;
    float best_dist = std::numeric_limits<float>::max();
    for (size_t i = 0; i < contours.size(); i++) {
        if (cv::contourArea(contours[i]) < 30) continue;
        cv::Moments m = cv::moments(contours[i]);
        if (m.m00 < 1) continue;
        cv::Point2f centroid(m.m10 / m.m00, m.m01 / m.m00);
        float d = cv::norm(centroid - patch_center);
        if (d < best_dist) { best_dist = d; best_idx = (int)i; }
    }
    if (best_idx < 0) return 0.0f;

    const auto& cnt = contours[best_idx];

    // Use image moments (PCA) for angle — weights every pixel in the
    // blob equally, much more stable than minAreaRect for near-square shapes.
    cv::Moments m = cv::moments(cnt);
    float angle_rad = 0.5f * std::atan2(2.0f * m.mu11, m.mu20 - m.mu02);
    float angle_deg = angle_rad * 180.0f / (float)M_PI;

    // Fill out_rect with minAreaRect for callers that need it
    cv::RotatedRect rect = cv::minAreaRect(cnt);
    out_rect = cv::RotatedRect(
        cv::Point2f(rect.center.x + x1p, rect.center.y + y1p),
        rect.size, rect.angle);

    return angle_deg;
}

// Not used — kept for API compatibility
float OBBDetector::extract_angle_hough(
    const cv::Mat& frame, float cx, float cy,
    float crop_w, float crop_h, float pad_factor) {
    return extract_angle_from_local_mask(frame, cx, cy, crop_w, crop_h, pad_factor);
}

// ---------------------------------------------------------------------------
// Angle smoothing (EMA per tracked object)
// ---------------------------------------------------------------------------

int OBBDetector::find_nearest_track(float cx, float cy) {
    float best_dist = TRACK_MATCH_DIST;
    int best_idx = -1;
    for (size_t i = 0; i < angle_tracks.size(); i++) {
        float dx = angle_tracks[i].cx - cx;
        float dy = angle_tracks[i].cy - cy;
        float d = std::sqrt(dx * dx + dy * dy);
        if (d < best_dist) { best_dist = d; best_idx = (int)i; }
    }
    return best_idx;
}

float OBBDetector::smooth_angle(float cx, float cy, float raw_angle) {
    int idx = find_nearest_track(cx, cy);

    if (idx < 0) {
        // New object — start a fresh track
        angle_tracks.push_back({cx, cy, raw_angle, 0});
        return raw_angle;
    }

    AngleTrack& t = angle_tracks[idx];
    // EMA with circular-aware blending
    float diff = raw_angle - t.smoothed_angle;
    // Wrap to [-90, 90] for RotatedRect angle range
    while (diff > 90.0f)  diff -= 180.0f;
    while (diff < -90.0f) diff += 180.0f;
    t.smoothed_angle += ANGLE_EMA_ALPHA * diff;
    // Keep in [-90, 90)
    while (t.smoothed_angle > 90.0f)  t.smoothed_angle -= 180.0f;
    while (t.smoothed_angle < -90.0f) t.smoothed_angle += 180.0f;

    t.cx = cx;
    t.cy = cy;
    t.age = 0;
    return t.smoothed_angle;
}

// ---------------------------------------------------------------------------
// Two-stage: YOLO box → OBB refinement
// ---------------------------------------------------------------------------

std::vector<OBB> OBBDetector::refine_yolo_detections(
    const cv::Mat& frame, const std::vector<Bbox>& yolo_boxes, int target_class_id) {

    std::vector<OBB> results;
    if (frame.empty() || yolo_boxes.empty()) return results;

    float prior_w = 0, prior_h = 0;
    if (priors.find(target_class_id) != priors.end()) {
        const ClassPrior& p = priors[target_class_id];
        prior_w = p.width_median;
        prior_h = p.height_median;
    }

    // Age all tracks; stale ones will be pruned below
    for (auto& t : angle_tracks) t.age++;

    for (const auto& bbox : yolo_boxes) {
        float bx1 = bbox.rect.x;
        float by1 = bbox.rect.y;
        float bw  = bbox.rect.width;
        float bh  = bbox.rect.height;
        float cx  = bx1 + bw / 2.0f;
        float cy  = by1 + bh / 2.0f;

        // Size is locked to the CSV prior — this is the known cylinder
        // dimension and keeps the OBB a perfect fit.
        float obb_w = (prior_w > 0) ? prior_w : bw;
        float obb_h = (prior_h > 0) ? prior_h : bh;

        // Extract angle from the bright cylinder contour (moments/PCA)
        float crop_w = std::max(bw, obb_w);
        float crop_h = std::max(bh, obb_h);
        cv::RotatedRect measured;
        float raw_angle = extract_angle_and_rect(frame, cx, cy,
                                                 crop_w, crop_h, 2.0f, measured);

        // Temporal smoothing via EMA
        float angle = smooth_angle(cx, cy, raw_angle);

        cv::RotatedRect rrect(cv::Point2f(cx, cy), cv::Size2f(obb_w, obb_h), angle);
        cv::Point2f pts[4];
        rrect.points(pts);

        auto ordered = order_corners_clockwise_start_tl({pts[0], pts[1], pts[2], pts[3]});

        OBB obb(ordered[0].x, ordered[0].y,
                ordered[1].x, ordered[1].y,
                ordered[2].x, ordered[2].y,
                ordered[3].x, ordered[3].y,
                target_class_id, bbox.prob, -1, true);
        results.push_back(obb);
    }

    // Prune stale tracks
    angle_tracks.erase(
        std::remove_if(angle_tracks.begin(), angle_tracks.end(),
                        [](const AngleTrack& t) { return t.age > TRACK_MAX_AGE; }),
        angle_tracks.end());

    return results;
}

// ---------------------------------------------------------------------------
// Seg-based OBB: reconstruct mask from coefficients × prototypes
// ---------------------------------------------------------------------------

std::vector<OBB> OBBDetector::refine_from_seg_masks(
    const std::vector<Bbox>& yolo_boxes,
    const float* mask_protos,
    int proto_h, int proto_w, int num_protos,
    const PreParam& pparam,
    int target_class_id) {

    std::vector<OBB> results;
    if (yolo_boxes.empty() || !mask_protos || num_protos <= 0) return results;

    // Coordinate chain:
    //   image coords → (/ ratio + dw/dh) → input 640 coords → (/ 4) → proto 160 coords
    // pparam: postprocess did  x_img = (x_raw - dw) * ratio
    // so reverse:              x_raw = x_img / ratio + dw
    //                          x_proto = x_raw / 4
    float ratio = pparam.ratio;
    float dw = pparam.dw;
    float dh = pparam.dh;

    for (const auto& bbox : yolo_boxes) {
        if (bbox.mask_coeffs.empty() || (int)bbox.mask_coeffs.size() != num_protos)
            continue;

        // Reconstruct mask: coeffs (32) × protos (32 × 160 × 160) → mask (160 × 160)
        cv::Mat mask_f(proto_h, proto_w, CV_32F, cv::Scalar(0));
        for (int k = 0; k < num_protos; k++) {
            float c = bbox.mask_coeffs[k];
            if (std::abs(c) < 1e-6f) continue;
            const float* proto_plane = mask_protos + k * proto_h * proto_w;
            for (int y = 0; y < proto_h; y++) {
                float* row = mask_f.ptr<float>(y);
                const float* proto_row = proto_plane + y * proto_w;
                for (int x = 0; x < proto_w; x++) {
                    row[x] += c * proto_row[x];
                }
            }
        }

        // Sigmoid
        for (int y = 0; y < proto_h; y++) {
            float* row = mask_f.ptr<float>(y);
            for (int x = 0; x < proto_w; x++) {
                row[x] = 1.0f / (1.0f + std::exp(-row[x]));
            }
        }

        // Threshold at 0.3 for wider coverage
        cv::Mat mask_bin;
        cv::threshold(mask_f, mask_bin, 0.3f, 255.0f, cv::THRESH_BINARY);
        mask_bin.convertTo(mask_bin, CV_8U);

        // Upscale mask from 160×160 to 640×640 before contour fitting —
        // this gives much better resolution for minAreaRect.
        cv::Mat mask_up;
        cv::resize(mask_bin, mask_up, cv::Size(proto_w * 4, proto_h * 4), 0, 0, cv::INTER_LINEAR);
        cv::threshold(mask_up, mask_up, 127, 255, cv::THRESH_BINARY);

        // Map bbox from image coords → input640 coords (upscaled mask space)
        float bx1_up = bbox.rect.x / ratio + dw;
        float by1_up = bbox.rect.y / ratio + dh;
        float bx2_up = (bbox.rect.x + bbox.rect.width) / ratio + dw;
        float by2_up = (bbox.rect.y + bbox.rect.height) / ratio + dh;

        int rx1 = std::max(0, (int)std::floor(bx1_up));
        int ry1 = std::max(0, (int)std::floor(by1_up));
        int rx2 = std::min(proto_w * 4, (int)std::ceil(bx2_up));
        int ry2 = std::min(proto_h * 4, (int)std::ceil(by2_up));
        if (rx2 <= rx1 || ry2 <= ry1) continue;

        // Zero out mask outside the bbox
        cv::Mat crop_mask = cv::Mat::zeros(proto_h * 4, proto_w * 4, CV_8U);
        mask_up(cv::Rect(rx1, ry1, rx2 - rx1, ry2 - ry1))
            .copyTo(crop_mask(cv::Rect(rx1, ry1, rx2 - rx1, ry2 - ry1)));

        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(crop_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (contours.empty()) continue;

        // Pick largest contour
        auto best_it = std::max_element(contours.begin(), contours.end(),
            [](const auto& a, const auto& b) { return cv::contourArea(a) < cv::contourArea(b); });
        if (cv::contourArea(*best_it) < 10) continue;

        // Fit minAreaRect in input640 space, then map to image space
        cv::RotatedRect rrect = cv::minAreaRect(*best_it);

        // input640 → image: (x - dw) * ratio
        rrect.center.x = (rrect.center.x - dw) * ratio;
        rrect.center.y = (rrect.center.y - dh) * ratio;
        rrect.size.width *= ratio;
        rrect.size.height *= ratio;

        cv::Point2f pts[4];
        rrect.points(pts);
        auto ordered = order_corners_clockwise_start_tl({pts[0], pts[1], pts[2], pts[3]});

        OBB obb(ordered[0].x, ordered[0].y,
                ordered[1].x, ordered[1].y,
                ordered[2].x, ordered[2].y,
                ordered[3].x, ordered[3].y,
                target_class_id, bbox.prob, -1, true);
        results.push_back(obb);
    }
    return results;
}

// ---------------------------------------------------------------------------
// Stable detection tracking
// ---------------------------------------------------------------------------

bool OBBDetector::should_update_detections(const std::vector<OBB>& new_detections) {
    if (!detections_stable) return true;
    
    if ((int)new_detections.size() != (int)stable_detections.size()) return true;
    
    if (new_detections.empty()) return false;
    
    std::vector<bool> stable_used(stable_detections.size(), false);
    
    for (const auto& nd : new_detections) {
        cv::Point2f nc = compute_centroid({
            {nd.x1, nd.y1}, {nd.x2, nd.y2}, {nd.x3, nd.y3}, {nd.x4, nd.y4}
        });
        
        float min_dist = std::numeric_limits<float>::max();
        int best = -1;
        for (size_t i = 0; i < stable_detections.size(); i++) {
            if (stable_used[i]) continue;
            cv::Point2f sc = compute_centroid({
                {stable_detections[i].x1, stable_detections[i].y1},
                {stable_detections[i].x2, stable_detections[i].y2},
                {stable_detections[i].x3, stable_detections[i].y3},
                {stable_detections[i].x4, stable_detections[i].y4}
            });
            float d = cv::norm(nc - sc);
            if (d < min_dist) { min_dist = d; best = (int)i; }
        }
        
        if (best >= 0 && min_dist > 50.0f) return true;
        if (best >= 0) stable_used[best] = true;
    }
    return false;
}
