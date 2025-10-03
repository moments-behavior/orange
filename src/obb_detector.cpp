#include "obb_detector.h"
#include "kernel.cuh"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

// Simple CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        throw std::runtime_error("CUDA error"); \
    } \
} while(0)

OBBDetector::OBBDetector(CameraParams* params, const std::vector<std::string>& csv_paths, 
                         const OBBDetectorParams& detector_params)
    : camera_params(params), csv_paths(csv_paths), params(detector_params),
      background_initialized(false), running(false), frame_ready(false), frames_processed(0), detections_found(0),
      d_frame_original(nullptr), h_frame_cpu(nullptr) {
    
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
        // Don't use CUDA_CHECK in destructor to avoid std::terminate
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
}

bool OBBDetector::initialize() {
    try {
        // Learn priors from CSV files
        learn_priors_from_csvs();
        
        if (priors.empty()) {
            std::cerr << "No priors learned from CSV files" << std::endl;
            return false;
        }
        
        std::cout << "OBBDetector initialized with priors for classes: ";
        for (const auto& p : priors) {
            std::cout << p.first << " ";
        }
        std::cout << std::endl;
        
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
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
    }
}

void OBBDetector::notify_frame_ready(void* device_image_ptr, cudaStream_t copy_stream) {
    // Safety check: don't process if we're shutting down
    if (!running.load()) {
        return;
    }
    
    // Store the device pointer directly - no need to copy since we'll copy to CPU later
    d_frame_original = device_image_ptr;
    
    cudaEventRecord(copy_done_event, copy_stream);
    frame_ready = true;
    cv.notify_one();
}

void OBBDetector::thread_loop() {
    CUDA_CHECK(cudaSetDevice(camera_params->gpu_id));
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // Allocate CPU memory for frame copy (3 bytes per pixel - BGR)
    size_t cpu_frame_size = camera_params->width * camera_params->height * 3 * sizeof(unsigned char);
    CUDA_CHECK(cudaMallocHost(&h_frame_cpu, cpu_frame_size));
    
    std::cout << "OBBDetector thread started for camera: " << camera_params->camera_serial << std::endl;
    
    while (running.load()) {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&] { return !running.load() || frame_ready; });
        
        if (!running.load()) break;
        
        // Reset frame ready flag
        frame_ready = false;
        lock.unlock();
        
        // Wait for frame copy to complete
        cudaStreamWaitEvent(stream, copy_done_event, 0);
        
        // Safety check: don't process if we're shutting down or d_frame_original is invalid
        if (!running.load() || !d_frame_original) {
            continue;
        }
        
        // Process frame
        cv::Mat frame;
        copy_frame_to_cpu(d_frame_original, frame);
        
        if (frame.empty()) {
            std::cerr << "Failed to copy frame to CPU" << std::endl;
            continue;
        }
        
        // Build background model from first few frames (true median like Python)
        if (!background_initialized) {
            if (frames_processed == 0) {
                // Initialize background frames collection
                background_frames.clear();
                std::cout << "OBB: Building background from " << params.bg_frames << " frames..." << std::endl;
            }
            
            if (frames_processed < params.bg_frames) {
                // Collect frames for median background
                background_frames.push_back(frame.clone());
                if ((frames_processed + 1) % 10 == 0 || frames_processed == params.bg_frames - 1) {
                    std::cout << "OBB: Background progress: " << (frames_processed + 1) << "/" << params.bg_frames << std::endl;
                }
            } else {
                // Compute true median background (like Python)
                if (!background_frames.empty()) {
                    background_model = compute_median_background(background_frames);
                    background_initialized = true;
                    std::cout << "OBB: Background ready - detection can start" << std::endl;
                } else {
                    std::cerr << "OBB: No background frames collected!" << std::endl;
                    background_initialized = true; // Continue anyway
                }
            }
            frames_processed++;
            continue;
        }
        
        // Convert background model back to CV_8U for detection
        cv::Mat background_u8;
        background_model.convertTo(background_u8, CV_8U);
        
        // Detect candidates
        auto candidates = detect_candidates(frame, background_u8);
        
        // Classify candidates
        std::vector<OBB> detections;
        for (const auto& candidate : candidates) {
            int class_id = classify_with_priors(candidate);
            if (class_id >= 0) {
                OBB obb(candidate[0].x, candidate[0].y, candidate[1].x, candidate[1].y,
                       candidate[2].x, candidate[2].y, candidate[3].x, candidate[3].y,
                       class_id, 1.0f);
                detections.push_back(obb);
            }
        }
        
        // Update results
        {
            std::lock_guard<std::mutex> lock(detections_mtx);
            latest_detections = detections;
        }
        
        frames_processed++;
        detections_found += detections.size();
    }
    
    std::cout << "OBBDetector thread finished. Total frames: " << frames_processed 
              << ", Total detections: " << detections_found << std::endl;
}

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
        
        // Skip header lines
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
        
        while (std::getline(iss, token, ',')) {
            tokens.push_back(token);
        }
        
        if (tokens.size() < 11) continue; // Need at least frame, obb_id, class_id, 8 coordinates
        
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
            std::cerr << "Error parsing line: " << line << " - " << e.what() << std::endl;
            continue;
        }
    }
    
    return true;
}

void OBBDetector::learn_priors_from_csvs() {
    std::cout << "=== OBB CSV Learning Debug ===" << std::endl;
    std::cout << "Number of CSV files to process: " << csv_paths.size() << std::endl;
    
    std::map<int, std::vector<std::vector<cv::Point2f>>> class_samples;
    
    for (const auto& csv_path : csv_paths) {
        std::cout << "Processing CSV file: " << csv_path << std::endl;
        std::vector<std::pair<int, std::vector<cv::Point2f>>> labels;
        if (parse_labels_csv(csv_path, labels)) {
            std::cout << "Successfully parsed " << labels.size() << " labels from " << csv_path << std::endl;
            for (const auto& label : labels) {
                class_samples[label.first].push_back(label.second);
            }
        } else {
            std::cout << "Failed to parse CSV file: " << csv_path << std::endl;
        }
    }
    
    std::cout << "Total classes found: " << class_samples.size() << std::endl;
    for (const auto& class_data : class_samples) {
        priors[class_data.first] = compute_class_prior(class_data.second);
        std::cout << "Learned priors for class " << class_data.first 
                  << " from " << class_data.second.size() << " samples" << std::endl;
    }
    std::cout << "=== End CSV Learning Debug ===" << std::endl;
}

ClassPrior OBBDetector::compute_class_prior(const std::vector<std::vector<cv::Point2f>>& class_samples) {
    ClassPrior prior;
    
    if (class_samples.empty()) return prior;
    
    std::vector<float> areas, aspects, angles;
    
    for (const auto& sample : class_samples) {
        auto wh = rect_wh_from_pts(sample);
        float area = wh.first * wh.second;
        float aspect = wh.first / std::max(wh.second, 1e-6f);
        float angle = long_axis_angle_deg(sample);
        
        areas.push_back(area);
        aspects.push_back(aspect);
        angles.push_back(angle);
    }
    
    // Compute IQR bounds for area and aspect
    std::sort(areas.begin(), areas.end());
    std::sort(aspects.begin(), aspects.end());
    
    size_t n = areas.size();
    float q1_idx = n * 0.25f;
    float q3_idx = n * 0.75f;
    
    prior.area_median = areas[n / 2];
    prior.area_iqr = areas[std::min((size_t)q3_idx, n-1)] - areas[std::min((size_t)q1_idx, n-1)];
    prior.area_lo = std::max(0.0f, prior.area_median - 2.0f * prior.area_iqr);
    prior.area_hi = prior.area_median + 2.0f * prior.area_iqr;
    
    prior.aspect_median = aspects[n / 2];
    prior.aspect_iqr = aspects[std::min((size_t)q3_idx, n-1)] - aspects[std::min((size_t)q1_idx, n-1)];
    prior.aspect_lo = std::max(0.0f, prior.aspect_median - 2.0f * prior.aspect_iqr);
    prior.aspect_hi = prior.aspect_median + 2.0f * prior.aspect_iqr;
    
    // Compute angular statistics
    std::sort(angles.begin(), angles.end());
    prior.angle_median = angles[n / 2];
    
    // Compute angular IQR (wrapped)
    std::vector<float> angular_diffs;
    for (float angle : angles) {
        angular_diffs.push_back(circular_diff(angle, prior.angle_median));
    }
    std::sort(angular_diffs.begin(), angular_diffs.end());
    prior.angle_iqr = angular_diffs[std::min((size_t)q3_idx, n-1)] - angular_diffs[std::min((size_t)q1_idx, n-1)];
    
    return prior;
}

cv::Point2f OBBDetector::compute_centroid(const std::vector<cv::Point2f>& points) {
    cv::Point2f centroid(0, 0);
    for (const auto& p : points) {
        centroid += p;
    }
    return centroid / static_cast<float>(points.size());
}

std::vector<cv::Point2f> OBBDetector::order_corners_clockwise_start_tl(const std::vector<cv::Point2f>& points) {
    if (points.size() != 4) return points;
    
    cv::Point2f centroid = OBBDetector::compute_centroid(points);
    
    // Sort by angle from centroid
    std::vector<std::pair<float, cv::Point2f>> angle_points;
    for (const auto& p : points) {
        float angle = std::atan2(p.y - centroid.y, p.x - centroid.x);
        angle_points.emplace_back(angle, p);
    }
    
    // Custom comparator for sorting by angle
    std::sort(angle_points.begin(), angle_points.end(), 
              [](const std::pair<float, cv::Point2f>& a, const std::pair<float, cv::Point2f>& b) {
                  return a.first < b.first;
              });
    
    std::vector<cv::Point2f> ordered;
    for (const auto& ap : angle_points) {
        ordered.push_back(ap.second);
    }
    
    // Find top-left corner (minimum y, then minimum x)
    int tl_idx = 0;
    for (int i = 1; i < 4; i++) {
        if (ordered[i].y < ordered[tl_idx].y || 
            (ordered[i].y == ordered[tl_idx].y && ordered[i].x < ordered[tl_idx].x)) {
            tl_idx = i;
        }
    }
    
    // Rotate to start from top-left
    std::rotate(ordered.begin(), ordered.begin() + tl_idx, ordered.end());
    
    return ordered;
}

float OBBDetector::long_axis_angle_deg(const std::vector<cv::Point2f>& points) {
    if (points.size() != 4) return 0.0f;
    
    auto ordered = order_corners_clockwise_start_tl(points);
    
    // Compute edge vectors
    cv::Point2f e01 = ordered[1] - ordered[0];
    cv::Point2f e12 = ordered[2] - ordered[1];
    
    float len01 = std::sqrt(e01.x * e01.x + e01.y * e01.y);
    float len12 = std::sqrt(e12.x * e12.x + e12.y * e12.y);
    
    cv::Point2f v = (len01 >= len12) ? e01 : e12;
    float angle = std::atan2(v.y, v.x);
    
    // Convert to degrees and wrap to [0, 180)
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
    // Convert 4 corner points to center, width, height, and rotation
    std::vector<cv::Point2f> points = {
        cv::Point2f(obb.x1, obb.y1),
        cv::Point2f(obb.x2, obb.y2),
        cv::Point2f(obb.x3, obb.y3),
        cv::Point2f(obb.x4, obb.y4)
    };
    
    // Compute center point
    cv::Point2f center = compute_centroid(points);
    
    // Compute width and height
    auto wh = rect_wh_from_pts(points);
    
    // Compute rotation angle
    float angle = long_axis_angle_deg(points);
    
    return {center.x, center.y, wh.first, wh.second, angle};
}

bool OBBDetector::build_background(cv::VideoCapture& cap, cv::Mat& background) {
    if (params.bg_mode == "first") {
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        return cap.read(background);
    } else if (params.bg_mode == "median") {
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);
        std::vector<cv::Mat> frames;
        
        for (int i = 0; i < params.bg_frames; i++) {
            cv::Mat frame;
            if (!cap.read(frame)) break;
            frames.push_back(frame);
        }
        
        if (frames.empty()) return false;
        
        // Compute median
        std::vector<cv::Mat> channels;
        cv::split(frames[0], channels);
        
        for (int c = 0; c < channels.size(); c++) {
            std::vector<cv::Mat> channel_frames;
            for (const auto& frame : frames) {
                std::vector<cv::Mat> frame_channels;
                cv::split(frame, frame_channels);
                channel_frames.push_back(frame_channels[c]);
            }
            cv::Mat median_channel;
            cv::merge(channel_frames, median_channel);
            // Compute median manually since REDUCE_MEDIAN is not available
            cv::Mat median_result = cv::Mat::zeros(median_channel.rows, median_channel.cols, CV_32F);
            for (int r = 0; r < median_channel.rows; r++) {
                for (int c = 0; c < median_channel.cols; c++) {
                    std::vector<float> values;
                    for (int ch = 0; ch < median_channel.channels(); ch++) {
                        values.push_back(median_channel.at<float>(r, c * median_channel.channels() + ch));
                    }
                    std::sort(values.begin(), values.end());
                    median_result.at<float>(r, c) = values[values.size() / 2];
                }
            }
            channels[c] = median_result;
        }
        
        cv::merge(channels, background);
        return true;
    }
    
    return false;
}

cv::Mat OBBDetector::compute_median_background(const std::vector<cv::Mat>& frames) {
    if (frames.empty()) {
        return cv::Mat();
    }
    
    // Validate all frames have the same size
    cv::Size frame_size = frames[0].size();
    int frame_channels = frames[0].channels();
    
    for (const auto& frame : frames) {
        if (frame.size() != frame_size || frame.channels() != frame_channels) {
            std::cerr << "OBB: Frame size mismatch! Expected " << frame_size.width << "x" << frame_size.height 
                      << " channels=" << frame_channels << ", got " << frame.size().width << "x" << frame.size().height 
                      << " channels=" << frame.channels() << std::endl;
            return cv::Mat();
        }
    }
    
    // Convert all frames to float32
    std::vector<cv::Mat> float_frames;
    for (const auto& frame : frames) {
        cv::Mat float_frame;
        frame.convertTo(float_frame, CV_32F);
        float_frames.push_back(float_frame);
    }
    
    // Compute median pixel-wise (like Python: np.median(np.stack(imgs,0), axis=0))
    cv::Mat median_result = cv::Mat::zeros(frame_size, CV_32FC3);
    
    for (int r = 0; r < frame_size.height; r++) {
        for (int c = 0; c < frame_size.width; c++) {
            for (int ch = 0; ch < frame_channels; ch++) {
                std::vector<float> values;
                for (const auto& frame : float_frames) {
                    values.push_back(frame.at<cv::Vec3f>(r, c)[ch]);
                }
                std::sort(values.begin(), values.end());
                median_result.at<cv::Vec3f>(r, c)[ch] = values[values.size() / 2];
            }
        }
    }
    
    // Convert back to uint8
    cv::Mat result;
    median_result.convertTo(result, CV_8UC3);
    return result;
}

std::vector<std::vector<cv::Point2f>> OBBDetector::detect_candidates(const cv::Mat& frame, const cv::Mat& background) {
    std::vector<std::vector<cv::Point2f>> candidates;
    
    if (frame.empty() || background.empty()) return candidates;
    
    // Compute difference
    cv::Mat diff;
    cv::subtract(frame, background, diff);
    
    // Use green channel for motion detection
    std::vector<cv::Mat> channels;
    cv::split(diff, channels);
    cv::Mat green_channel = channels[1];
    
    // Threshold for motion detection
    cv::Mat mask;
    cv::threshold(green_channel, mask, params.threshold, 255, cv::THRESH_BINARY);
    
    // If no motion detected, try lower threshold
    if (cv::countNonZero(mask) == 0) {
        cv::threshold(green_channel, mask, params.threshold * 0.5, 255, cv::THRESH_BINARY);
    }
    
    // Morphological operations - use smaller kernels to avoid removing all motion
    cv::Mat kernel = create_disk_kernel(1);  // Smaller erosion kernel
    cv::erode(mask, mask, kernel, cv::Point(-1, -1), 1);
    
    kernel = create_disk_kernel(3);  // Smaller dilation kernel
    cv::dilate(mask, mask, kernel, cv::Point(-1, -1), 1);
    
    // If morphological operations removed all pixels, try without them
    if (cv::countNonZero(mask) == 0) {
        // Re-threshold and skip morphological operations
        cv::threshold(green_channel, mask, params.threshold, 255, cv::THRESH_BINARY);
    }
    
    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    // Convert to oriented bounding boxes
    for (const auto& contour : contours) {
        if (contour.size() < 5) continue;
        
        cv::RotatedRect rect = cv::minAreaRect(contour);
        cv::Point2f rect_points[4];
        rect.points(rect_points);
        
        std::vector<cv::Point2f> points(rect_points, rect_points + 4);
        auto ordered_points = order_corners_clockwise_start_tl(points);
        candidates.push_back(ordered_points);
    }
    
    return candidates;
}

cv::Mat OBBDetector::create_disk_kernel(int radius) {
    int size = 2 * radius + 1;
    cv::Mat kernel = cv::Mat::zeros(size, size, CV_8UC1);
    
    cv::Point center(radius, radius);
    cv::circle(kernel, center, radius, cv::Scalar(255), -1);
    
    return kernel;
}

int OBBDetector::classify_with_priors(const std::vector<cv::Point2f>& points) {
    if (priors.empty() || points.size() != 4) return -1;
    
    auto wh = rect_wh_from_pts(points);
    float area = wh.first * wh.second;
    float aspect = wh.first / std::max(wh.second, 1e-6f);
    float angle = long_axis_angle_deg(points);
    
    int best_class = -1;
    float best_score = std::numeric_limits<float>::max();
    
    for (const auto& prior_pair : priors) {
        int class_id = prior_pair.first;
        const ClassPrior& prior = prior_pair.second;
        
        // Check gates
        float area_lo_bound = prior.area_lo * params.area_lo_gate;
        float area_hi_bound = prior.area_hi * params.area_hi_gate;
        if (area < area_lo_bound || area > area_hi_bound) {
            continue;
        }
        
        float aspect_lo_bound = prior.aspect_lo * params.aspect_lo_gate;
        float aspect_hi_bound = prior.aspect_hi * params.aspect_hi_gate;
        if (aspect < aspect_lo_bound || aspect > aspect_hi_bound) {
            continue;
        }
        
        float ang_dev = circular_diff(angle, prior.angle_median);
        float ang_threshold = params.angle_k_gate * std::max(prior.angle_iqr, 5.0f);
        if (ang_dev > ang_threshold) {
            continue;
        }
        
        // Compute score
        float score = compute_classification_score(points, class_id);
        
        if (score < best_score) {
            best_score = score;
            best_class = class_id;
        }
    }
    
    return best_class;
}

void OBBDetector::print_priors() {
    for (const auto& prior_pair : priors) {
        int class_id = prior_pair.first;
        const ClassPrior& prior = prior_pair.second;
        
        std::cout << "Class " << class_id << ":" << std::endl;
        std::cout << "  Area: median=" << prior.area_median << ", iqr=" << prior.area_iqr 
                  << ", range=[" << prior.area_lo << ", " << prior.area_hi << "]" << std::endl;
        std::cout << "  Aspect: median=" << prior.aspect_median << ", iqr=" << prior.aspect_iqr 
                  << ", range=[" << prior.aspect_lo << ", " << prior.aspect_hi << "]" << std::endl;
        std::cout << "  Angle: median=" << prior.angle_median << ", iqr=" << prior.angle_iqr << std::endl;
        std::cout << std::endl;
    }
}

float OBBDetector::compute_classification_score(const std::vector<cv::Point2f>& points, int class_id) {
    if (priors.find(class_id) == priors.end()) return std::numeric_limits<float>::max();
    
    const ClassPrior& prior = priors[class_id];
    auto wh = rect_wh_from_pts(points);
    float area = wh.first * wh.second;
    float aspect = wh.first / std::max(wh.second, 1e-6f);
    float angle = long_axis_angle_deg(points);
    
    // Normalized deviations
    float s_area = std::abs(area - prior.area_median) / std::max(prior.area_iqr, 1e-6f);
    float s_aspect = std::abs(aspect - prior.aspect_median) / std::max(prior.aspect_iqr, 1e-6f);
    float s_angle = circular_diff(angle, prior.angle_median) / std::max(prior.angle_iqr, 5.0f);
    
    return s_area + s_aspect + s_angle;
}

void OBBDetector::copy_frame_to_cpu(void* device_ptr, cv::Mat& cpu_frame) {
    // Safety check: don't process if we're shutting down or device_ptr is invalid
    if (!running.load() || !device_ptr || !h_frame_cpu) {
        return;
    }
    
    // Use the same GPU-to-CPU conversion pattern as the working branch
    // First convert RGBA to BGR on GPU, then copy to CPU
    
    // Allocate temporary GPU buffer for BGR conversion
    unsigned char* d_convert;
    cudaError_t err = cudaMalloc(&d_convert, camera_params->width * camera_params->height * 3);
    if (err != cudaSuccess) {
        std::cerr << "OBBDetector: Failed to allocate GPU memory: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // Convert RGBA to BGR on GPU (like the working branch)
    rgba2bgr_convert(d_convert, (unsigned char*)device_ptr, camera_params->width, camera_params->height, 0);
    
    // Copy BGR frame from GPU to CPU (async)
    err = cudaMemcpy2DAsync(
        h_frame_cpu, camera_params->width * 3,  // 3 bytes per pixel (BGR)
        d_convert, camera_params->width * 3,    // same pitch on device
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
    
    // Create OpenCV Mat from CPU frame (BGR format) - same as working branch
    cpu_frame = cv::Mat(camera_params->width * camera_params->height * 3, 1, CV_8U, h_frame_cpu)
                .reshape(3, camera_params->height).clone();
    
    // Clean up temporary GPU buffer
    cudaFree(d_convert);
}

std::vector<OBB> OBBDetector::get_latest_detections() {
    std::lock_guard<std::mutex> lock(detections_mtx);
    return latest_detections;
}

std::vector<OBB> OBBDetector::process_frame_sync(const cv::Mat& frame) {
    if (frame.empty()) {
        std::cout << "OBB: Empty frame received" << std::endl;
        return std::vector<OBB>();
    }
    
    // Build background from first N frames if not ready
    if (!background_initialized) {
        if (frames_processed < params.bg_frames) {
            // Add frame to background building
            if (frames_processed == 0) {
                // Ensure background_model has same type and channels as input frame
                frame.convertTo(background_model, CV_32F);
                std::cout << "OBB: Started background building, frame size: " << frame.cols << "x" << frame.rows 
                          << ", channels: " << frame.channels() << ", type: " << frame.type() << std::endl;
            } else {
                // Running average for background - convert frame to float32 first
                cv::Mat frame_f32;
                frame.convertTo(frame_f32, CV_32F);
                double alpha = 1.0 / (frames_processed + 1);
                cv::accumulateWeighted(frame_f32, background_model, alpha);
            }
            frames_processed++;
            
            std::cout << "OBB: Background building progress: " << frames_processed << "/" << params.bg_frames << std::endl;
            
            if (frames_processed >= params.bg_frames) {
                background_initialized = true;
                std::cout << "OBB: Background ready after " << frames_processed << " frames" << std::endl;
            }
        }
        return std::vector<OBB>();
    }
    
    // Detect candidates (returns contour points)
    // Convert background_model back to CV_8U for detection
    cv::Mat background_u8;
    background_model.convertTo(background_u8, CV_8U);
    auto candidates = detect_candidates(frame, background_u8);
    std::cout << "OBB: Found " << candidates.size() << " motion candidates" << std::endl;
    
    // Classify candidates against learned priors
    std::vector<OBB> detections;
    for (size_t i = 0; i < candidates.size(); i++) {
        const auto& candidate = candidates[i];
        int class_id = classify_with_priors(candidate);
        std::cout << "OBB: Candidate " << i << " classified as class " << class_id << std::endl;
        
        if (class_id >= 0) {
            // Convert contour points to OBB structure
            OBB obb(candidate[0].x, candidate[0].y, candidate[1].x, candidate[1].y,
                   candidate[2].x, candidate[2].y, candidate[3].x, candidate[3].y,
                   class_id, 1.0f);
            detections.push_back(obb);
            std::cout << "OBB: Added detection at (" << obb.x1 << "," << obb.y1 << ") to (" << obb.x3 << "," << obb.y3 << ")" << std::endl;
        }
    }
    
    std::cout << "OBB: Total detections: " << detections.size() << std::endl;
    
    // Update latest detections (thread-safe)
    {
        std::lock_guard<std::mutex> lock(detections_mtx);
        latest_detections = detections;
    }
    
    return detections;
}

void OBBDetector::draw_obb_objects(const cv::Mat& image, cv::Mat& res,
                                   const std::vector<OBB>& obbs,
                                   const std::vector<std::string>& class_names,
                                   const std::vector<std::vector<unsigned int>>& colors,
                                   int line_thickness) {
    res = image.clone();
    
    for (const auto& obb : obbs) {
        // Create vector of points for the oriented bounding box
        std::vector<cv::Point2f> obb_points = {
            cv::Point2f(obb.x1, obb.y1),
            cv::Point2f(obb.x2, obb.y2),
            cv::Point2f(obb.x3, obb.y3),
            cv::Point2f(obb.x4, obb.y4)
        };
        
        // Convert to integer points for drawing
        std::vector<cv::Point> int_points;
        for (const auto& pt : obb_points) {
            int_points.emplace_back(static_cast<int>(std::round(pt.x)), 
                                   static_cast<int>(std::round(pt.y)));
        }
        
        // Get color for this class
        cv::Scalar color;
        if (obb.class_id >= 0 && obb.class_id < colors.size()) {
            color = cv::Scalar(colors[obb.class_id][0], 
                              colors[obb.class_id][1], 
                              colors[obb.class_id][2]);
        } else {
            color = cv::Scalar(0, 255, 0); // Default green color
        }
        
        // Draw the oriented bounding box as a polygon
        cv::polylines(res, int_points, true, color, line_thickness);
        
        // Draw class label and confidence
        if (obb.class_id >= 0 && obb.class_id < class_names.size()) {
            char text[256];
            sprintf(text, "%s %.1f%%", class_names[obb.class_id].c_str(), 
                    obb.confidence * 100);
            
            // Position label at the first corner (top-left)
            int x = static_cast<int>(std::round(obb.x1));
            int y = static_cast<int>(std::round(obb.y1)) - 5;
            
            // Ensure label is within image bounds
            if (y < 0) y = static_cast<int>(std::round(obb.y1)) + 15;
            if (x < 0) x = 0;
            if (x > res.cols - 100) x = res.cols - 100;
            
            // Get text size for background rectangle
            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 
                                                 0.4, 1, &baseLine);
            
            // Draw background rectangle for text
            cv::rectangle(res, 
                         cv::Rect(x, y - label_size.height - baseLine, 
                                 label_size.width, label_size.height + baseLine),
                         color, -1);
            
            // Draw text
            cv::putText(res, text, cv::Point(x, y), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
        }
        
        // Draw corner markers for better visibility
        for (const auto& pt : int_points) {
            cv::circle(res, pt, 3, color, -1);
        }
    }
}

