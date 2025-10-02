#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "obb_detector.h"
#include "camera.h"

// Simple test to verify OBB detector on real image
int main() {
    std::cout << "Testing OBB Detector on extracted video frames..." << std::endl;
    
    // Load test frame (extracted from video)
    std::string frame_path = "../csv/frames/frame_200.jpg";  // Use frame 200 as test
    cv::Mat test_frame = cv::imread(frame_path);
    
    if (test_frame.empty()) {
        std::cerr << "Failed to load frame: " << frame_path << std::endl;
        return -1;
    }
    
    std::cout << "Loaded test frame: " << test_frame.rows << "x" << test_frame.cols << std::endl;
    
    // Set up camera parameters for the frame
    CameraParams camera_params;
    camera_params.width = test_frame.cols;
    camera_params.height = test_frame.rows;
    camera_params.gpu_id = 0;
    camera_params.color = true;
    camera_params.gpu_direct = false; // We'll use CPU for this test
    camera_params.camera_serial = "Cam2005325";
    
    // CSV files containing hand-labeled training data
    std::vector<std::string> csv_paths = {
        "../csv/Cam2005325_obb.csv"
    };
    
    // OBB detector parameters
    OBBDetectorParams detector_params;
    detector_params.threshold = 36;
    detector_params.bg_mode = "first";
    detector_params.keep_negatives = true;
    
    // Use standard gates now that we have the correct scale
    detector_params.area_lo_gate = 0.9f;    // Allow slightly smaller than lo
    detector_params.area_hi_gate = 1.1f;    // Allow slightly larger than hi
    detector_params.aspect_lo_gate = 0.9f;
    detector_params.aspect_hi_gate = 1.1f;
    detector_params.angle_k_gate = 2.0f;    // Allow up to 2*IQR angular deviation
    
    // Create OBB detector
    OBBDetector detector(&camera_params, csv_paths, detector_params);
    
    // Initialize (learn priors from CSV files)
    if (!detector.initialize()) {
        std::cerr << "Failed to initialize OBB detector" << std::endl;
        return -1;
    }
    
    std::cout << "OBB detector initialized successfully!" << std::endl;
    
    // Print learned priors for debugging
    std::cout << "\nLearned priors:" << std::endl;
    detector.print_priors();
    
    // Test motion detection and OBB generation
    std::cout << "Testing motion detection and OBB generation..." << std::endl;
    
    // Load background frame (frame 0)
    std::string bg_path = "../csv/frames/frame_000.jpg";
    cv::Mat background = cv::imread(bg_path);
    
    if (background.empty()) {
        std::cerr << "Failed to load background frame: " << bg_path << std::endl;
        return -1;
    }
    std::cout << "Using frame 0 as background: " << background.rows << "x" << background.cols << std::endl;
    
    // Test the detection functions on the test frame
    std::cout << "\nProcessing test frame..." << std::endl;
    
    // Simulate the detection process
    cv::Mat diff;
    cv::subtract(test_frame, background, diff);
    
    // Use green channel for motion detection
    std::vector<cv::Mat> channels;
    cv::split(diff, channels);
    cv::Mat green_channel = channels[1];
    
    // Try different thresholds using the exact Python approach
    std::vector<int> thresholds = {20, 25, 30, 35, 40, 45, 50};
    cv::Mat best_mask;
    int best_threshold = detector_params.threshold;
    int best_matching_contours = 0;
    
    for (int thresh : thresholds) {
        cv::Mat test_mask;
        cv::threshold(green_channel, test_mask, thresh, 255, cv::THRESH_BINARY);
        
        // Use the exact Python morphological operations
        cv::Mat kernel_2 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)); // disk_kernel(2)
        cv::Mat kernel_5 = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(11, 11)); // disk_kernel(5)
        
        cv::erode(test_mask, test_mask, kernel_2, cv::Point(-1, -1), 1);
        cv::dilate(test_mask, test_mask, kernel_5, cv::Point(-1, -1), 1);
        
        // Count contours and check how many match the learned priors
        std::vector<std::vector<cv::Point>> test_contours;
        cv::findContours(test_mask, test_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        int matching_contours = 0;
        for (const auto& contour : test_contours) {
            if (contour.size() < 5) continue;
            
            cv::RotatedRect rect = cv::minAreaRect(contour);
            cv::Point2f rect_points[4];
            rect.points(rect_points);
            
            std::vector<cv::Point2f> points(rect_points, rect_points + 4);
            points = OBBDetector::order_corners_clockwise_start_tl(points);
            
            auto wh = detector.rect_wh_from_pts(points);
            float area = wh.first * wh.second;
            
            // Debug: print area for first few contours
            if (matching_contours < 3) {
                std::cout << "    Contour area: " << area << std::endl;
            }
            
            // Check if this contour matches any of the learned priors
            bool matches_any = false;
            for (const auto& prior_pair : detector.get_priors()) {
                const ClassPrior& prior = prior_pair.second;
                float area_lo_bound = prior.area_lo * detector_params.area_lo_gate;
                float area_hi_bound = prior.area_hi * detector_params.area_hi_gate;
                if (area >= area_lo_bound && area <= area_hi_bound) {
                    matches_any = true;
                    break;
                }
            }
            
            if (matches_any) {
                matching_contours++;
            }
        }
        
        std::cout << "Threshold " << thresh << ": Found " << test_contours.size() << " contours (" 
                  << matching_contours << " matching priors)" << std::endl;
        
        if (matching_contours > best_matching_contours) {
            best_matching_contours = matching_contours;
            best_mask = test_mask.clone();
            best_threshold = thresh;
        }
    }
    
    cv::Mat mask = best_mask;
    std::cout << "Using threshold " << best_threshold << " with " << best_matching_contours << " matching contours" << std::endl;
    
    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    std::cout << "Found " << contours.size() << " contours" << std::endl;
    
    // Convert to oriented bounding boxes using proper corner ordering
    std::vector<std::vector<cv::Point2f>> candidates;
    for (const auto& contour : contours) {
        if (contour.size() < 5) continue;
        
        // Filter by area - only keep contours with area > 1000 pixels
        double area = cv::contourArea(contour);
        if (area < 1000) continue;
        
        cv::RotatedRect rect = cv::minAreaRect(contour);
        cv::Point2f rect_points[4];
        rect.points(rect_points);
        
        // Convert to vector and order corners clockwise starting from top-left
        std::vector<cv::Point2f> points(rect_points, rect_points + 4);
        points = OBBDetector::order_corners_clockwise_start_tl(points);
        candidates.push_back(points);
        
        std::cout << "Added candidate with area: " << area << std::endl;
    }
    
    std::cout << "Generated " << candidates.size() << " candidate OBBs" << std::endl;
    
    // Test classification using the learned priors
    std::vector<OBB> classified_obbs;
    for (size_t i = 0; i < candidates.size(); i++) {
        const auto& candidate = candidates[i];
        if (candidate.size() == 4) {
            std::cout << "OBB " << i << " corners: (" 
                      << candidate[0].x << "," << candidate[0].y << ") ("
                      << candidate[1].x << "," << candidate[1].y << ") ("
                      << candidate[2].x << "," << candidate[2].y << ") ("
                      << candidate[3].x << "," << candidate[3].y << ")" << std::endl;
            
            // Compute properties for debugging
            auto wh = detector.rect_wh_from_pts(candidate);
            float area = wh.first * wh.second;
            float aspect = wh.first / std::max(wh.second, 1e-6f);
            float angle = detector.long_axis_angle_deg(candidate);
            
            std::cout << "  Properties: area=" << area << ", aspect=" << aspect << ", angle=" << angle << std::endl;
            
            // Classify this candidate using the learned priors
            int classified_class = detector.classify_with_priors(candidate);
            
            if (classified_class >= 0) {
                std::vector<std::string> class_names = {"CylinderVertical", "Ball", "CylinderSide"};
                std::cout << "  -> Classified as class " << classified_class << " (" << class_names[classified_class] << ")" << std::endl;
                OBB obb(candidate[0].x, candidate[0].y, candidate[1].x, candidate[1].y,
                       candidate[2].x, candidate[2].y, candidate[3].x, candidate[3].y,
                       classified_class, 1.0f);
                classified_obbs.push_back(obb);
            } else {
                std::cout << "  -> Rejected (doesn't match any class priors)" << std::endl;
            }
        }
    }
    
    std::cout << "Created " << classified_obbs.size() << " classified OBBs (out of " << candidates.size() << " candidates)" << std::endl;
    
    // Test drawing function
    cv::Mat result_image;
    std::vector<std::string> class_names = {"CylinderVertical", "Ball", "CylinderSide"};
    std::vector<std::vector<unsigned int>> colors = {
        {255, 0, 0},    // Red for class 0 (CylinderVertical)
        {0, 255, 0},    // Green for class 1 (Ball)
        {0, 0, 255}     // Blue for class 2 (CylinderSide)
    };
    
    if (!classified_obbs.empty()) {
        OBBDetector::draw_obb_objects(test_frame, result_image, classified_obbs, class_names, colors, 2);
        
        // Save the result
        std::string output_path = "../csv/frame_200_detected.jpg";
        cv::imwrite(output_path, result_image);
        std::cout << "Saved detection result to: " << output_path << std::endl;
    } else {
        // No detections, just save the original frame
        std::string output_path = "../csv/frame_200_detected.jpg";
        cv::imwrite(output_path, test_frame);
        std::cout << "No detections found, saved original frame to: " << output_path << std::endl;
    }
    
    // Also save the motion mask and difference for debugging
    if (!mask.empty()) {
        cv::imwrite("../csv/frame_200_mask.jpg", mask);
        std::cout << "Saved motion mask: ../csv/frame_200_mask.jpg" << std::endl;
    }
    if (!diff.empty()) {
        cv::imwrite("../csv/frame_200_diff.jpg", diff);
        std::cout << "Saved difference: ../csv/frame_200_diff.jpg" << std::endl;
    }
    if (!green_channel.empty()) {
        cv::imwrite("../csv/frame_200_green.jpg", green_channel);
        std::cout << "Saved green channel: ../csv/frame_200_green.jpg" << std::endl;
    }
    
    std::cout << "OBB detector test completed successfully!" << std::endl;
    return 0;
}
