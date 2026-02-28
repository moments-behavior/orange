#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "obb_detector.h"
#include "common.hpp"
#include "camera.h"

// Test the two-stage OBB detector: fake YOLO boxes → refine with local mask + priors
int main() {
    std::cout << "Testing OBB Detector (YOLO two-stage mode)..." << std::endl;
    
    std::string frame_path = "../csv/frames/frame_200.jpg";
    cv::Mat test_frame = cv::imread(frame_path);
    if (test_frame.empty()) {
        std::cerr << "Failed to load frame: " << frame_path << std::endl;
        return -1;
    }
    std::cout << "Loaded test frame: " << test_frame.rows << "x" << test_frame.cols << std::endl;
    
    CameraParams camera_params;
    camera_params.width = test_frame.cols;
    camera_params.height = test_frame.rows;
    camera_params.gpu_id = 0;
    camera_params.color = true;
    camera_params.gpu_direct = false;
    camera_params.camera_serial = "Cam2005325";
    
    std::vector<std::string> csv_paths = { "../csv/Cam2005325_obb.csv" };
    OBBDetectorParams detector_params;
    
    OBBDetector detector(&camera_params, csv_paths, detector_params);
    if (!detector.initialize()) {
        std::cerr << "Failed to initialize OBB detector" << std::endl;
        return -1;
    }
    
    std::cout << "\nLearned priors:" << std::endl;
    detector.print_priors();
    
    // Simulate YOLO detections: axis-aligned boxes around known object locations.
    // In production these come from YOLOv8::postprocess().
    // Place test boxes in the center region of the frame where objects typically appear.
    std::vector<Bbox> fake_yolo;
    {
        Bbox b;
        b.rect = cv::Rect_<float>(
            test_frame.cols * 0.35f, test_frame.rows * 0.3f,
            test_frame.cols * 0.12f, test_frame.rows * 0.15f);
        b.prob = 0.9f;
        b.label = 2;
        fake_yolo.push_back(b);
        std::cout << "Fake YOLO box: (" << b.rect.x << ", " << b.rect.y
                  << ", " << b.rect.width << ", " << b.rect.height << ")" << std::endl;
    }
    {
        Bbox b;
        b.rect = cv::Rect_<float>(
            test_frame.cols * 0.55f, test_frame.rows * 0.35f,
            test_frame.cols * 0.12f, test_frame.rows * 0.15f);
        b.prob = 0.85f;
        b.label = 2;
        fake_yolo.push_back(b);
        std::cout << "Fake YOLO box: (" << b.rect.x << ", " << b.rect.y
                  << ", " << b.rect.width << ", " << b.rect.height << ")" << std::endl;
    }
    
    auto refined = detector.refine_yolo_detections(test_frame, fake_yolo);
    std::cout << "\nTwo-stage produced " << refined.size() << " OBBs:" << std::endl;
    
    for (size_t i = 0; i < refined.size(); i++) {
        auto xywhr = detector.obb_to_xywhr(refined[i]);
        std::cout << "  OBB " << i << ": cx=" << xywhr.x << " cy=" << xywhr.y
                  << " w=" << xywhr.w << " h=" << xywhr.h
                  << " theta=" << xywhr.r << " deg" << std::endl;
    }
    
    // Draw and save
    std::vector<std::string> class_names = {"CylinderVertical", "Ball", "CylinderSide"};
    std::vector<std::vector<unsigned int>> colors = {
        {255, 0, 0}, {0, 255, 0}, {0, 0, 255}
    };
    
    cv::Mat result;
    OBBDetector::draw_obb_objects(test_frame, result, refined, class_names, colors, 2);
    
    // Also draw the YOLO axis-aligned boxes for comparison
    for (const auto& b : fake_yolo) {
        cv::rectangle(result, b.rect, cv::Scalar(200, 200, 200), 1);
    }
    
    std::string out = "../csv/frame_200_two_stage.jpg";
    cv::imwrite(out, result);
    std::cout << "\nSaved: " << out << std::endl;
    
    std::cout << "Test completed successfully!" << std::endl;
    return 0;
}
