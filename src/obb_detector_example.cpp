/*
 * Example usage of OBBDetector
 * 
 * This example shows how to integrate the OBBDetector with your camera system
 * to perform real-time oriented bounding box detection using learned priors.
 */

#include "obb_detector.h"
#include "camera.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>

int main() {
    // Example camera parameters (adjust according to your setup)
    CameraParams camera_params;
    camera_params.width = 1920;
    camera_params.height = 1080;
    camera_params.frame_rate = 60;
    camera_params.gpu_id = 0;
    camera_params.color = true;
    camera_params.gpu_direct = true;
    camera_params.camera_serial = "Cam2005325";
    
    // CSV files containing hand-labeled training data
    std::vector<std::string> csv_paths = {
        "../csv/Cam2005325_obb.csv"
        // Add more CSV files as needed
    };
    
    // OBB detector parameters
    OBBDetectorParams detector_params;
    detector_params.threshold = 36;           // Motion detection threshold
    detector_params.frame_stride = 1;         // Process every frame
    detector_params.keep_negatives = true;    // Keep frames with no detections
    detector_params.bg_mode = "first";        // Use first frame as background
    detector_params.bg_frames = 30;           // Frames for median background (if used)
    
    // Classification gates (multipliers around IQR bounds)
    detector_params.area_lo_gate = 0.9f;      // Allow slightly smaller than lo
    detector_params.area_hi_gate = 1.1f;      // Allow slightly larger than hi
    detector_params.aspect_lo_gate = 0.9f;
    detector_params.aspect_hi_gate = 1.1f;
    detector_params.angle_k_gate = 2.0f;      // Allow up to 2*IQR angular deviation
    
    // Create OBB detector
    OBBDetector detector(&camera_params, csv_paths, detector_params);
    
    // Initialize (learn priors from CSV files)
    if (!detector.initialize()) {
        std::cerr << "Failed to initialize OBB detector" << std::endl;
        return -1;
    }
    
    // Start detection
    detector.start();
    
    std::cout << "OBB detector started. Press Ctrl+C to stop." << std::endl;
    
    // Main loop - in a real application, this would be integrated with your camera capture loop
    int frame_count = 0;
    while (true) {
        // In a real application, you would:
        // 1. Capture a frame from your camera
        // 2. Get the device pointer to the frame data
        // 3. Call detector.notify_frame_ready(device_ptr, copy_stream)
        
        // For this example, we'll simulate frame processing
        std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
        
        // Get latest detections
        auto detections = detector.get_latest_detections();
        
        if (!detections.empty()) {
            std::cout << "Frame " << frame_count << ": Found " << detections.size() << " detections" << std::endl;
            
            for (const auto& obb : detections) {
                std::cout << "  Class " << obb.class_id << " (confidence: " << obb.confidence << ")" << std::endl;
                std::cout << "    Corners: (" << obb.x1 << "," << obb.y1 << ") "
                          << "(" << obb.x2 << "," << obb.y2 << ") "
                          << "(" << obb.x3 << "," << obb.y3 << ") "
                          << "(" << obb.x4 << "," << obb.y4 << ")" << std::endl;
            }
            
            // Example of drawing OBBs on the frame
            // In a real application, you would get the current frame from your camera
            cv::Mat current_frame; // This would be your actual camera frame
            cv::Mat frame_with_obbs;
            
            // Define class names and colors (similar to YOLOv8)
            std::vector<std::string> class_names = {"sphere", "vertical_cylinder", "horizontal_cylinder"};
            std::vector<std::vector<unsigned int>> colors = {
                {255, 0, 0},    // Red for class 0
                {0, 255, 0},    // Green for class 1  
                {0, 0, 255}     // Blue for class 2
            };
            
            // Draw OBBs on the frame
            OBBDetector::draw_obb_objects(current_frame, frame_with_obbs, detections, 
                                         class_names, colors, 2);
            
            // In a real application, you would display or save frame_with_obbs
            // cv::imshow("OBB Detection", frame_with_obbs);
            // cv::waitKey(1);
        }
        
        frame_count++;
        
        // Stop after processing some frames (for example)
        if (frame_count > 1000) {
            break;
        }
    }
    
    // Stop detector
    detector.stop();
    
    std::cout << "OBB detector stopped." << std::endl;
    
    return 0;
}

/*
 * Integration with existing camera system:
 * 
 * 1. In your camera capture loop, after getting a frame:
 *    detector.notify_frame_ready(device_image_ptr, copy_stream);
 * 
 * 2. In a separate thread or periodically, get detections:
 *    auto detections = detector.get_latest_detections();
 * 
 * 3. Process the detections as needed:
 *    for (const auto& obb : detections) {
 *        // obb.x1, obb.y1, obb.x2, obb.y2, obb.x3, obb.y3, obb.x4, obb.y4
 *        // are the four corners of the oriented bounding box
 *        // obb.class_id is the detected class
 *        // obb.confidence is the detection confidence
 *    }
 * 
 * The detector runs in its own thread and processes frames asynchronously,
 * so it won't block your main camera capture loop.
 */
