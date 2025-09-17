# 3D Pose Tracking Integration Plan (Revised)

## Overview

This document outlines the detailed plan for integrating 3D pose tracking using the Jarvis-Hybrid Net model into the Orange camera controller application. Based on analysis of the Jarvis system, this plan has been revised to account for the three-stage pipeline architecture.

## Jarvis-Hybrid Net Analysis

### System Architecture
Jarvis-Hybrid Net uses a sophisticated three-stage pipeline:

1. **Center Detection**: EfficientTrack model detects object centers in 2D (320x320 input)
2. **Keypoint Detection**: EfficientTrack model detects keypoints in cropped regions (192x192 input)  
3. **3D Reconstruction**: V2VNet (3D CNN) fuses multi-camera data into 3D poses

### Key Characteristics
- **Models**: 3 separate PyTorch models (.pth files)
- **Input**: 4 synchronized cameras
- **Output**: 4 keypoints (Snout, EarL, EarR, Tail) with 3D coordinates and confidence
- **Performance**: ~30-50ms total latency, ~1GB GPU memory
- **Accuracy**: ~1-2mm 3D precision

### Current System Analysis

### Existing Architecture Strengths
1. **Multi-camera infrastructure**: Already supports multiple synchronized cameras
2. **GPU-accelerated pipeline**: CUDA-based preprocessing and inference
3. **3D triangulation framework**: Existing 3D position estimation system
4. **Real-time processing**: Low-latency detection pipeline
5. **Modular design**: Clean separation between detection, triangulation, and visualization
6. **TensorRT integration**: Existing YOLO models use TensorRT engines
7. **Camera calibration**: OpenCV-based calibration system already implemented

### Key Integration Points

#### 1. Model Conversion
**New**: Convert Jarvis PyTorch models to TensorRT engines
**Files**: `convert_jarvis_to_tensorrt.py` (created)
**Output**: 3 TensorRT engines + model info JSON

#### 2. Detection Pipeline (`src/yolov8_det.h`, `src/yolov8_det.cpp`)
**Current**: YOLO-based 2D object detection
**Integration**: Add Jarvis-Hybrid Net three-stage pipeline

#### 3. 3D Processing (`src/detect3d.h`, `src/detect3d.cpp`)
**Current**: Ball/object 3D triangulation
**Integration**: Replace with Jarvis 3D reconstruction pipeline

#### 4. Data Structures (`src/realtime_tool.h`)
**Current**: Ball2d, Ball3d structures
**Integration**: Add Jarvis-specific pose structures

#### 5. GUI Interface (`src/orange.cpp`)
**Current**: Detection mode selection and visualization
**Integration**: Add pose detection controls and 3D pose visualization

## Model Conversion Process

### Step 1: Convert PyTorch Models to TensorRT
**Script**: `convert_jarvis_to_tensorrt.py`
**Input**: 3 PyTorch models (.pth files)
**Output**: 3 TensorRT engines (.engine files) + model_info.json

```bash
# Run conversion script
python convert_jarvis_to_tensorrt.py \
    --config jarvis/config.yaml \
    --output_dir models/tensorrt/
```

**Output Files**:
- `center_detect.engine` - Center detection model
- `keypoint_detect.engine` - Keypoint detection model  
- `hybrid_net.engine` - 3D reconstruction model
- `model_info.json` - Model configuration and metadata

### Step 2: Model Integration
The converted engines will be integrated into Orange's existing TensorRT pipeline, similar to how YOLO models are currently handled.

### Step 3: Understanding the Complete Pipeline
**Key Insight**: Jarvis uses a sophisticated **three-stage pipeline** that's more complex than simple "per-camera 3D pose detection":

1. **Center Detection**: All 4 cameras → 2D centers → 3D center triangulation
2. **Keypoint Detection**: 3D center → project to 2D → crop regions → 2D keypoints  
3. **3D Reconstruction**: 2D keypoints → 3D heatmap → V2VNet → 3D keypoints

**Output**: Both 3D center point AND 3D keypoints (4 keypoints: Snout, EarL, EarR, Tail)
**Visualization**: Project both 3D center and 3D keypoints back to 2D for camera overlay

*See `JARVIS_3D_PIPELINE_DETAILED.md` for complete implementation details.*

## Detailed Implementation Plan

### Phase 0: Model Conversion (Prerequisite)
1. **Setup Environment**: Install PyTorch, TensorRT, torch2trt
2. **Run Conversion**: Execute conversion script
3. **Validate Models**: Test converted engines
4. **Document Integration**: Create integration guide

### Phase 1: Core Infrastructure

#### 1.1 Extend Detection Modes
**File**: `src/video_capture.h`
```cpp
enum DetectMode {
    Detect_OFF,
    Detect2D_GLThread,
    Detect2D_Standoff,
    Detect3D_Standoff,
    Detect3D_Pose  // NEW: 3D pose detection mode
};
```

#### 1.2 Add Jarvis Pose Data Structures
**File**: `src/realtime_tool.h`
```cpp
// Jarvis-Hybrid Net specific constants
#define JARVIS_NUM_KEYPOINTS 4  // Snout, EarL, EarR, Tail
#define JARVIS_NUM_CAMERAS 4    // Default number of cameras
#define JARVIS_CENTER_IMG_SIZE 320
#define JARVIS_KEYPOINT_IMG_SIZE 192

// Jarvis keypoint names (from config)
enum JarvisKeypoint {
    SNOUT = 0,
    EAR_L = 1, 
    EAR_R = 2,
    TAIL = 3
};

struct JarvisCenter2d {
    std::atomic<bool> find_center;
    cv::Point2f center;  // 2D center coordinates
    float confidence;    // Detection confidence
    cv::Point2f proj_center;  // Projected from 3D
    
    JarvisCenter2d() : find_center(false), confidence(0.0f) {}
};

struct JarvisKeypoints2d {
    std::atomic<bool> find_keypoints;
    cv::Point2f keypoints[JARVIS_NUM_KEYPOINTS];  // 2D keypoint coordinates
    float confidence[JARVIS_NUM_KEYPOINTS];       // Confidence per keypoint
    cv::Point2f proj_keypoints[JARVIS_NUM_KEYPOINTS]; // Projected from 3D
    
    JarvisKeypoints2d() : find_keypoints(false) {
        for (int i = 0; i < JARVIS_NUM_KEYPOINTS; ++i) {
            confidence[i] = 0.0f;
        }
    }
};

struct JarvisCenter3d {
    cv::Point3f center;           // 3D center coordinates (x, y, z)
    float confidence;             // Detection confidence
    std::atomic_bool new_detection;
    
    JarvisCenter3d() : confidence(0.0f), new_detection(false) {}
};

struct JarvisPose3d {
    cv::Point3f keypoints[JARVIS_NUM_KEYPOINTS];  // 3D keypoint coordinates
    float confidence[JARVIS_NUM_KEYPOINTS];       // Confidence per keypoint
    std::atomic_bool new_detection;
    
    JarvisPose3d() : new_detection(false) {
        for (int i = 0; i < JARVIS_NUM_KEYPOINTS; ++i) {
            confidence[i] = 0.0f;
        }
    }
};

// Extend DetectionDataPerCam
struct DetectionDataPerCam {
    bool has_calibration_results;
    std::string calibration_file;
    CameraCalibResults camera_calib;
    Aruco2d marker2d;
    Ball2d ball2d;
    JarvisCenter2d jarvis_center;      // NEW: Jarvis center detection
    JarvisKeypoints2d jarvis_keypoints; // NEW: Jarvis keypoint detection
};

// Extend Detection3d
struct Detection3d {
    Aruco3d marker3d;
    Ball3d ball3d;
    JarvisCenter3d jarvis_center;  // NEW: Jarvis 3D center results
    JarvisPose3d jarvis_pose;      // NEW: Jarvis 3D pose results
};
```

#### 1.3 Create Jarvis-Hybrid Net Class
**File**: `src/jarvis_pose_det.h` (NEW)
```cpp
#ifndef JARVIS_POSE_DET_H
#define JARVIS_POSE_DET_H

#include "NvInferPlugin.h"
#include "common.hpp"
#include "fstream"
#include <nppi.h>

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
    
    // Helper functions
    void preprocess_center_detection(unsigned char **d_input_images, int img_width, int img_height);
    void preprocess_keypoint_detection(unsigned char **d_input_images, int img_width, int img_height);
    void postprocess_center_detection();
    void postprocess_keypoint_detection();
    void postprocess_3d_reconstruction();
    void triangulate_3d_center();
    void crop_keypoint_regions();
};

#endif // JARVIS_POSE_DET_H
```

### Phase 2: Detection Integration

#### 2.1 Extend Camera Selection Structure
**File**: `src/video_capture.h`
```cpp
struct CameraEachSelect {
    // ... existing members ...
    std::string pose_model;  // NEW: Path to Jarvis-Hybrid Net model
    int max_poses = 1;       // NEW: Maximum number of poses to detect
    // ... rest of existing members ...
};
```

#### 2.2 Integrate Pose Detection in Camera Thread
**File**: `src/video_capture.cpp` (or relevant camera processing file)
```cpp
// Add pose detection logic in the camera acquisition loop
if (camera_select->detect_mode == Detect3D_Pose) {
    // Run Jarvis-Hybrid Net inference
    pose_detector->preprocess_gpu();
    pose_detector->infer();
    
    std::vector<PoseKeypoint> detected_poses;
    pose_detector->postprocess(detected_poses);
    
    // Update detection2d[i].pose2d with results
    if (!detected_poses.empty()) {
        detection2d[i].pose2d.find_pose.store(true);
        // Copy keypoints and confidence scores
        for (size_t k = 0; k < detected_poses.size() && k < MAX_POSE_KEYPOINTS; ++k) {
            detection2d[i].pose2d.keypoints[k] = detected_poses[k].position;
            detection2d[i].pose2d.confidence[k] = detected_poses[k].confidence;
        }
    } else {
        detection2d[i].pose2d.find_pose.store(false);
    }
    
    // Signal that detection is ready
    camera_select->frame_detect_state.store(State_Frame_Detection_Ready);
}
```

### Phase 3: 3D Pose Triangulation

#### 3.1 Extend 3D Detection Processing
**File**: `src/detect3d.cpp`
```cpp
// Add pose triangulation function
bool find_pose3d(TriangulatePoints *pose_2d, Pose3d *pose3d) {
    if (pose_2d->detected_cameras.size() < 2) {
        return false;  // Need at least 2 cameras for triangulation
    }
    
    // For each keypoint, triangulate from multiple camera views
    for (int kp = 0; kp < MAX_POSE_KEYPOINTS; ++kp) {
        std::vector<cv::Point2f> keypoint_2d;
        std::vector<CameraCalibResults*> calib_for_keypoint;
        
        // Collect 2D keypoints from all cameras that detected this keypoint
        for (size_t cam_idx = 0; cam_idx < pose_2d->detected_cameras.size(); ++cam_idx) {
            int cam_id = pose_2d->detected_cameras[cam_idx];
            if (detection2d[cam_id].pose2d.confidence[kp] > 0.5f) {  // Confidence threshold
                keypoint_2d.push_back(detection2d[cam_id].pose2d.keypoints[kp]);
                calib_for_keypoint.push_back(pose_2d->calib_results[cam_idx]);
            }
        }
        
        if (keypoint_2d.size() >= 2) {
            // Triangulate 3D keypoint
            cv::Mat triangulated = triangulate_points(keypoint_2d, calib_for_keypoint);
            if (triangulated.rows > 0) {
                pose3d->keypoints[kp] = cv::Point3f(
                    triangulated.at<float>(0, 0),
                    triangulated.at<float>(0, 1),
                    triangulated.at<float>(0, 2)
                );
                // Average confidence from all cameras
                float avg_confidence = 0.0f;
                for (size_t i = 0; i < keypoint_2d.size(); ++i) {
                    avg_confidence += detection2d[pose_2d->detected_cameras[i]].pose2d.confidence[kp];
                }
                pose3d->confidence[kp] = avg_confidence / keypoint_2d.size();
            }
        }
    }
    
    // Calculate center of mass for tracking
    cv::Point3f center(0, 0, 0);
    int valid_keypoints = 0;
    for (int kp = 0; kp < MAX_POSE_KEYPOINTS; ++kp) {
        if (pose3d->confidence[kp] > 0.3f) {
            center += pose3d->keypoints[kp];
            valid_keypoints++;
        }
    }
    if (valid_keypoints > 0) {
        pose3d->center = center / valid_keypoints;
    }
    
    return valid_keypoints > 5;  // Require at least 5 valid keypoints
}

// Modify detection3d_proc to handle pose detection
void detection3d_proc(CameraControl *camera_control,
                      CameraEachSelect *cameras_select, int num_cameras) {
    
    // ... existing ball detection code ...
    
    // Add pose detection processing
    std::vector<int> pose_cam_idx;
    for (int i = 0; i < num_cameras; i++) {
        if (cameras_select[i].detect_mode == Detect3D_Pose) {
            pose_cam_idx.push_back(i);
        }
    }
    
    if (!pose_cam_idx.empty()) {
        TriangulatePoints pose2d_all_cams;
        
        // Collect pose detections from all cameras
        for (int idx : pose_cam_idx) {
            if (detection2d[idx].pose2d.find_pose.load()) {
                pose2d_all_cams.detected_cameras.push_back(idx);
                pose2d_all_cams.calib_results.push_back(&detection2d[idx].camera_calib);
            }
        }
        
        // Triangulate 3D pose
        detection3d.pose3d.new_detection.store(
            find_pose3d(&pose2d_all_cams, &detection3d.pose3d));
        
        // Project 3D pose back to all streaming cameras
        if (detection3d.pose3d.new_detection.load()) {
            for (int i = 0; i < num_cameras; i++) {
                if (cameras_select[i].stream_on && 
                    detection2d[i].has_calibration_results) {
                    
                    cv::Mat image_pts;
                    CameraCalibResults *cam_calib = &detection2d[i].camera_calib;
                    
                    // Project all keypoints
                    cv::projectPoints(detection3d.pose3d.keypoints, 
                                    cam_calib->rvec, cam_calib->tvec, 
                                    cam_calib->k, cam_calib->dist_coeffs, 
                                    image_pts);
                    
                    // Update projected keypoints
                    for (int kp = 0; kp < MAX_POSE_KEYPOINTS; ++kp) {
                        detection2d[i].pose2d.proj_keypoints[kp].x = 
                            image_pts.at<float>(kp, 0);
                        detection2d[i].pose2d.proj_keypoints[kp].y = 
                            image_pts.at<float>(kp, 1);
                    }
                }
            }
        }
    }
}
```

### Phase 4: GUI Integration

#### 4.1 Add Pose Detection Controls
**File**: `src/orange.cpp`
```cpp
// In the camera control section, add pose detection options
if (ImGui::BeginTable("Camera Control Setting", 6,  // Increased column count
                      ImGuiTableFlags_Resizable |
                      ImGuiTableFlags_NoSavedSettings |
                      ImGuiTableFlags_Borders)) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::Text("name");
    ImGui::TableNextColumn();
    ImGui::Text("serial");
    ImGui::TableNextColumn();
    ImGui::Text("stream");
    ImGui::TableNextColumn();
    ImGui::Text("record");
    ImGui::TableNextColumn();
    ImGui::Text("yolo");
    ImGui::TableNextColumn();
    ImGui::Text("pose");  // NEW: Pose detection column
    
    for (int i = 0; i < num_cameras; i++) {
        ImGui::TableNextRow();
        // ... existing columns ...
        
        // NEW: Pose detection mode selection
        ImGui::TableNextColumn();
        int current_pose_mode = 
            (cameras_select[i].detect_mode == Detect3D_Pose) ? 1 : 0;
        sprintf(temp_string, "##pose_mode%d", i);
        if (ImGui::Checkbox(temp_string, &current_pose_mode)) {
            if (current_pose_mode && cameras_select[i].pose_model.empty()) {
                current_pose_mode = 0;
                error_message = "Specify pose model first in Camera Property.";
                show_error = true;
            } else {
                cameras_select[i].detect_mode = 
                    current_pose_mode ? Detect3D_Pose : Detect_OFF;
            }
        }
    }
    ImGui::EndTable();
}
```

#### 4.2 Add Pose Visualization
**File**: `src/orange.cpp`
```cpp
// In the camera display section, add pose visualization
if (detection2d[i].has_calibration_results) {
    // ... existing ball detection visualization ...
    
    // NEW: Pose visualization
    if (detection2d[i].pose2d.find_pose.load()) {
        // Draw 2D detected keypoints
        for (int kp = 0; kp < MAX_POSE_KEYPOINTS; ++kp) {
            if (detection2d[i].pose2d.confidence[kp] > 0.3f) {
                std::string kp_name = "pose_kp_" + std::to_string(i) + "_" + std::to_string(kp);
                draw_ball_center(
                    detection2d[i].pose2d.keypoints[kp],
                    cameras_params[i].height,
                    (ImVec4)ImColor::HSV(kp * 0.1f, 0.8f, 1.0f),  // Different color per keypoint
                    kp_name, ImPlotMarker_Circle, 4.0);
            }
        }
        
        // Draw projected 3D pose keypoints
        if (detection3d.pose3d.new_detection.load()) {
            for (int kp = 0; kp < MAX_POSE_KEYPOINTS; ++kp) {
                if (detection3d.pose3d.confidence[kp] > 0.3f) {
                    std::string proj_kp_name = "pose_proj_" + std::to_string(i) + "_" + std::to_string(kp);
                    draw_ball_center(
                        detection2d[i].pose2d.proj_keypoints[kp],
                        cameras_params[i].height,
                        (ImVec4)ImColor::HSV(kp * 0.1f, 0.6f, 0.8f),  // Slightly different color
                        proj_kp_name, ImPlotMarker_Cross, 6.0);
                }
            }
        }
    }
}
```

### Phase 5: Model Integration

#### 5.1 Model Loading and Initialization
**File**: `src/jarvis_pose_det.cpp` (NEW)
```cpp
#include "jarvis_pose_det.h"
#include "common.hpp"
#include "utils.h"
#include <npp.h>
#include <nvToolsExt.h>

JarvisPoseDetector::JarvisPoseDetector(const std::string &engine_file_path, 
                                     int width, int height,
                                     cudaStream_t stream, 
                                     unsigned char *d_input_image,
                                     const NppStreamContext &npp_ctx)
    : stream(stream), d_input_image(d_input_image), npp_ctx_(npp_ctx) {
    
    // Similar initialization to YOLOv8 but adapted for pose estimation
    img_width = width;
    img_height = height;
    
    // Load TensorRT engine
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char *trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
    
    initLibNvInferPlugins(&this->gLogger, "");
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);
    
    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->engine != nullptr);
    delete[] trtModelStream;
    this->context = this->engine->createExecutionContext();
    assert(this->context != nullptr);
    
    // Setup input/output bindings (similar to YOLOv8)
    this->num_bindings = this->engine->getNbIOTensors();
    // ... rest of initialization similar to YOLOv8
}

void JarvisPoseDetector::postprocess(std::vector<PoseKeypoint> &poses) {
    poses.clear();
    
    // Parse model output to extract keypoints
    // This will depend on the specific output format of Jarvis-Hybrid Net
    // Example assuming output format: [batch, num_poses, num_keypoints, 3] (x, y, confidence)
    
    float *output_data = static_cast<float *>(this->host_ptrs[0]);
    int *num_poses = static_cast<int *>(this->host_ptrs[1]);  // If model outputs pose count
    
    // Process each detected pose
    for (int p = 0; p < num_poses[0] && p < MAX_DETECTED_POSES; ++p) {
        for (int kp = 0; kp < MAX_POSE_KEYPOINTS; ++kp) {
            int offset = p * MAX_POSE_KEYPOINTS * 3 + kp * 3;
            
            float x = output_data[offset] * this->pparam.ratio + this->pparam.dw;
            float y = output_data[offset + 1] * this->pparam.ratio + this->pparam.dh;
            float confidence = output_data[offset + 2];
            
            if (confidence > 0.3f) {  // Confidence threshold
                PoseKeypoint keypoint;
                keypoint.position = cv::Point2f(x, y);
                keypoint.confidence = confidence;
                keypoint.keypoint_id = kp;
                poses.push_back(keypoint);
            }
        }
    }
}
```

## Implementation Timeline

### Week 0: Model Conversion (Prerequisite)
- [ ] Setup PyTorch and TensorRT environment
- [ ] Run model conversion script
- [ ] Validate converted TensorRT engines
- [ ] Test model loading and basic inference

### Week 1-2: Core Infrastructure
- [ ] Extend detection modes and data structures
- [ ] Create JarvisPoseDetector class skeleton
- [ ] Implement three-stage pipeline architecture
- [ ] Update CMakeLists.txt for new files

### Week 3-4: Detection Integration
- [ ] Implement center detection stage
- [ ] Add keypoint detection stage
- [ ] Integrate with camera pipeline
- [ ] Test single-camera pose detection

### Week 5-6: 3D Reconstruction
- [ ] Implement 3D reconstruction stage
- [ ] Add camera calibration integration
- [ ] Test multi-camera 3D pose estimation
- [ ] Validate accuracy with ground truth

### Week 7-8: GUI and Visualization
- [ ] Add pose detection controls to GUI
- [ ] Implement 3D pose visualization
- [ ] Add configuration options for Jarvis models
- [ ] Create pose tracking display

### Week 9-10: Optimization and Testing
- [ ] Performance optimization for real-time processing
- [ ] Memory management optimization
- [ ] Integration testing with existing features
- [ ] End-to-end system validation

## Testing Strategy

### Unit Tests
1. **Pose Detection**: Test JarvisPoseDetector with sample images
2. **3D Triangulation**: Test pose triangulation with known 3D positions
3. **Multi-camera**: Test pose detection across multiple cameras

### Integration Tests
1. **Real-time Performance**: Measure latency and throughput
2. **Accuracy Validation**: Compare with ground truth pose data
3. **System Stability**: Long-running tests with multiple cameras

### Performance Benchmarks
- **Latency**: < 50ms end-to-end pose detection
- **Throughput**: 60 FPS per camera
- **Accuracy**: < 5cm 3D position error
- **Memory**: < 2GB additional GPU memory usage

## Risk Mitigation

### Technical Risks
1. **Model Compatibility**: Ensure Jarvis-Hybrid Net outputs are compatible
2. **Performance**: Optimize for real-time processing
3. **Multi-person**: Handle multiple poses simultaneously

### Mitigation Strategies
1. **Prototype Early**: Create minimal working version first
2. **Incremental Integration**: Add features gradually
3. **Fallback Options**: Maintain existing functionality if pose detection fails

## Success Criteria

### Functional Requirements
- [ ] Convert Jarvis PyTorch models to TensorRT engines
- [ ] Implement three-stage Jarvis pipeline (center → keypoint → 3D)
- [ ] Detect 3D center point from 4 synchronized cameras
- [ ] Detect 3D poses (4 keypoints: Snout, EarL, EarR, Tail) from 4 synchronized cameras
- [ ] Real-time visualization of both 3D center and 3D keypoints projected to 2D
- [ ] Configurable detection parameters
- [ ] Integration with existing recording system

### Performance Requirements
- [ ] < 50ms total latency for complete pipeline
- [ ] 60 FPS processing per camera
- [ ] < 2mm 3D position accuracy (Jarvis specification)
- [ ] < 1.5GB GPU memory usage
- [ ] Stable operation for extended periods

### Usability Requirements
- [ ] Intuitive GUI controls for Jarvis models
- [ ] Clear visualization of 3D pose results
- [ ] Easy model configuration and loading
- [ ] Comprehensive documentation
- [ ] Seamless integration with existing Orange features

### Technical Requirements
- [ ] Successful model conversion from .pth to .engine
- [ ] Three-stage pipeline working correctly
- [ ] Camera calibration integration
- [ ] Multi-camera synchronization
- [ ] Error handling and fallback mechanisms

---

*This plan provides a comprehensive roadmap for integrating 3D pose tracking into the Orange camera controller system.*
