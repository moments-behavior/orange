# Orange Camera Controller - Project Wiki

## Overview

Orange is a multi-camera capture, streaming, and recording GUI application designed for Emergent cameras. It provides real-time 2D object detection using YOLO models and supports multiple camera synchronization with PTP (Precision Time Protocol).

## Key Features

### Current Capabilities
- **Multi-camera streaming**: Support for multiple Emergent cameras simultaneously
- **PTP synchronization**: Precise time synchronization across cameras
- **GPU-accelerated encoding**: H.264/H.265 video encoding using NVIDIA hardware
- **Real-time 2D object detection**: YOLO-based detection with TensorRT optimization
- **3D triangulation**: Multi-camera 3D position estimation for detected objects
- **Network distribution**: Support for multiple servers to scale camera count
- **Live streaming and recording**: Real-time display and video recording capabilities

### Camera Support
- **Emergent cameras**: Mono and color variants
- **GigE Vision protocol**: Network-based camera communication
- **GPU Direct**: Direct GPU memory access for image processing
- **Configurable parameters**: Gain, exposure, frame rate, resolution, etc.

## Architecture

### Core Components

#### 1. Camera Management (`camera.h`, `camera.cpp`)
- **CameraParams**: Configuration structure for each camera
- **CameraEmergent**: Wrapper for Emergent camera API
- **PTP synchronization**: Time synchronization across cameras
- **Frame buffer management**: Efficient memory handling for video streams

#### 2. Detection System (`yolov8_det.h`, `yolov8_det.cpp`)
- **YOLOv8 class**: TensorRT-optimized YOLO inference
- **GPU preprocessing**: CUDA-based image preprocessing
- **Real-time inference**: Optimized for low-latency detection
- **Bounding box output**: 2D detection results with confidence scores

#### 3. 3D Triangulation (`detect3d.h`, `detect3d.cpp`)
- **Multi-camera fusion**: Combines 2D detections from multiple cameras
- **Camera calibration**: Uses OpenCV calibration for 3D projection
- **Triangulation**: 3D position estimation from 2D detections
- **Projection back**: Projects 3D results back to 2D camera views

#### 4. GUI Interface (`orange.cpp`)
- **ImGui-based interface**: Modern, responsive GUI
- **Real-time visualization**: Live camera feeds with detection overlays
- **Configuration management**: Camera settings and detection parameters
- **Recording controls**: Start/stop recording with PTP synchronization

### Data Flow

```
Camera Streams → GPU Preprocessing → YOLO Detection → 2D Results
                                                      ↓
3D Triangulation ← Camera Calibration ← Multi-camera Fusion
        ↓
3D Position → Projection to 2D → Visualization
```

## Detection Modes

### Current Detection Types
1. **Detect_OFF**: No detection
2. **Detect2D_GLThread**: 2D detection in OpenGL thread
3. **Detect2D_Standoff**: 2D detection in separate thread
4. **Detect3D_Standoff**: 3D triangulation from multiple cameras

### Detection Data Structures
- **Bbox**: 2D bounding box with confidence and label
- **Ball2d**: 2D detection results per camera
- **Ball3d**: 3D triangulated position
- **DetectionDataPerCam**: Per-camera detection state and calibration

## Integration Points for 3D Pose Tracking

### Current 3D Capabilities
The system already has a foundation for 3D tracking:
- **Multi-camera setup**: Multiple synchronized cameras
- **Camera calibration**: Intrinsic and extrinsic parameters
- **3D triangulation**: Existing 3D position estimation
- **Real-time processing**: Low-latency detection pipeline

### Proposed 3D Pose Integration

#### 1. New Detection Mode
Add a new detection mode: `Detect3D_Pose` alongside existing modes

#### 2. Pose Detection Class
Create a new class similar to `YOLOv8` but for pose estimation:
```cpp
class JarvisHybridNet {
    // Similar structure to YOLOv8 but for pose estimation
    // Input: Camera frames
    // Output: 3D pose keypoints
};
```

#### 3. Integration Points

**File: `src/yolov8_det.h`**
- Add new pose detection class
- Extend detection modes enum

**File: `src/detect3d.cpp`**
- Add pose triangulation logic
- Integrate with existing 3D pipeline

**File: `src/orange.cpp`**
- Add GUI controls for pose detection
- Extend visualization for pose keypoints

**File: `src/video_capture.h`**
- Add pose detection mode to `DetectMode` enum
- Extend `CameraEachSelect` structure

#### 4. Data Structures for Pose Tracking
```cpp
struct Pose2d {
    std::atomic<bool> find_pose;
    std::vector<cv::Point2f> keypoints;  // 2D keypoints per camera
    std::vector<float> confidence;       // Confidence per keypoint
    cv::Point2f proj_keypoints[MAX_KEYPOINTS]; // Projected from 3D
};

struct Pose3d {
    std::vector<cv::Point3f> keypoints;  // 3D keypoints
    std::vector<float> confidence;       // Confidence per keypoint
    std::atomic_bool new_detection;
};
```

#### 5. Multi-Camera Pose Fusion
- **Keypoint detection**: Run Jarvis-Hybrid Net on each camera
- **Keypoint matching**: Match keypoints across cameras
- **3D triangulation**: Triangulate 3D pose from 2D keypoints
- **Temporal consistency**: Maintain pose tracking across frames

### Implementation Strategy

#### Phase 1: Basic Integration
1. Add Jarvis-Hybrid Net class structure
2. Integrate with existing detection pipeline
3. Add basic 3D pose triangulation

#### Phase 2: Multi-Camera Fusion
1. Implement keypoint matching across cameras
2. Add robust 3D triangulation for poses
3. Handle occlusions and missing keypoints

#### Phase 3: Optimization
1. GPU optimization for pose inference
2. Temporal smoothing and filtering
3. Real-time performance tuning

## Configuration

### Camera Configuration
Cameras are configured via JSON files in `config/local/` or `config/network/`:
```json
{
    "width": 1920,
    "height": 1080,
    "frame_rate": 60,
    "gpu_direct": true,
    "gpu_id": 0,
    "yolo_model": "path/to/model.engine"
}
```

### Detection Models
- **YOLO models**: TensorRT engine files (.engine)
- **Pose models**: Jarvis-Hybrid Net engine files (to be added)
- **Calibration files**: Camera calibration parameters

## Dependencies

### Core Dependencies
- **Emergent SDK**: Camera communication
- **CUDA Toolkit**: GPU acceleration
- **TensorRT**: Deep learning inference
- **OpenCV**: Computer vision operations
- **FFmpeg**: Video encoding/decoding
- **ImGui**: GUI framework
- **ENET**: Network communication

### Build System
- **CMake**: Build configuration
- **CUDA**: GPU programming
- **OpenGL**: Graphics rendering

## Performance Considerations

### Current Performance
- **GPU acceleration**: CUDA-based preprocessing and inference
- **Multi-threading**: Separate threads for detection and streaming
- **Memory optimization**: Efficient buffer management
- **PTP synchronization**: Sub-microsecond timing accuracy

### 3D Pose Performance Targets
- **Latency**: < 50ms end-to-end
- **Throughput**: 60 FPS per camera
- **Accuracy**: Sub-centimeter 3D precision
- **Robustness**: Handle partial occlusions

## Future Enhancements

### Planned Features
1. **3D Pose Tracking**: Integration of Jarvis-Hybrid Net
2. **Multi-object tracking**: Track multiple poses simultaneously
3. **Temporal filtering**: Smooth pose trajectories
4. **Export capabilities**: Save pose data for analysis
5. **Real-time visualization**: 3D pose overlay in GUI

### Technical Improvements
1. **Model optimization**: Quantization and pruning
2. **Pipeline optimization**: Reduce latency
3. **Error handling**: Robust failure recovery
4. **Documentation**: API documentation and examples

## Getting Started

### Prerequisites
1. Install CUDA Toolkit (12.x)
2. Install Emergent Camera SDK
3. Install FFmpeg 4.4
4. Install OpenCV with CUDA support
5. Install TensorRT

### Building
```bash
git clone --recursive https://github.com/JohnsonLabJanelia/orange.git
cd orange
./build.sh
```

### Running
```bash
./run.sh
```

## Contact

For questions about the software, contact [Jinyao Yan](mailto:yanj11@janelia.hhmi.org).

---

*This wiki will be updated as the 3D pose tracking functionality is implemented.*
