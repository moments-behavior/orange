# OBB Detector

A C++ implementation of an oriented bounding box (OBB) detector that learns priors from hand-labeled CSV files and performs real-time detection on camera streams using CUDA.

## Overview

The OBB detector implements the same logic as the Python script `autolabel_yolo_obb_with_priors.py` but optimized for real-time processing with CUDA streams. It:

1. **Learns priors** from hand-labeled CSV files containing oriented bounding box annotations
2. **Processes camera streams** in real-time using CUDA
3. **Detects motion** using background subtraction
4. **Generates candidate OBBs** from motion contours
5. **Classifies candidates** against learned priors using area, aspect ratio, and angle features
6. **Returns detected OBBs** with class IDs and corner coordinates

## Files

- `obb_detector.h` - Header file with class definition and interfaces
- `obb_detector.cpp` - Implementation file
- `obb_detector_example.cpp` - Usage example
- `OBB_DETECTOR_README.md` - This documentation

## Dependencies

- OpenCV (for image processing and computer vision operations)
- CUDA (for GPU acceleration)
- NPP (NVIDIA Performance Primitives for image processing)
- Standard C++ libraries

## CSV Format

The detector expects CSV files with the following format:
```csv
frame,obb_id,class_id,corner_x1,corner_y1,corner_x2,corner_y2,corner_x3,corner_y3,corner_x4,corner_y4
32,0,0,1787.83,1164.32,1946.94,1163.46,1947.78,1319.24,1788.67,1320.1
33,0,0,1789.1,1162.87,1949.84,1162.87,1949.84,1322.28,1789.1,1322.28
```

Where:
- `frame`: Frame number
- `obb_id`: Object ID within the frame
- `class_id`: Class ID (0, 1, 2, etc.)
- `corner_x1,y1` to `corner_x4,y4`: Four corner coordinates of the oriented bounding box

### Class Mapping

The class IDs correspond to the following object types (defined in `csv/class_names.txt`):
- **Class 0**: `CylinderVertical` - Vertical cylinders (standing upright)
- **Class 1**: `Ball` - Spherical objects
- **Class 2**: `CylinderSide` - Horizontal cylinders (lying on their side)

## Usage

### Basic Usage

```cpp
#include "obb_detector.h"

// Camera parameters
CameraParams camera_params;
camera_params.width = 1920;
camera_params.height = 1080;
camera_params.gpu_id = 0;
camera_params.color = true;
camera_params.gpu_direct = true;

// CSV files for learning priors
std::vector<std::string> csv_paths = {"csv/Cam2005325_obb.csv"};

// Detector parameters
OBBDetectorParams params;
params.threshold = 36;
params.bg_mode = "first";

// Create and initialize detector
OBBDetector detector(&camera_params, csv_paths, params);
if (!detector.initialize()) {
    std::cerr << "Failed to initialize detector" << std::endl;
    return -1;
}

// Start detection
detector.start();

// In your camera loop:
detector.notify_frame_ready(device_image_ptr, copy_stream);

// Get detections
auto detections = detector.get_latest_detections();

// Draw OBBs on frame
cv::Mat frame_with_obbs;
std::vector<std::string> class_names = {"CylinderVertical", "Ball", "CylinderSide"};
std::vector<std::vector<unsigned int>> colors = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}};
OBBDetector::draw_obb_objects(current_frame, frame_with_obbs, detections, class_names, colors);

for (const auto& obb : detections) {
    // Process detection: obb.x1,y1,x2,y2,x3,y3,x4,y4, obb.class_id, obb.confidence
}

// Stop detector
detector.stop();
```

### Integration with Existing Camera System

The detector is designed to integrate seamlessly with existing camera capture systems:

1. **Thread-safe**: Runs in its own thread and uses mutexes for thread safety
2. **Asynchronous**: Non-blocking frame processing
3. **CUDA-optimized**: Uses CUDA streams for efficient GPU processing
4. **Memory efficient**: Reuses GPU memory buffers

## API Reference

### OBBDetector Class

#### Constructor
```cpp
OBBDetector(CameraParams* params, const std::vector<std::string>& csv_paths, 
            const OBBDetectorParams& detector_params = OBBDetectorParams());
```

#### Methods
- `bool initialize()` - Learn priors from CSV files
- `void start()` - Start the detection thread
- `void stop()` - Stop the detection thread
- `void notify_frame_ready(void* device_image_ptr, cudaStream_t copy_stream)` - Process a new frame
- `std::vector<OBB> get_latest_detections()` - Get latest detection results
- `bool is_running()` - Check if detector is running
- `static void draw_obb_objects(...)` - Draw oriented bounding boxes on image

### OBB Structure
```cpp
struct OBB {
    float x1, y1, x2, y2, x3, y3, x4, y4;  // Four corner coordinates
    int class_id;                           // Detected class ID
    float confidence;                       // Detection confidence
};
```

### OBBDetectorParams Structure
```cpp
struct OBBDetectorParams {
    int threshold = 36;           // Motion detection threshold
    int frame_stride = 1;         // Process every N-th frame
    bool keep_negatives = true;   // Keep frames with no detections
    std::string bg_mode = "first"; // Background mode: "first" or "median"
    int bg_frames = 30;           // Frames for median background
    
    // Classification gates
    float area_lo_gate = 0.9f;    // Area lower bound multiplier
    float area_hi_gate = 1.1f;    // Area upper bound multiplier
    float aspect_lo_gate = 0.9f;  // Aspect ratio lower bound multiplier
    float aspect_hi_gate = 1.1f;  // Aspect ratio upper bound multiplier
    float angle_k_gate = 2.0f;    // Angle deviation multiplier
};
```

## Drawing Oriented Bounding Boxes

The detector includes a static function to draw oriented bounding boxes on camera frames, similar to the YOLOv8 `draw_objects` function:

```cpp
static void draw_obb_objects(const cv::Mat& image, cv::Mat& res,
                            const std::vector<OBB>& obbs,
                            const std::vector<std::string>& class_names,
                            const std::vector<std::vector<unsigned int>>& colors,
                            int line_thickness = 2);
```

### Features:
- **Oriented polygons**: Draws the actual 4-corner oriented bounding boxes as polygons
- **Class labels**: Shows class name and confidence percentage
- **Color coding**: Different colors for different classes
- **Corner markers**: Small circles at each corner for better visibility
- **Text background**: Colored background rectangle for better text readability

### Example:
```cpp
// Define class names and colors
std::vector<std::string> class_names = {"CylinderVertical", "Ball", "CylinderSide"};
std::vector<std::vector<unsigned int>> colors = {
    {255, 0, 0},    // Red for class 0 (CylinderVertical)
    {0, 255, 0},    // Green for class 1 (Ball)
    {0, 0, 255}     // Blue for class 2 (CylinderSide)
};

// Draw OBBs on frame
cv::Mat frame_with_obbs;
OBBDetector::draw_obb_objects(current_frame, frame_with_obbs, detections, 
                             class_names, colors, 2);

// Display or save the result
cv::imshow("OBB Detection", frame_with_obbs);
```

## Algorithm Details

### Prior Learning
1. Parse CSV files to extract class labels and corner coordinates
2. For each class, compute statistics:
   - **Area**: Width × Height of oriented bounding box
   - **Aspect ratio**: Long axis / Short axis
   - **Angle**: Long axis angle in degrees [0, 180)
3. Compute IQR (Interquartile Range) bounds for robust classification

### Motion Detection
1. Background subtraction using first frame or median of multiple frames
2. Use green channel for motion detection (MATLAB-style)
3. Binary thresholding
4. Morphological operations (erosion + dilation)
5. Contour detection

### OBB Generation
1. Find contours from motion mask
2. Compute minimum area rectangle for each contour
3. Extract four corner points
4. Order corners clockwise starting from top-left

### Classification
1. For each candidate OBB, compute features:
   - Area, aspect ratio, and angle
2. Check against learned priors using IQR-based gates
3. Compute normalized deviation score
4. Return class with lowest score if within gates

## Performance Considerations

- **GPU Memory**: Allocates GPU memory for frame processing
- **CUDA Streams**: Uses asynchronous CUDA operations
- **Thread Safety**: Mutex-protected detection results
- **Memory Reuse**: Reuses GPU buffers to minimize allocations

## Error Handling

The detector includes comprehensive error handling:
- CSV parsing errors
- CUDA memory allocation failures
- Invalid camera parameters
- Thread synchronization issues

## Example Output

```
OBBDetector initialized with priors for classes: 0 1 2
Learned priors for class 0 from 40 samples
Learned priors for class 1 from 29 samples
Learned priors for class 2 from 83 samples
OBBDetector thread started for camera: Cam2005325
Background model initialized
Frame 200: Found 2 detections
  Class 0 (CylinderVertical) - confidence: 1.0
    Corners: (1849.23,1261.04) (1897.75,1415.09) (1754.71,1460.14) (1706.19,1306.09)
    Properties: area=24220.9, aspect=1.08, angle=72.5°
  Class 0 (CylinderVertical) - confidence: 1.0
    Corners: (1843.54,849.56) (1972.44,938.113) (1887.41,1061.9) (1758.5,973.345)
    Properties: area=23486.8, aspect=1.04, angle=34.5°
```

## Testing and Validation

The detector has been tested and validated with real video data:

### Test Results
- **Frame 030**: Successfully detected 1 CylinderVertical object
- **Frame 200**: Successfully detected 2 CylinderVertical objects
- **Accuracy**: Objects correctly classified based on learned priors
- **Performance**: Real-time processing on 3208x2200 resolution video

### Test Files
- `obb_detector_simple_test.cpp` - Simple test using extracted video frames
- `extract_video_frames.py` - Python script to extract frames from video for testing
- Test frames: `csv/frames/frame_000.jpg`, `csv/frames/frame_030.jpg`, etc.

### Running Tests
```bash
# Extract test frames from video
python3 extract_video_frames.py csv/Cam2005325.mp4 csv/frames 0,30,200

# Compile and run test
cd build
make obb_detector_simple_test -j4
./obb_detector_simple_test
```

## Troubleshooting

1. **No priors learned**: Check CSV file format and paths
2. **CUDA errors**: Verify GPU memory availability and CUDA installation
3. **No detections**: Adjust threshold and gate parameters
4. **Performance issues**: Check GPU utilization and memory usage
5. **Scale mismatch**: Ensure test images match the resolution of training data
6. **Wrong classifications**: Verify CSV labels match actual object types
