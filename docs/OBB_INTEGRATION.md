# OBB Detector Real-Time Integration Guide

This guide provides step-by-step instructions for integrating the Oriented Bounding Box (OBB) detector into the main Orange project for real-time detection and display.

## Overview

The integration adds OBB detection capability alongside the existing YOLO detection system. Users can enable OBB detection for specific cameras through config files, and the system will:

1. Learn priors from CSV files
2. Process camera frames in real-time
3. Detect oriented bounding boxes using motion detection
4. Classify objects against learned priors
5. Draw 4-corner OBB overlays directly on the camera stream

## Visual Output

**What you'll see on screen:**
- **Oriented Bounding Boxes**: 4-corner polygons drawn around detected objects (not axis-aligned rectangles)
- **Class-based colors**: Each object class gets a different color (CylinderVertical=Red, Ball=Green, CylinderSide=Blue)
- **Real-time overlay**: OBBs are drawn directly on the OpenGL camera stream, updating every frame
- **Multiple objects**: Each detected object gets its own oriented bounding box

**Example visual result:**
- A cylinder lying at an angle will have a rotated rectangle around it (not a straight rectangle)
- A ball will have a square/rectangle around it that may be rotated based on its orientation
- Multiple objects will each have their own colored oriented bounding box

**Debug output in console:**
```
OBB Detection Results: 2 objects detected
  Object 0: Class=0, Confidence=0.850, Corners=(100.0,50.0) (200.0,50.0) (200.0,150.0) (100.0,150.0)
  Object 1: Class=1, Confidence=0.720, Corners=(300.0,100.0) (350.0,100.0) (350.0,150.0) (300.0,150.0)
```

The OBB overlays are drawn using the existing GPU drawing infrastructure (`gpu_draw_obb`), so they appear seamlessly integrated with the camera stream display.

## Architecture

The system has two main processing paths:

- **FrameDetector Path**: Processes frames on a separate thread (used by YOLO)
- **OpenGL Display Path**: Processes frames inline with display rendering

**RESOLUTION CONSIDERATION**: The OpenGL Display path downsamples frames for display performance (`camera_select->downsample` can be 1, 2, 4, 8, or 16). However, this is actually fine for OBB detection because:

- **Scale Consistency**: Downsampling is uniform, preserving object proportions
- **Simpler Architecture**: Process and draw at the same resolution
- **No Coordinate Scaling**: OBB coordinates match display coordinates directly
- **Better Performance**: No thread synchronization or coordinate conversion needed

**Recommended Approach**: Use the **OpenGL Display Path** because:
- Much simpler integration
- Direct GPU processing and drawing
- No thread synchronization complexity
- Priors can be learned/scaled to match display resolution

## Integration Steps

### Step 0: Enable OBB via Per-Camera Config (no GUI)

OBB configuration is automatically loaded through the existing JSON config system. Orange already loads camera configurations from JSON files, and we've extended this system to support OBB fields.

**Config file locations:**
- Primary: `config/<serial>.json` (e.g., `config/2002496.json`)
- Fallback: `example_config/<serial>.json` (e.g., `example_config/2012856.json`)

**Expected JSON fields (all optional, sensible defaults used if missing):**
```json
{
  "name": "Cam0",
  "width": 3208,
  "height": 2200,
  "frame_rate": 30,
  "gain": 1500,
  "iris": 0,
  "focus": 345,
  "exposure": 2500,
  "pixel_format": "BayerRG8",
  "gpu_id": 0,
  "color_temp": "CT_3000K",
  "gpu_direct": false,
  "color": true,
  "yolo": "/path/to/yolo.engine",
  "enable_obb": true,
  "obb_csv_path": "csv_prior/Cam2005325_obb.csv",
  "obb_threshold": 30.0,
  "obb_bg_frames": 10
}
```

**Implementation:**
The existing `load_camera_json_config_files()` function in `project.cpp` has been extended to automatically load OBB configuration fields when present in the JSON config file. No additional code is needed - the system will automatically pick up OBB settings from the camera's JSON config file.

### Step 1: Add OBB Support to Camera Selection

First, extend the camera selection structure to include OBB configuration.

**File: `src/camera.h`** (or wherever `CameraEachSelect` is defined)

```cpp
struct CameraEachSelect {
    // ... existing fields ...
    
    // OBB Detection Configuration
    bool enable_obb = false;
    std::string obb_csv_path = "";
    float obb_threshold = 30.0f;
    int obb_bg_frames = 10;
};
```

### Step 2: Extend OpenGL Display for OBB

Modify `COpenGLDisplay` to support OBB detection alongside YOLO, processing frames at display resolution.

**File: `src/opengldisplay.h`**

```cpp
#include "obb_detector.h"

class COpenGLDisplay : public CThreadWorker {
public:
    // ... existing members ...
    
    // OBB Detection
    OBBDetector *obb_detector = nullptr;
    float *d_obb_points;  // GPU buffer for OBB corner points
    
private:
    // ... existing members ...
};
```

**File: `src/opengldisplay.cpp`**

Add OBB initialization in constructor:

```cpp
COpenGLDisplay::COpenGLDisplay(const char *name, CameraParams *camera_params,
                               CameraEachSelect *camera_select, unsigned char *display_buffer,
                               INDIGOSignalBuilder *indigo_signal_builder)
    : CThreadWorker(name), camera_params(camera_params),
      camera_select(camera_select), display_buffer(display_buffer),
      indigo_signal_builder(indigo_signal_builder) {
    
    // ... existing initialization ...
    
    // Initialize OBB detector if enabled
    if (camera_select->enable_obb) {
        obb_detector = new OBBDetector();
        
        // Scale CSV priors to match display resolution
        std::string scaled_csv_path = camera_select->obb_csv_path;
        if (camera_select->downsample != 1) {
            // Create scaled version of CSV for display resolution
            scaled_csv_path = scale_csv_for_display_resolution(
                camera_select->obb_csv_path, camera_select->downsample);
        }
        
        // Initialize with scaled CSV priors
        if (!obb_detector->initialize(scaled_csv_path)) {
            std::cout << "Failed to initialize OBB detector with CSV: " 
                      << scaled_csv_path << std::endl;
            delete obb_detector;
            obb_detector = nullptr;
        } else {
            std::cout << "OBB detector initialized successfully at display resolution" << std::endl;
            
            // Background will be built automatically from first N frames
            // No static background loading needed
            
            // Set detection parameters
            OBBDetectorParams params;
            params.threshold = camera_select->obb_threshold;
            params.bg_frames = camera_select->obb_bg_frames;
            obb_detector->set_params(params);
        }
    }
}
```

Add OBB cleanup in destructor:

```cpp
COpenGLDisplay::~COpenGLDisplay() {
    // ... existing cleanup ...
    
    if (obb_detector) {
        delete obb_detector;
        obb_detector = nullptr;
    }
    
    if (d_obb_points) {
        cudaFree(d_obb_points);
        d_obb_points = nullptr;
    }
}
```

### Step 3: Add OBB Processing to Display Thread

Modify the `ThreadRunning()` method to process OBB detection at display resolution:

```cpp
void COpenGLDisplay::ThreadRunning() {
    // ... existing initialization ...
    
    // Allocate OBB GPU resources if needed
    if (camera_select->enable_obb && obb_detector) {
        cudaMalloc((void **)&d_obb_points, sizeof(float) * 8 * 10); // Support up to 10 OBBs
    }
    
    std::vector<Bbox> objs;
    std::vector<Bbox> objs_last_frame;
    std::vector<OBB> obb_detections;  // Add OBB results
    
    while (IsMachineOn()) {
        // ... existing frame processing ...
        
        // YOLO Detection (existing code)
        if (camera_select->detect_mode == Detect2D_GLThread) {
            // ... existing YOLO processing ...
        }
        
        // OBB Detection (new code)
        if (camera_select->enable_obb && obb_detector) {
            // Get frame at display resolution for OBB processing
            cv::Mat cpu_frame;
            if (camera_select->downsample != 1) {
                // Use resized frame
                cpu_frame = cv::Mat(output_image_size.height, output_image_size.width, CV_8UC4);
                CHECK(cudaMemcpy2D(cpu_frame.data, output_image_size.width * 4,
                                   d_resize, output_image_size.width * 4,
                                   output_image_size.width * 4, output_image_size.height,
                                   cudaMemcpyDeviceToHost));
                cv::cvtColor(cpu_frame, cpu_frame, cv::COLOR_BGRA2RGB);
            } else {
                // Use original frame
                cpu_frame = cv::Mat(camera_params->height, camera_params->width, CV_8UC4);
                CHECK(cudaMemcpy2D(cpu_frame.data, camera_params->width * 4,
                                   debayer.d_debayer, camera_params->width * 4,
                                   camera_params->width * 4, camera_params->height,
                                   cudaMemcpyDeviceToHost));
                cv::cvtColor(cpu_frame, cpu_frame, cv::COLOR_BGRA2RGB);
            }
            
            // Process frame with OBB detector
            obb_detector->process_frame(cpu_frame);
            obb_detections = obb_detector->get_latest_detections();
            
            // Draw OBB overlays on GPU and print debug info
            if (obb_detections.size() > 0) {
                std::cout << "OBB Detection Results: " << obb_detections.size() << " objects detected" << std::endl;
                
                for (size_t i = 0; i < obb_detections.size() && i < 10; i++) {
                    const OBB& obb = obb_detections[i];
                    
                    // Debug print for each object
                    std::cout << "  Object " << i << ": Class=" << obb.class_id 
                              << ", Confidence=" << std::fixed << std::setprecision(3) << obb.confidence
                              << ", Corners=(" << obb.x1 << "," << obb.y1 << ") (" 
                              << obb.x2 << "," << obb.y2 << ") (" << obb.x3 << "," << obb.y3 
                              << ") (" << obb.x4 << "," << obb.y4 << ")" << std::endl;
                    
                    // Copy OBB corners to GPU (already at display resolution)
                    float obb_points[8] = {
                        obb.x1, obb.y1,  // Top-left
                        obb.x2, obb.y2,  // Top-right  
                        obb.x3, obb.y3,  // Bottom-right
                        obb.x4, obb.y4   // Bottom-left
                    };
                    
                    CHECK(cudaMemcpyAsync(d_obb_points + i * 8, obb_points, 
                                         sizeof(float) * 8, cudaMemcpyHostToDevice, 0));
                    
                    // Draw OBB on the appropriate buffer
                    if (camera_select->downsample != 1) {
                        // Draw on resized buffer
                        gpu_draw_obb(d_resize, output_image_size.width, 
                                    output_image_size.height, d_obb_points + i * 8, 
                                    obb.class_id, 0);
                    } else {
                        // Draw on original buffer
                        gpu_draw_obb(debayer.d_debayer, camera_params->width, 
                                    camera_params->height, d_obb_points + i * 8, 
                                    obb.class_id, 0);
                    }
                }
            } else {
                // Optional: print when no objects detected (can be commented out for less noise)
                // std::cout << "OBB Detection: No objects detected" << std::endl;
            }
        }
        
        // ... existing display buffer copy ...
    }
}
```

### Step 4: Create GPU OBB Drawing Function

Add a new GPU kernel for drawing oriented bounding boxes.

**File: `src/kernel.cu`**

```cpp
__global__ void gpu_draw_obb(unsigned char* src, const int width, const int height, 
                            float* d_points, int label_id, double current_time) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < width) && (y < height)) {
        // Draw 4 connected lines for OBB
        for (int i = 0; i < 4; i++) {
            int idx0 = i;
            int idx1 = (i + 1) % 4; 

            float x0 = d_points[idx0 * 2];
            float y0 = d_points[idx0 * 2 + 1];
            float x1 = d_points[idx1 * 2];
            float y1 = d_points[idx1 * 2 + 1];

            // Calculate distance from pixel to line segment
            float length_squared = (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0);
            if (length_squared < 0.001f) continue; // Skip degenerate lines
            
            float dot_product = (x - x0) * (x1 - x0) + (y - y0) * (y1 - y0);
            float t = fmaxf(0.0f, fminf(1.0f, dot_product / length_squared));
            float proj_x = x0 + t * (x1 - x0);
            float proj_y = y0 + t * (y1 - y0);
            float distance_squared = (x - proj_x) * (x - proj_x) + (y - proj_y) * (y - proj_y);

            unsigned char r, g, b;
            get_class_color(label_id, r, g, b);

            // Draw line with thickness
            if (distance_squared < 16.0f) { // 4 pixel thickness
                *(src + ((y * width * 4) + (x * 4))) = r;
                *(src + ((y * width * 4) + (x * 4)) + 1) = g;
                *(src + ((y * width * 4) + (x * 4)) + 2) = b;
                *(src + ((y * width * 4) + (x * 4)) + 3) = 255;
            }
        }
    }
}

void gpu_draw_obb(unsigned char* src, int width, int height, float* d_points, 
                  int label_id, cudaStream_t stream) {
    dim3 threads_per_block(32, 32);
    dim3 num_blocks((width + threads_per_block.x - 1) / threads_per_block.x, 
                    (height + threads_per_block.y - 1) / threads_per_block.y);
    double current_time = (double)(std::chrono::system_clock::now().time_since_epoch()).count();
    gpu_draw_obb<<<num_blocks, threads_per_block, 0, stream>>>(
        src, width, height, d_points, label_id, current_time);
}
```

**File: `src/kernel.cuh`**

```cpp
void gpu_draw_obb(unsigned char* src, int width, int height, float* d_points, 
                  int label_id, cudaStream_t stream);
```

### Step 5: Add CSV Scaling Helper Function

Add a helper function to scale CSV coordinates for display resolution.

**File: `src/obb_detector.cpp`** (add this function)

```cpp
std::string scale_csv_for_display_resolution(const std::string& original_csv_path, int downsample) {
    if (downsample == 1) {
        return original_csv_path; // No scaling needed
    }
    
    // Create scaled CSV filename
    std::string scaled_csv_path = original_csv_path;
    size_t dot_pos = scaled_csv_path.find_last_of('.');
    if (dot_pos != std::string::npos) {
        scaled_csv_path = scaled_csv_path.substr(0, dot_pos) + 
                         "_scaled_" + std::to_string(downsample) + 
                         scaled_csv_path.substr(dot_pos);
    } else {
        scaled_csv_path += "_scaled_" + std::to_string(downsample);
    }
    
    // Check if scaled version already exists
    std::ifstream scaled_file(scaled_csv_path);
    if (scaled_file.good()) {
        scaled_file.close();
        return scaled_csv_path; // Use existing scaled version
    }
    
    // Create scaled CSV
    std::ifstream original_file(original_csv_path);
    std::ofstream scaled_file_out(scaled_csv_path);
    
    if (!original_file.is_open() || !scaled_file_out.is_open()) {
        printf("Warning: Could not create scaled CSV, using original\n");
        return original_csv_path;
    }
    
    std::string line;
    float scale_factor = 1.0f / downsample;
    
    while (std::getline(original_file, line)) {
        std::istringstream iss(line);
        std::string token;
        std::vector<std::string> tokens;
        
        // Split line by commas
        while (std::getline(iss, token, ',')) {
            tokens.push_back(token);
        }
        
        // Scale coordinate columns (indices 3-10: corner_x1,corner_y1,corner_x2,corner_y2,corner_x3,corner_y3,corner_x4,corner_y4)
        if (tokens.size() >= 11) {
            for (int i = 3; i <= 10; i++) {
                float coord = std::stof(tokens[i]);
                tokens[i] = std::to_string(coord * scale_factor);
            }
        }
        
        // Write scaled line
        for (size_t i = 0; i < tokens.size(); i++) {
            if (i > 0) scaled_file_out << ",";
            scaled_file_out << tokens[i];
        }
        scaled_file_out << "\n";
    }
    
    original_file.close();
    scaled_file_out.close();
    
    std::cout << "Created scaled CSV: " << scaled_csv_path 
              << " (scale factor: " << scale_factor << ")" << std::endl;
    return scaled_csv_path;
}
```

### Step 6: Add GUI Controls

Add GUI controls for OBB configuration.

**File: `src/gui.h`** (or wherever GUI is implemented)

Add OBB configuration section:

```cpp
// In the camera configuration GUI section
if (ImGui::CollapsingHeader("OBB Detection")) {
    ImGui::Checkbox("Enable OBB Detection", &camera_select->enable_obb);
    
    if (camera_select->enable_obb) {
        // CSV file path
        static char csv_path[256] = "";
        if (ImGui::InputText("CSV Prior File", csv_path, sizeof(csv_path))) {
            camera_select->obb_csv_path = std::string(csv_path);
        }
        
        // Background image path  
        static char bg_path[256] = "";
        if (ImGui::InputText("Background Image", bg_path, sizeof(bg_path))) {
            camera_select->obb_bg_path = std::string(bg_path);
        }
        
        // Detection threshold
        ImGui::SliderFloat("Motion Threshold", &camera_select->obb_threshold, 10.0f, 100.0f);
        
        // Background frames
        ImGui::SliderInt("Background Frames", &camera_select->obb_bg_frames, 5, 50);
    }
}
```

### Step 7: Update CMake Build

Ensure OBB detector files are included in the build.

**File: `CMakeLists.txt`**

The OBB detector files should already be included via the glob pattern `src/*.cpp`, but verify:

```cmake
# OBB detector should be automatically included via:
file(GLOB ORANGE_SOURCES "src/*.cpp")
```

### Step 8: Background Handling

OBB needs a background frame for motion detection. We support two modes:

- Static background (optional): if `obb_bg_path` is provided, we load it and scale to display resolution (already covered in Step 2)
- Auto background (recommended default): if no `obb_bg_path` is given, we build a background from the first N frames after the camera starts

Implementation sketch for auto background in `COpenGLDisplay::ThreadRunning()`:

```cpp
// Add at top of ThreadRunning() after allocations
const int bg_collect_frames = std::max(5, camera_select->obb_bg_frames);
int bg_collected = 0;
std::vector<cv::Mat> bg_samples; // store as RGB at display resolution
bool bg_built = false;

// Inside while(IsMachineOn()) loop, after debayer/resize produces either d_resize or debayer.d_debayer
if (camera_select->enable_obb && obb_detector && !bg_built) {
    // 1) Grab current frame at display resolution into CPU
    cv::Mat cpu_frame_rgba;
    if (camera_select->downsample != 1) {
        cpu_frame_rgba = cv::Mat(output_image_size.height, output_image_size.width, CV_8UC4);
        CHECK(cudaMemcpy2D(cpu_frame_rgba.data, output_image_size.width * 4,
                           d_resize, output_image_size.width * 4,
                           output_image_size.width * 4, output_image_size.height,
                           cudaMemcpyDeviceToHost));
    } else {
        cpu_frame_rgba = cv::Mat(camera_params->height, camera_params->width, CV_8UC4);
        CHECK(cudaMemcpy2D(cpu_frame_rgba.data, camera_params->width * 4,
                           debayer.d_debayer, camera_params->width * 4,
                           camera_params->width * 4, camera_params->height,
                           cudaMemcpyDeviceToHost));
    }
    cv::Mat cpu_frame_rgb;
    cv::cvtColor(cpu_frame_rgba, cpu_frame_rgb, cv::COLOR_BGRA2RGB);

    // 2) Accumulate a few frames
    bg_samples.push_back(cpu_frame_rgb.clone());
    bg_collected++;

    // 3) When enough frames collected, compute median background
    if (bg_collected >= bg_collect_frames) {
        cv::Mat median_bg(cpu_frame_rgb.size(), cpu_frame_rgb.type());

        // Compute per-channel median across samples
        std::vector<cv::Mat> channels(3);
        for (int c = 0; c < 3; ++c) {
            std::vector<cv::Mat> ch_samples;
            ch_samples.reserve(bg_samples.size());
            for (auto &s : bg_samples) {
                std::vector<cv::Mat> sch; cv::split(s, sch);
                ch_samples.push_back(sch[c]);
            }
            cv::Mat stack;
            cv::merge(ch_samples, stack); // shape: HxWxN interleaved
            // Fallback simple median: sort per-pixel via reshape to Nx1 and pick middle
            // For performance-critical builds, replace with a faster median implementation
            cv::sort(stack.reshape(1, stack.total()), stack, cv::SORT_ASCENDING);
            int mid = (int)ch_samples.size() / 2;
            cv::Mat median_vec = stack.row(mid);
            channels[c] = median_vec.reshape(1, cpu_frame_rgb.rows).clone();
        }
        cv::merge(channels, median_bg);

        // 4) Set background in detector
        obb_detector->set_background(median_bg);
        bg_built = true;
        bg_samples.clear();
    }
}
```

Faster background build (recommended in practice): running average using `cv::accumulateWeighted` (temporal mean). It's robust enough for static backgrounds and much faster than exact median:

```cpp
// Add at top of ThreadRunning()
const int bg_warmup_frames = std::max(5, camera_select->obb_bg_frames);
int bg_count = 0;
cv::Mat bg_f32; // running average in float32 RGB, display resolution
bool bg_ready = false;

// Inside while(IsMachineOn()), after you have cpu_frame_rgb (RGB at display resolution)
if (camera_select->enable_obb && obb_detector && !bg_ready) {
    cv::Mat f32;
    cpu_frame_rgb.convertTo(f32, CV_32F);
    if (bg_f32.empty()) {
        bg_f32 = f32.clone();
    } else {
        // alpha can be tuned; smaller alpha => smoother, slower convergence
        double alpha = 1.0 / std::min(bg_warmup_frames, 30);
        cv::accumulateWeighted(f32, bg_f32, alpha);
    }

    bg_count++;
    if (bg_count >= bg_warmup_frames) {
        cv::Mat bg_u8;
        bg_f32.convertTo(bg_u8, CV_8U);
        obb_detector->set_background(bg_u8);
        bg_ready = true;
    }
}
```

Notes:
- Running average converges quickly and is far cheaper than computing an exact per-pixel temporal median.
- If you need extra robustness to early moving objects, increase `bg_warmup_frames` (e.g., 20–50) or temporarily gate OBB until background is ready.
- You can keep collecting frames even after `bg_ready` and refresh the background periodically if lighting drifts.

Notes:
- This collects and builds background at the display resolution, matching the OBB processing scale
- You can adjust `bg_collect_frames` via `obb_bg_frames` in config
- For performance, replace the naive median with a more efficient per-pixel median if needed

## Usage Instructions

### 1. Prepare Data Files

- **CSV Prior File**: Contains hand-labeled OBB data for learning priors
  - Format: `frame,obb_id,class_id,corner_x1,corner_y1,corner_x2,corner_y2,corner_x3,corner_y3,corner_x4,corner_y4`
  - Example: `csv/Cam2005325_obb.csv`

- **Background Image**: Static background for motion detection
  - Should be the same resolution as camera frames
  - Example: `bg.jpg`

### 2. Configure Camera

1. Launch the Orange application
2. Select the camera you want to enable OBB detection for
3. In the GUI, expand "OBB Detection" section
4. Check "Enable OBB Detection"
5. Set the CSV file path (e.g., `csv/Cam2005325_obb.csv`)
6. Set the background image path (e.g., `bg.jpg`)
7. Adjust motion threshold (default: 30.0)
8. Set number of background frames (default: 10)

### 3. Run Detection

1. Start camera streaming
2. The system will automatically:
   - Load priors from CSV
   - Load background image
   - Process frames in real-time
   - Draw OBB overlays on the camera stream

## Performance Considerations

### Resolution Handling

- **OBB Detection**: Runs at display resolution (downsampled if needed)
- **CSV Scaling**: Priors are automatically scaled to match display resolution
- **Background Scaling**: Background images are scaled to display resolution
- **No Coordinate Conversion**: OBB coordinates match display coordinates directly

### CPU Processing

- Motion detection runs on CPU (OpenCV) at display resolution
- Background subtraction is the most expensive operation
- Consider processing every Nth frame to reduce overhead

### Threading

- OBB processing runs inline in the display thread
- No thread synchronization needed
- Direct GPU processing and drawing

## Troubleshooting

### Common Issues

1. **No OBBs Detected**
   - Check CSV file path and format
   - Verify background image loads correctly
   - Adjust motion threshold (try 20-50)
   - Check that scaled CSV was created correctly
   - Verify display resolution matches scaled priors

2. **Poor Detection Quality**
   - Verify background image matches current lighting
   - Check that objects in CSV match current objects
   - Adjust morphological operation parameters
   - Increase minimum contour area filter
   - Ensure CSV scaling is working correctly

3. **Overlay Not Appearing**
   - Check that OBB detection is producing results
   - Verify GPU drawing function is working
   - Ensure coordinates are at display resolution
   - Check that downsampling is handled correctly

4. **Performance Issues**
   - Reduce frame processing frequency
   - Use smaller background images
   - Disable OBB for non-critical cameras
   - Consider higher downsampling factors

5. **Build Errors**
   - Ensure all OBB detector files are in `src/`
   - Check OpenCV linking in CMakeLists.txt
   - Verify CUDA includes are available

### Debug Output

Enable debug output by setting:

```cpp
// In obb_detector.cpp, set debug flag
bool DEBUG_OBB = true;
```

This will print:
- Learned prior statistics
- Motion detection results
- Classification scores
- Rejection reasons

## Advanced Configuration

### Custom Class Colors

Modify `get_class_color()` in `kernel.cu` to add custom colors for OBB classes:

```cpp
__device__ void get_class_color(int class_id, unsigned char& r, unsigned char& g, unsigned char& b) {
    const unsigned char colors[10][3] = {
        {255, 0, 0},     // CylinderVertical - Red
        {0, 255, 0},     // Ball - Green  
        {0, 0, 255},     // CylinderSide - Blue
        // ... add more colors as needed
    };
    // ... rest of function
}
```

### Multi-Camera Support

Each camera can have independent OBB settings:

```cpp
// Each camera gets its own OBB detector instance
for (int cam_id = 0; cam_id < num_cameras; cam_id++) {
    if (camera_selects[cam_id]->enable_obb) {
        // Initialize OBB detector for this camera
        // Each camera can have different CSV files and parameters
    }
}
```

## Integration Checklist

- [ ] Add OBB fields to `CameraEachSelect` structure
- [ ] Extend `COpenGLDisplay` with OBB detector
- [ ] Add OBB processing to display thread
- [ ] Create GPU OBB drawing function
- [ ] Add GUI controls for OBB configuration
- [ ] Implement background image loading
- [ ] Test with sample CSV and background files
- [ ] Verify performance with real camera streams
- [ ] Add error handling and debug output
- [ ] Document any custom modifications

## Example Files

### Sample CSV Prior File
```csv
frame,obb_id,class_id,corner_x1,corner_y1,corner_x2,corner_y2,corner_x3,corner_y3,corner_x4,corner_y4
0,0,0,100,100,200,100,200,200,100,200
0,1,1,300,150,400,150,400,250,300,250
```

### Sample Background Image
- Resolution: Match camera resolution (e.g., 1920x1080)
- Format: JPG or PNG
- Content: Static background without moving objects

This integration provides a complete real-time OBB detection system that works alongside the existing YOLO detection, with GPU-accelerated overlay rendering and flexible configuration options.
