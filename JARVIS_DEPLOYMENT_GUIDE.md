# Jarvis 3D Pose Detection Integration - Deployment & Troubleshooting Guide

## Overview

This document provides a comprehensive guide for deploying and troubleshooting the Jarvis 3D pose detection integration in the Orange camera controller application. It covers the complete implementation, common deployment issues, and step-by-step solutions.

## Table of Contents

1. [Implementation Summary](#implementation-summary)
2. [File Structure Changes](#file-structure-changes)
3. [Build System Updates](#build-system-updates)
4. [Common Deployment Issues](#common-deployment-issues)
5. [Troubleshooting Steps](#troubleshooting-steps)
6. [Testing and Validation](#testing-and-validation)
7. [Performance Considerations](#performance-considerations)

## Implementation Summary

### What Was Added

The Jarvis 3D pose detection system was integrated into the Orange application with the following components:

1. **New Detection Mode**: `Detect3D_Pose` for 3D pose tracking
2. **Data Structures**: Jarvis-specific structures for 2D/3D centers and keypoints
3. **Core Classes**: `JarvisPoseDetector` and `JarvisFrameDetector`
4. **3D Processing**: `jarvis_3d_pose_proc` function for pose reconstruction
5. **GUI Integration**: Controls and visualization for 3D pose data
6. **Build System**: Updated compilation scripts

### Key Features

- **3D Center Detection**: Estimates 3D center point from multiple camera views
- **3D Keypoint Detection**: Detects 4 keypoints (Snout, EarL, EarR, Tail) in 3D space
- **Multi-camera Fusion**: Combines data from multiple cameras for accurate 3D reconstruction
- **Real-time Processing**: Integrated into the existing real-time pipeline
- **GUI Visualization**: Projects 3D data back to 2D for display

## File Structure Changes

### New Files Added

```
src/
├── jarvis_pose_det.h          # JarvisPoseDetector class header
├── jarvis_pose_det.cpp        # JarvisPoseDetector implementation
├── JarvisFrameDetector.h      # JarvisFrameDetector class header
└── JarvisFrameDetector.cpp    # JarvisFrameDetector implementation
```

### Modified Files

```
src/
├── video_capture.h            # Added Detect3D_Pose mode and jarvis_model_dir
├── video_capture.cpp          # Integrated JarvisFrameDetector
├── realtime_tool.h            # Added Jarvis data structures
├── detect3d.h                 # Added jarvis_3d_pose_proc declaration
├── detect3d.cpp               # Implemented jarvis_3d_pose_proc
├── gui.h                      # Updated streaming functions
└── orange.cpp                 # Added GUI controls and visualization

quick_build/
└── orange.sh                  # Updated build script
```

## Build System Updates

### Updated Build Script

The `quick_build/orange.sh` script was modified to include the new Jarvis source files:

```bash
# Added these files to the compilation command:
src/JarvisFrameDetector.cpp
src/jarvis_pose_det.cpp
```

### Compilation Flags

The build uses the following key flags:
- `-std=c++17`: C++17 standard
- `-Ofast -ffast-math`: Optimization flags
- TensorRT includes: `-I$HOME/nvidia/TensorRT/include`
- CUDA includes: `-I/usr/local/cuda/include`

## Common Deployment Issues

### 1. TensorRT API Compatibility Issues

**Problem**: `getTensorSize` method not found
**Error**: `error: 'class nvinfer1::ICudaEngine' has no member named 'getTensorSize'`

**Solution**: Replace `getTensorSize` with `getTensorShape` and calculate size manually:

```cpp
// OLD (deprecated):
size_t size = engine->getTensorSize(tensor_name) * 4;

// NEW (compatible):
auto shape = engine->getTensorShape(tensor_name);
size_t size = 1;
for (int j = 0; j < shape.nbDims; j++) {
    size *= shape.d[j];
}
size *= 4;
```

### 2. Missing Constants

**Problem**: Undefined constants like `JARVIS_CENTER_IMG_SIZE`
**Error**: `error: 'JARVIS_CENTER_IMG_SIZE' was not declared in this scope`

**Solution**: Ensure `realtime_tool.h` is included in `jarvis_pose_det.cpp`:

```cpp
#include "jarvis_pose_det.h"
#include "realtime_tool.h"  // Add this line
#include "common.hpp"
#include "utils.h"
```

### 3. Build Script Issues

**Problem**: Source files not being compiled
**Error**: Undefined references to JarvisPoseDetector methods

**Solution**: Update build script to compile source files individually:

```bash
# Add individual compilation steps:
g++ -Ofast -ffast-math -std=c++17 -c -o targets/jarvis_pose_det.o src/jarvis_pose_det.cpp [flags]
g++ -Ofast -ffast-math -std=c++17 -c -o targets/JarvisFrameDetector.o src/JarvisFrameDetector.cpp [flags]
```

### 4. Data Structure Definition Order

**Problem**: Types used before definition
**Error**: `error: 'JarvisCenter2d' was not declared in this scope`

**Solution**: Move Jarvis data structure definitions before their usage in `realtime_tool.h`:

```cpp
// Move these definitions BEFORE DetectionDataPerCam:
#define JARVIS_NUM_KEYPOINTS 4
#define JARVIS_NUM_CAMERAS 4
// ... other Jarvis structures
```

### 5. Function Signature Mismatches

**Problem**: Function calls with wrong number of arguments
**Error**: `error: too few arguments to function 'start_camera_streaming'`

**Solution**: Update all function calls to match new signatures:

```cpp
// Update calls to include jarvis_3d_thread parameter:
start_camera_streaming(cameras_select, num_cameras, detection3d_thread, jarvis_3d_thread);
stop_camera_streaming(cameras_select, num_cameras, detection3d_thread, jarvis_3d_thread);
```

## Troubleshooting Steps

### Step 1: Verify Build Environment

```bash
# Check if required libraries are installed:
ls -la /usr/local/cuda/include/
ls -la $HOME/nvidia/TensorRT/include/
ls -la /opt/EVT/eSDK/include/

# Verify CUDA version:
nvcc --version
```

### Step 2: Clean Build

```bash
# Clean previous build:
rm -rf targets/*.o
rm -f targets/orange

# Rebuild:
./quick_build/orange.sh
```

### Step 3: Check Compilation Errors

```bash
# Compile individual files to isolate errors:
g++ -Ofast -ffast-math -std=c++17 -c -o targets/jarvis_pose_det.o src/jarvis_pose_det.cpp [flags]
g++ -Ofast -ffast-math -std=c++17 -c -o targets/JarvisFrameDetector.o src/JarvisFrameDetector.cpp [flags]
```

### Step 4: Verify Object Files

```bash
# Check if object files were created:
ls -la targets/*.o | grep -E "(jarvis|Jarvis)"
```

### Step 5: Check Linking

```bash
# Verify all symbols are resolved:
nm targets/jarvis_pose_det.o | grep -E "(JarvisPoseDetector|detect_centers)"
```

## Testing and Validation

### 1. Basic Compilation Test

```bash
# Run build script and check exit code:
./quick_build/orange.sh
echo $?  # Should be 0 for success
```

### 2. Executable Creation Test

```bash
# Verify executable was created:
ls -la targets/orange
file targets/orange
```

### 3. Runtime Test

```bash
# Run application (if no GUI):
./targets/orange --help

# Check for runtime errors:
./targets/orange 2>&1 | grep -i error
```

### 4. GUI Integration Test

1. Launch application with GUI
2. Navigate to camera properties
3. Verify "Jarvis Model Dir" input field appears
4. Check that "3D Pose" detection mode is available
5. Verify no crashes when switching modes

## Performance Considerations

### Memory Usage

- **TensorRT Engines**: Each engine loads into GPU memory
- **Buffer Allocation**: Pre-allocated buffers for inference
- **Multi-camera Processing**: Memory scales with number of cameras

### GPU Requirements

- **CUDA Compute Capability**: 6.0+ recommended
- **GPU Memory**: 4GB+ recommended for multiple models
- **TensorRT Version**: 8.0+ for compatibility

### Optimization Tips

1. **Model Quantization**: Use FP16 or INT8 for faster inference
2. **Batch Processing**: Process multiple frames simultaneously
3. **Memory Pooling**: Reuse GPU memory buffers
4. **Async Processing**: Use CUDA streams for parallel execution

## Model Conversion (Future)

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- TensorRT 8.0+
- CUDA 11.6+

### Conversion Script

Use the provided `convert_jarvis_to_tensorrt.py` script:

```bash
python convert_jarvis_to_tensorrt.py \
    --center_model jarvis/trained_models/CenterDetect/model.pth \
    --keypoint_model jarvis/trained_models/KeypointDetect/model.pth \
    --hybrid_model jarvis/trained_models/HybridNet/model.pth \
    --output_dir models/tensorrt/
```

## Configuration

### Camera Settings

```cpp
// In camera properties GUI:
cameras_select[i].detect_mode = Detect3D_Pose;
cameras_select[i].jarvis_model_dir = "/path/to/jarvis/models";
```

### Model Paths

The application expects the following model structure:
```
jarvis_model_dir/
├── center_detect.engine
├── keypoint_detect.engine
├── hybrid_net.engine
└── model_info.json
```

## Debugging Tips

### 1. Enable Verbose Logging

```cpp
// Add debug prints in jarvis_pose_det.cpp:
printf("Loading center detection model...\n");
printf("Model loaded successfully\n");
```

### 2. Check GPU Memory

```bash
# Monitor GPU memory usage:
nvidia-smi -l 1
```

### 3. Validate Input Data

```cpp
// Check input frame dimensions:
printf("Input frame: %dx%d\n", frame.width, frame.height);
```

### 4. Test Individual Components

```cpp
// Test each stage separately:
// 1. Center detection only
// 2. Keypoint detection only  
// 3. 3D reconstruction only
```

## Common Error Messages and Solutions

### "TensorRT engine not found"
- **Cause**: Model files missing or wrong path
- **Solution**: Verify model directory and file names

### "CUDA out of memory"
- **Cause**: Insufficient GPU memory
- **Solution**: Reduce batch size or use smaller models

### "Invalid tensor dimensions"
- **Cause**: Input image size mismatch
- **Solution**: Check image preprocessing and model input requirements

### "Engine execution failed"
- **Cause**: TensorRT engine corruption or incompatibility
- **Solution**: Rebuild engines with correct TensorRT version

## Maintenance

### Regular Updates

1. **TensorRT Version**: Keep TensorRT updated for compatibility
2. **CUDA Version**: Ensure CUDA compatibility
3. **Model Updates**: Rebuild engines when models change
4. **Dependencies**: Update third-party libraries

### Monitoring

1. **Performance Metrics**: Track inference times
2. **Memory Usage**: Monitor GPU memory consumption
3. **Error Rates**: Log and analyze detection failures
4. **System Resources**: Monitor CPU/GPU utilization

## Support

### Documentation References

- [3D_POSE_INTEGRATION_PLAN.md](3D_POSE_INTEGRATION_PLAN.md)
- [JARVIS_ANALYSIS.md](JARVIS_ANALYSIS.md)
- [JARVIS_3D_PIPELINE_DETAILED.md](JARVIS_3D_PIPELINE_DETAILED.md)
- [ORANGE_PROJECT_WIKI.md](ORANGE_PROJECT_WIKI.md)

### Key Files for Reference

- `src/jarvis_pose_det.h` - Core class interface
- `src/jarvis_pose_det.cpp` - Implementation details
- `src/realtime_tool.h` - Data structures
- `quick_build/orange.sh` - Build configuration

---

**Last Updated**: [Current Date]
**Version**: 1.0
**Status**: Implementation Complete, Ready for Deployment
