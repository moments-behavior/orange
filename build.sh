#!/bin/bash

echo "========================================"
echo "Starting build process..."
echo "========================================"

# Parse command line arguments
DEBUG=0
CLEAN=0
CUDA_DEBUG=0
NVTX_PROFILE=0
for arg in "$@"; do
  case $arg in
    --debug)
    DEBUG=1
    shift
    ;;
    --clean)
    CLEAN=1
    shift
    ;;
    --cuda-debug)
    CUDA_DEBUG=1
    shift
    ;;
    --nvtx)
    NVTX_PROFILE=1
    shift
    ;;
    --help)
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --debug       Build in debug mode"
    echo "  --clean       Remove all generated files and rebuild"
    echo "  --cuda-debug  Enable CUDA context debugging (can be combined with --debug)"
    echo "  --nvtx        Enable NVTX profiling markers (recommended for Nsight)"
    echo "  --help        Show this help message"
    echo ""
    echo "Profiling Examples:"
    echo "  # Build with profiling support:"
    echo "  $0 --nvtx"
    echo ""
    echo "  # Build debug version with profiling:"
    echo "  $0 --debug --nvtx"
    echo ""
    echo "  # Profile the application:"
    echo "  nsys profile --duration=30 --output=orange_profile targets/orange"
    exit 0
    ;;
  esac
done

# Setup compiler flags based on build type
if [ $DEBUG -eq 1 ]; then
  echo "==> Building in DEBUG mode"
  CFLAGS="-g -O0 -DDEBUG"
  BUILD_DIR="targets/debug"
else
  echo "==> Building in RELEASE mode"
  CFLAGS="-Ofast -ffast-math"
  BUILD_DIR="targets/release"
fi

# Add CUDA debug flag if requested
if [ $CUDA_DEBUG -eq 1 ]; then
  echo "==> Enabling CUDA context debugging"
  CFLAGS="$CFLAGS -DDEBUG_CUDA_CONTEXT -DENABLE_CUDA_DEBUG_LOGGING"
  if [ $DEBUG -eq 1 ]; then
    BUILD_DIR="targets/debug_cuda"
  else
    BUILD_DIR="targets/release_cuda"
  fi
fi

# Add NVTX profiling support
if [ $NVTX_PROFILE -eq 1 ]; then
  echo "==> Enabling NVTX profiling markers"
  CFLAGS="$CFLAGS -DENABLE_NVTX_PROFILING"
  # Append nvtx to build directory name
  if [ $CUDA_DEBUG -eq 1 ] && [ $DEBUG -eq 1 ]; then
    BUILD_DIR="targets/debug_cuda_nvtx"
  elif [ $CUDA_DEBUG -eq 1 ]; then
    BUILD_DIR="targets/release_cuda_nvtx"
  elif [ $DEBUG -eq 1 ]; then
    BUILD_DIR="targets/debug_nvtx"
  else
    BUILD_DIR="targets/release_nvtx"
  fi
fi

echo "==> Using build directory: $BUILD_DIR"
echo "==> Compiler flags: $CFLAGS"

# Handle clean option
if [ $CLEAN -eq 1 ]; then
  echo "========================================"
  echo "Cleaning build directory"
  echo "========================================"
  echo "==> Removing all build artifacts from $BUILD_DIR"
  rm -rf $BUILD_DIR
  rm -f targets/orange targets/orange_debug targets/orange_cuda_debug targets/orange_nvtx
  echo "==> Clean completed"
fi

mkdir -p $BUILD_DIR
echo "==> Created build directory if it didn't exist"

# Installation paths
echo "==> Setting up dependency paths"
ORANGE_ROOT="/opt/orange"
FFMPEG_ROOT="${ORANGE_ROOT}/lib/ffmpeg-nvidia"
OPENCV_ROOT="${ORANGE_ROOT}/lib/opencv"
TENSORRT_ROOT="/usr/local/TensorRT-10.0.1.6"
CUDA_ROOT="/usr/local/cuda"
EVT_ROOT="/opt/EVT/eSDK"

DIR_IMGUI="third_party/imgui"
DIR_IMGUI_BACKEND="third_party/imgui/backends"
DIR_IMPLOT="third_party/implot"
DIR_FILEBROWSER="third_party/ImGuiFileDialog"
DIR_ICONFONT="third_party/IconFontCppHeaders"

# Check for NVTX availability
if [ $NVTX_PROFILE -eq 1 ]; then
  echo "========================================"
  echo "Checking NVTX availability"
  echo "========================================"
  
  # Check if NVTX headers are available (updated paths for newer CUDA)
  NVTX_FOUND=0
  for nvtx_path in "/usr/local/cuda/include/nvtx3/nvToolsExt.h" \
                   "/usr/include/nvtx3/nvToolsExt.h" \
                   "/usr/local/cuda/include/nvToolsExt.h" \
                   "/opt/nvidia/nsight-systems/*/target-linux-x64/nvtx/include/nvtx3/nvToolsExt.h"; do
    if [ -f "$nvtx_path" ]; then
      echo "==> NVTX headers found at: $nvtx_path"
      NVTX_FOUND=1
      break
    fi
  done
  
  if [ $NVTX_FOUND -eq 0 ]; then
    echo "WARNING: NVTX headers not found!"
    echo "Install with: sudo apt install nvidia-nsight-systems-cli"
    echo "Or use CUDA toolkit installation"
    echo "Continuing build anyway..."
  fi
fi

# Helper function to check if a file is outdated (source is newer than object)
is_outdated() {
  local src="$1"
  local obj="$2"
  
  if [ ! -f "$obj" ]; then
    # Object file doesn't exist
    return 0
  fi
  
  if [ "$src" -nt "$obj" ]; then
    # Source is newer than object
    return 0
  fi
  
  # Object is up to date
  return 1
}

# Check if any header files have changed
headers_changed() {
  local executable="$1"
  
  # Check if any header file is newer than the executable
  for header in src/*.h src/*.hpp; do
    if [ -f "$header" ] && [ "$header" -nt "$executable" ]; then
      echo "    - Header file $header is newer than executable"
      return 0
    fi
  done
  
  # Check for the new cuda_context_debug.h header specifically
  if [ -f "src/cuda_context_debug.h" ] && [ "src/cuda_context_debug.h" -nt "$executable" ]; then
    echo "    - CUDA context debug header is newer than executable"
    return 0
  fi
  
  # Check for NVTX profiling header
  if [ -f "src/nvtx_profiling.h" ] && [ "src/nvtx_profiling.h" -nt "$executable" ]; then
    echo "    - NVTX profiling header is newer than executable"
    return 0
  fi
  
  # Check for optimized YOLO preprocessing header
  if [ -f "src/optimized_yolo_preprocess.h" ] && [ "src/optimized_yolo_preprocess.h" -nt "$executable" ]; then
    echo "    - Optimized YOLO preprocessing header is newer than executable"
    return 0
  fi
  
  return 1
}

echo "========================================"
echo "Checking for required headers"
echo "========================================"
# Check if required headers exist
MISSING_HEADERS=0

if [ ! -f "src/cuda_context_debug.h" ]; then
  echo "WARNING: cuda_context_debug.h not found in src/"
  echo "         This file is required for enhanced CUDA debugging"
  MISSING_HEADERS=1
fi

if [ ! -f "src/nvtx_profiling.h" ]; then
  echo "WARNING: nvtx_profiling.h not found in src/"
  echo "         This file is required for NVTX profiling support"
  MISSING_HEADERS=1
fi

if [ ! -f "src/optimized_yolo_preprocess.h" ]; then
  echo "WARNING: optimized_yolo_preprocess.h not found in src/"
  echo "         This file is required for optimized YOLO preprocessing"
  MISSING_HEADERS=1
fi

if [ $MISSING_HEADERS -eq 1 ]; then
  echo "         Please ensure all required header files are present"
  echo "         Build will continue but may fail..."
fi

echo "========================================"
echo "Compiling CUDA kernels"
echo "========================================"
# Compile CUDA kernels with appropriate flags
NVCC_FLAGS="-arch=sm_86"

NVCC_FLAGS="$NVCC_FLAGS -I$TENSORRT_ROOT/include"

if [ $DEBUG -eq 1 ]; then
  NVCC_FLAGS="$NVCC_FLAGS -G -g -O0"
else
  NVCC_FLAGS="$NVCC_FLAGS -O3"
fi

# Add NVTX support to CUDA compilation
if [ $NVTX_PROFILE -eq 1 ]; then
  NVCC_FLAGS="$NVCC_FLAGS -DENABLE_NVTX_PROFILING"
fi

# Add CUDA debug support to CUDA compilation
if [ $CUDA_DEBUG -eq 1 ]; then
  NVCC_FLAGS="$NVCC_FLAGS -DDEBUG_CUDA_CONTEXT -DENABLE_CUDA_DEBUG_LOGGING"
fi

# Helper function to compile CUDA files
compile_cuda_file() {
  local src="$1"
  local obj="$2"
  local desc="$3"
  
  if is_outdated "$src" "$obj"; then
    echo "==> Compiling $desc with flags: $NVCC_FLAGS"
    nvcc -c "$src" $NVCC_FLAGS $CFLAGS -o "$obj"
    if [ $? -eq 0 ]; then
      echo "==> $desc compilation complete"
    else
      echo "==> ERROR: $desc compilation failed!"
      exit 1
    fi
  else
    echo "==> $desc is up to date, skipping compilation"
  fi
}

# Compile original kernel
compile_cuda_file "src/kernel.cu" "$BUILD_DIR/kernel.o" "CUDA kernel"

# Compile optimized YOLO preprocessing kernel
if [ -f "src/optimized_yolo_preprocess.cu" ]; then
  compile_cuda_file "src/optimized_yolo_preprocess.cu" "$BUILD_DIR/optimized_yolo_preprocess.o" "Optimized YOLO preprocessing kernel"
else
  echo "WARNING: optimized_yolo_preprocess.cu not found in src/"
  echo "         Optimized YOLO preprocessing will not be available"
  echo "         Build will continue with original preprocessing..."
fi

echo "========================================"
echo "Checking ImGui files"
echo "========================================"
# Using individual checks rather than associative arrays for better compatibility
check_and_compile_imgui() {
  local src="$1"
  local obj="$2"
  
  echo "==> Checking $src"
  if is_outdated "$src" "$obj"; then
    echo "    - File needs compilation"
    g++ -std=c++11 -I./third_party/imgui -I./third_party/imgui/backends -I/usr/local/cuda/include $CFLAGS -Wall -Wformat `pkg-config --cflags glfw3` -c -o "$obj" "$src"
    if [ $? -eq 0 ]; then
      echo "    - Compilation complete"
    else
      echo "    - ERROR: Compilation failed for $src"
      exit 1
    fi
  else
    echo "    - File is up to date, skipping"
  fi
}

# Compile ImGui files if needed
check_and_compile_imgui "$DIR_IMGUI/imgui.cpp" "$BUILD_DIR/imgui.o"
check_and_compile_imgui "$DIR_IMGUI/imgui_demo.cpp" "$BUILD_DIR/imgui_demo.o"
check_and_compile_imgui "$DIR_IMGUI/imgui_draw.cpp" "$BUILD_DIR/imgui_draw.o"
check_and_compile_imgui "$DIR_IMGUI/imgui_tables.cpp" "$BUILD_DIR/imgui_tables.o"
check_and_compile_imgui "$DIR_IMGUI/imgui_widgets.cpp" "$BUILD_DIR/imgui_widgets.o"
check_and_compile_imgui "$DIR_IMGUI_BACKEND/imgui_impl_glfw.cpp" "$BUILD_DIR/imgui_impl_glfw.o"
check_and_compile_imgui "$DIR_IMGUI_BACKEND/imgui_impl_opengl3.cpp" "$BUILD_DIR/imgui_impl_opengl3.o"

echo "========================================"
echo "Checking ImPlot files"
echo "========================================"
# Compile ImPlot files if needed
check_and_compile_implot() {
  local src="$1"
  local obj="$2"
  
  echo "==> Checking $src"
  if is_outdated "$src" "$obj"; then
    echo "    - File needs compilation"
    g++ -std=c++17 -I$DIR_IMPLOT -I$DIR_IMGUI -I/usr/local/cuda/include $CFLAGS -Wall -c -o "$obj" "$src"
    if [ $? -eq 0 ]; then
      echo "    - Compilation complete"
    else
      echo "    - ERROR: Compilation failed for $src"
      exit 1
    fi
  else
    echo "    - File is up to date, skipping"
  fi
}

check_and_compile_implot "$DIR_IMPLOT/implot.cpp" "$BUILD_DIR/implot.o"
check_and_compile_implot "$DIR_IMPLOT/implot_items.cpp" "$BUILD_DIR/implot_items.o"
check_and_compile_implot "$DIR_IMPLOT/implot_demo.cpp" "$BUILD_DIR/implot_demo.o"

echo "========================================"
echo "Checking if linking is needed"
echo "========================================"
# Check if any object files or source files have changed
need_link=0
if [ ! -f "$BUILD_DIR/orange" ]; then
  echo "==> Final executable doesn't exist, linking required"
  need_link=1
else
  # Check if any object file is newer than the executable
  echo "==> Checking if object files changed"
  for obj in $BUILD_DIR/*.o; do
    if [ "$obj" -nt "$BUILD_DIR/orange" ]; then
      echo "    - Object file $obj is newer than executable"
      need_link=1
      break
    fi
  done
  
  # Check if any source file is newer than the executable
  if [ $need_link -eq 0 ]; then
    echo "==> Checking if source files changed"
    for src in src/*.cpp src/NvEncoder/*.cpp src/*.cu; do # Added src/*.cu to check kernel.cu
      if [ -f "$src" ] && [ "$src" -nt "$BUILD_DIR/orange" ]; then
        echo "    - Source file $src is newer than executable"
        need_link=1
        break
      fi
    done
  fi
  
  # Check if any header file is newer than the executable
  if [ $need_link -eq 0 ]; then
    echo "==> Checking if header files changed"
    if headers_changed "$BUILD_DIR/orange"; then
      need_link=1
    fi
  fi
fi

# Build library flags
LIBS=""
INCLUDES=""

# Add NVTX libraries if profiling is enabled
if [ $NVTX_PROFILE -eq 1 ]; then
  echo "==> Adding NVTX libraries"
  LIBS="$LIBS -L/usr/local/cuda/lib64 -lnvToolsExt -Wl,-rpath,/usr/local/cuda/lib64"
fi

# Main compilation command with all the libraries
if [ $need_link -eq 1 ]; then
  echo "========================================"
  echo "Linking final executable"
  echo "========================================"
  echo "==> Starting link process"
  
  # Show what debugging options are enabled
  if [ $CUDA_DEBUG -eq 1 ]; then
    echo "==> CUDA context debugging ENABLED"
  fi
  
  if [ $NVTX_PROFILE -eq 1 ]; then
    echo "==> NVTX profiling ENABLED"
  fi
  
  # Check if optimized preprocessing is available
  if [ -f "$BUILD_DIR/optimized_yolo_preprocess.o" ]; then
    echo "==> Optimized YOLO preprocessing ENABLED"
  else
    echo "==> Optimized YOLO preprocessing NOT AVAILABLE (using fallback)"
  fi
  
  g++ $CFLAGS -std=c++17 -Wno-deprecated-declarations $BUILD_DIR/*.o \
      -o $BUILD_DIR/orange \
      -I./src/ \
      src/orange.cpp \
      src/network_base.cpp \
      src/FFmpegWriter.cpp \
      src/camera.cpp \
      src/video_capture.cpp \
      src/acquire_frames.cpp \
      src/offthreadmachine.cpp \
      src/opengldisplay.cpp \
      src/gpu_video_encoder.cpp \
      src/yolov8_det.cpp \
      src/yolo_worker.cpp \
      src/enet_thread.cpp \
      src/project.cpp \
      src/global.cpp \
      src/image_writer_worker.cpp \
      src/crop_and_encode_worker.cpp \
      $DIR_FILEBROWSER/ImGuiFileDialog.cpp \
      -I$DIR_IMGUI \
      -I$DIR_IMGUI_BACKEND \
      -I$DIR_IMPLOT \
      -I$DIR_FILEBROWSER \
      -I$DIR_ICONFONT \
      -I./src/NvEncoder/ ./src/NvEncoder/*.cpp \
      -I./nvenc_api/include \
      -I${EVT_ROOT}/include \
      -I/usr/local/cuda/include \
      -I$FFMPEG_ROOT/include \
      -I./third_party/flatbuffers/include \
      -I$TENSORRT_ROOT/include \
      $INCLUDES \
      -L${EVT_ROOT}/lib \
      -lEmergentCamera -lEmergentGenICam -lEmergentGigEVision \
      -lm \
      -lpthread \
      -I./third_party/flatbuffers/include \
      -lenet -I/usr/local/include/ \
      -L/usr/local/cuda/lib64/ -lcudart -lcuda -lnppicc -lnppidei -lnvidia-encode -lnppc -lnppig -lnppial \
      -lGLEW -lGL \
      -L$FFMPEG_ROOT/lib -lavformat -lswscale -lswresample -lavutil -lavcodec \
      -I/usr/local/include/opencv4 \
      -L$OPENCV_ROOT/lib -lopencv_sfm -lopencv_core -lopencv_imgcodecs -lopencv_imgproc \
      -I$TENSORRT_ROOT/include \
      -L$TENSORRT_ROOT/lib -lnvinfer -lnvinfer_plugin \
      $LIBS \
      `pkg-config --static --libs glfw3`
  
  if [ $? -eq 0 ]; then
    echo "==> Linking complete"
  else
    echo "==> ERROR: Linking failed!"
    echo "    Check library paths and dependencies"
    if [ $NVTX_PROFILE -eq 1 ]; then
      echo "    NVTX-related linking errors? Try:"
      echo "    - sudo apt install nvidia-nsight-systems-cli"
      echo "    - Check CUDA installation includes NVTX libraries"
    fi
    exit 1
  fi
else
  echo "==> Final executable is up to date, skipping linking"
fi

echo "========================================"
echo "Creating symbolic links"
echo "========================================"
if [ $NVTX_PROFILE -eq 1 ] && [ $DEBUG -eq 1 ] && [ $CUDA_DEBUG -eq 1 ]; then
  echo "==> Creating symbolic link for debug+CUDA debug+NVTX binary"
  ln -sf debug_cuda_nvtx/orange targets/orange_nvtx
elif [ $NVTX_PROFILE -eq 1 ] && [ $DEBUG -eq 1 ]; then
  echo "==> Creating symbolic link for debug+NVTX binary"
  ln -sf debug_nvtx/orange targets/orange_nvtx
elif [ $NVTX_PROFILE -eq 1 ] && [ $CUDA_DEBUG -eq 1 ]; then
  echo "==> Creating symbolic link for CUDA debug+NVTX binary"
  ln -sf release_cuda_nvtx/orange targets/orange_nvtx
elif [ $NVTX_PROFILE -eq 1 ]; then
  echo "==> Creating symbolic link for NVTX binary"
  ln -sf release_nvtx/orange targets/orange_nvtx
elif [ $DEBUG -eq 1 ] && [ $CUDA_DEBUG -eq 1 ]; then
  echo "==> Creating symbolic link for debug+CUDA debug binary"
  ln -sf debug_cuda/orange targets/orange_cuda_debug
elif [ $DEBUG -eq 1 ]; then
  echo "==> Creating symbolic link for debug binary"
  ln -sf debug/orange targets/orange_debug
elif [ $CUDA_DEBUG -eq 1 ]; then
  echo "==> Creating symbolic link for CUDA debug binary"
  ln -sf release_cuda/orange targets/orange_cuda_debug
else
  echo "==> Creating symbolic link for release binary"
  ln -sf release/orange targets/orange
fi

echo "========================================"
if [ $CLEAN -eq 1 ]; then
  echo "Clean build completed successfully!"
else
  echo "Build completed successfully!"
fi
echo "Binary location: $BUILD_DIR/orange"

if [ $CUDA_DEBUG -eq 1 ]; then
  echo ""
  echo "CUDA DEBUGGING ENABLED:"
  echo "- Enhanced context logging will be displayed"
  echo "- Resource allocation/deallocation tracking"
  echo "- TensorRT operation monitoring"
  echo "- Run with 2>&1 | tee debug.log to capture all output"
fi

if [ $NVTX_PROFILE -eq 1 ]; then
  echo ""
  echo "NVTX PROFILING ENABLED:"
  echo "- Timeline markers added to critical sections"
  echo "- Ready for Nsight Systems profiling"
  echo ""
  echo "Quick profiling commands:"
  if [ -f "targets/orange_nvtx" ]; then
    echo "  # Basic 30-second capture:"
    echo "  nsys profile --duration=30 --output=orange_profile targets/orange_nvtx"
    echo ""
    echo "  # Focused CUDA profiling:"
    echo "  nsys profile --trace=cuda,nvtx --duration=30 --output=orange_cuda targets/orange_nvtx"
    echo ""
    echo "  # Memory transfer analysis:"
    echo "  nsys profile --trace=cuda,nvtx,cublas,cudnn --duration=30 --output=orange_full targets/orange_nvtx"
  else
    echo "  # Basic 30-second capture:"
    echo "  nsys profile --duration=30 --output=orange_profile $BUILD_DIR/orange"
  fi
  echo ""
  echo "  # View results:"
  echo "  nsight-sys orange_profile.nsys-rep"
fi

# Show optimization status
if [ -f "$BUILD_DIR/optimized_yolo_preprocess.o" ]; then
  echo ""
  echo "OPTIMIZED YOLO PREPROCESSING ENABLED:"
  echo "- Single kernel replaces multi-step preprocessing"
  echo "- ~85% reduction in memory bandwidth usage"
  echo "- Should significantly improve YOLO performance on A16 GPUs"
fi

echo "========================================"
echo "Run with --help to see available options"