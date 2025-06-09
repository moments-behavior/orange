#!/bin/bash

echo "========================================"
echo "Starting build process..."
echo "========================================"

# Parse command line arguments
DEBUG=0
CLEAN=0
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
    --help)
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --debug    Build in debug mode"
    echo "  --clean    Remove all generated files and rebuild"
    echo "  --help     Show this help message"
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

echo "==> Using build directory: $BUILD_DIR"

# Handle clean option
if [ $CLEAN -eq 1 ]; then
  echo "========================================"
  echo "Cleaning build directory"
  echo "========================================"
  echo "==> Removing all build artifacts from $BUILD_DIR"
  rm -rf $BUILD_DIR
  rm -f targets/orange targets/orange_debug
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

echo "========================================"
echo "Compiling CUDA kernel"
echo "========================================"
# Compile CUDA kernel with appropriate flags
if [ $DEBUG -eq 1 ]; then
  if is_outdated "src/kernel.cu" "$BUILD_DIR/kernel.o"; then
    echo "==> Compiling CUDA kernel (debug mode)"
    nvcc -c src/kernel.cu -G -g -O0 -arch=sm_86 -o $BUILD_DIR/kernel.o
    echo "==> CUDA kernel compilation complete"
  else
    echo "==> CUDA kernel is up to date, skipping compilation"
  fi
else
  if is_outdated "src/kernel.cu" "$BUILD_DIR/kernel.o"; then
    echo "==> Compiling CUDA kernel (release mode)"
    nvcc -c src/kernel.cu -arch=sm_86 -o $BUILD_DIR/kernel.o
    echo "==> CUDA kernel compilation complete"
  else
    echo "==> CUDA kernel is up to date, skipping compilation"
  fi
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
    g++ -std=c++11 -I./third_party/imgui -I./third_party/imgui/backends $CFLAGS -Wall -Wformat `pkg-config --cflags glfw3` -c -o "$obj" "$src"
    echo "    - Compilation complete"
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
    g++ -std=c++17 -I$DIR_IMPLOT -I$DIR_IMGUI $CFLAGS -Wall -c -o "$obj" "$src"
    echo "    - Compilation complete"
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
      if [ "$src" -nt "$BUILD_DIR/orange" ]; then
        echo "    - Source file $src is newer than executable"
        need_link=1
        break
      fi
    done
  fi
fi

# Main compilation command with all the libraries
if [ $need_link -eq 1 ]; then
  echo "========================================"
  echo "Linking final executable"
  echo "========================================"
  echo "==> Starting link process"
  g++ $CFLAGS -std=c++17 $BUILD_DIR/*.o \
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
      `pkg-config --static --libs glfw3`
  echo "==> Linking complete"
else
  echo "==> Final executable is up to date, skipping linking"
fi

echo "========================================"
echo "Creating symbolic links"
echo "========================================"
# Create a symbolic link to the built binary
if [ $DEBUG -eq 1 ]; then
  echo "==> Creating symbolic link for debug binary"
  ln -sf $BUILD_DIR/orange targets/orange_debug
else
  echo "==> Creating symbolic link for release binary"
  ln -sf $BUILD_DIR/orange targets/orange
fi

echo "========================================"
if [ $CLEAN -eq 1 ]; then
  echo "Clean build completed successfully!"
else
  echo "Build completed successfully!"
fi
echo "Binary location: $BUILD_DIR/orange"
echo "========================================"
echo "Run with --help to see available options"