#!/bin/bash

# ANSI color codes
RED="\033[0;31m"
GREEN="\033[0;32m"
BLUE="\033[0;34m"
NC="\033[0m"

# Error handling
error_exit() {
    echo -e "${RED}Error: $1${NC}" >&2
    exit 1
}

# Installation paths
ORANGE_ROOT="/opt/orange"
FFMPEG_ROOT="${ORANGE_ROOT}/lib/ffmpeg-nvidia"
OPENCV_ROOT="${ORANGE_ROOT}/lib/opencv"
TENSORRT_ROOT="/usr/local/TensorRT-10.0.1.6"
CUDA_ROOT="/usr/local/cuda"
EVT_ROOT="/opt/EVT/eSDK"

# Third-party directories
IMGUI_DIR="third_party/imgui"
IMPLOT_DIR="third_party/implot"
FILEBROWSER_DIR="third_party/ImGuiFileDialog"

# Create build directory
mkdir -p targets || error_exit "Failed to create targets directory"
rm -f targets/orange

# Debug flags
DEBUG_FLAGS="-g3 -O0 -DDEBUG -fno-omit-frame-pointer -fno-inline"
# SANITIZER_FLAGS="-fsanitize=address -fsanitize=undefined"
THREAD_SANITIZER_FLAGS="-fsanitize=thread -pie -fPIE"
WARNING_FLAGS="-Wall -Wextra -Wpedantic -Wformat=2 -Wno-unused-parameter"

# Common compiler flags with debug support
COMMON_INCLUDES="\
-I/usr/local/include \
-I/usr/local/include/ffnvcodec \
-I$FFMPEG_ROOT/include \
-I$EVT_ROOT/include \
-I$CUDA_ROOT/include \
-I$TENSORRT_ROOT/include \
-I./include \
-I./include/gui \
-I./src \
-I./src/NvEncoder \
-I./src/gui \
-I$IMGUI_DIR \
-I$IMGUI_DIR/backends \
-I$IMPLOT_DIR \
-I$FILEBROWSER_DIR \
-I./third_party/IconFontCppHeaders \
-I./third_party/flatbuffers/include \
$(pkg-config --cflags opencv4 glfw3)"

# Library paths and flags
LIBS="\
-Wl,-rpath,$EVT_ROOT/lib \
-Wl,-rpath,$CUDA_ROOT/lib64 \
-Wl,-rpath,$FFMPEG_ROOT/lib \
-Wl,-rpath,$TENSORRT_ROOT/lib \
-L$EVT_ROOT/lib -lEmergentCamera -lEmergentGenICam -lEmergentGigEVision \
-L$CUDA_ROOT/lib64 -lcudart -lcuda -lnppicc -lnppidei -lnvidia-encode -lnppc -lnppig -lnppial \
-L$TENSORRT_ROOT/lib -lnvinfer -lnvinfer_plugin \
-L$FFMPEG_ROOT/lib -lavformat -lavcodec -lavutil -lswscale -lswresample \
$(pkg-config --libs opencv4) \
-lGLEW -lGL -lglfw -lenet -lpthread -lm"

# Build type selection
BUILD_TYPE=${1:-debug}
if [ "$BUILD_TYPE" = "release" ]; then
    echo -e "${BLUE}Building in Release mode...${NC}"
    COMPILER_FLAGS="-O3"
else
    echo -e "${BLUE}Building in Debug mode...${NC}"
    COMPILER_FLAGS="$DEBUG_FLAGS $WARNING_FLAGS $SANITIZER_FLAGS"
fi

# Build steps
echo -e "${BLUE}Building Orange...${NC}"

# 1. CUDA kernel with debug info
echo -e "${BLUE}Compiling CUDA kernel...${NC}"
nvcc -G -g -O0 -c src/kernel.cu -arch=sm_86 -o targets/kernel.o \
    -I./include -I./src || error_exit "CUDA kernel compilation failed"

# 2. ImGui and dependencies
echo -e "${BLUE}Compiling ImGui and dependencies...${NC}"
for src in \
    "$IMGUI_DIR/imgui.cpp" \
    "$IMGUI_DIR/imgui_demo.cpp" \
    "$IMGUI_DIR/imgui_draw.cpp" \
    "$IMGUI_DIR/imgui_tables.cpp" \
    "$IMGUI_DIR/imgui_widgets.cpp" \
    "$IMGUI_DIR/backends/imgui_impl_glfw.cpp" \
    "$IMGUI_DIR/backends/imgui_impl_opengl3.cpp" \
    "$IMPLOT_DIR/implot.cpp" \
    "$IMPLOT_DIR/implot_items.cpp"; do
    echo -e "${BLUE}Compiling $(basename $src)...${NC}"
    g++ -std=c++17 $COMPILER_FLAGS -fPIC -c "$src" -o "targets/$(basename ${src%.cpp}).o" $COMMON_INCLUDES || \
        error_exit "Failed to compile $src"
done

# 3. Compile core camera and streaming components
echo -e "${BLUE}Compiling core components...${NC}"
CORE_SOURCES="\
    src/emergent_camera.cpp \
    src/frame_streaming.cpp \
    src/gpu_streaming.cpp \
    src/gpu_manager.cpp \
    src/camera_manager.cpp"

for src in $CORE_SOURCES; do
    echo -e "${BLUE}Compiling $(basename $src)...${NC}"
    g++ -std=c++17 $COMPILER_FLAGS -fPIC -c "$src" -o "targets/$(basename ${src%.cpp}).o" $COMMON_INCLUDES || \
        error_exit "Failed to compile $src"
done

# 4. Compile GUI components
echo -e "${BLUE}Compiling GUI components...${NC}"
GUI_SOURCES="\
    src/gui/camera_control_panel.cpp \
    src/gui/main_window.cpp"

for src in $GUI_SOURCES; do
    echo -e "${BLUE}Compiling $(basename $src)...${NC}"
    g++ -std=c++17 $COMPILER_FLAGS -fPIC -c "$src" -o "targets/$(basename ${src%.cpp}).o" $COMMON_INCLUDES || \
        error_exit "Failed to compile $src"
done

# 5. Direct compilation and linking with debug symbols
echo -e "${GREEN}Compiling and linking project...${NC}"
g++ -std=c++17 $COMPILER_FLAGS \
    targets/*.o \
    src/network_base.cpp \
    src/FFmpegWriter.cpp \
    src/video_capture.cpp \
    src/acquire_frames.cpp \
    src/offthreadmachine.cpp \
    src/opengldisplay.cpp \
    src/threadworker.cpp \
    src/gpu_video_encoder.cpp \
    src/yolov8_det.cpp \
    src/fs_utils.cpp \
    $FILEBROWSER_DIR/ImGuiFileDialog.cpp \
    src/NvEncoder/*.cpp \
    src/orange.cpp \
    $COMMON_INCLUDES $LIBS \
    -o targets/orange || error_exit "Compilation and linking failed"

# Create debug symbols file
if [ "$BUILD_TYPE" != "release" ]; then
    echo -e "${BLUE}Extracting debug symbols...${NC}"
    objcopy --only-keep-debug targets/orange targets/orange.debug
    strip --strip-debug targets/orange
    objcopy --add-gnu-debuglink=targets/orange.debug targets/orange
fi

# Verify build
if [ -f "targets/orange" ]; then
    echo -e "${GREEN}Build completed successfully${NC}"
    echo -e "${BLUE}Executable: $(pwd)/targets/orange${NC}"
    if [ "$BUILD_TYPE" != "release" ]; then
        echo -e "${BLUE}Debug symbols: $(pwd)/targets/orange.debug${NC}"
    fi
    echo -e "${GREEN}Checking dependencies...${NC}"
    ldd targets/orange
else
    error_exit "Build failed - executable not created"
fi