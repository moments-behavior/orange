#!/bin/bash

# Create targets directory and clean existing binary
mkdir -p targets
rm -f targets/orange

# Base directories
ORANGE_ROOT="/opt/orange/lib"
FFMPEG_DIR="${ORANGE_ROOT}/ffmpeg-nvidia"
TENSORRT_DIR="/usr/local/TensorRT-10.0.1.6"
CUDA_DIR="/usr/local/cuda"

# Third party directories
DIR_IMGUI="third_party/imgui"
DIR_IMGUI_BACKEND="third_party/imgui/backends"
DIR_IMPLOT="third_party/implot"
DIR_FILEBROWSER="third_party/ImGuiFileDialog"
DIR_ICONFONT="third_party/IconFontCppHeaders"

# Compile CUDA kernel
nvcc -c src/kernel.cu -arch=sm_86 -o targets/kernel.o

# Compile ImGui core
g++ -std=c++11 -I$DIR_IMGUI -I$DIR_IMGUI_BACKEND -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o targets/imgui.o $DIR_IMGUI/imgui.cpp
g++ -std=c++11 -I$DIR_IMGUI -I$DIR_IMGUI_BACKEND -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o targets/imgui_demo.o $DIR_IMGUI/imgui_demo.cpp
g++ -std=c++11 -I$DIR_IMGUI -I$DIR_IMGUI_BACKEND -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o targets/imgui_draw.o $DIR_IMGUI/imgui_draw.cpp
g++ -std=c++11 -I$DIR_IMGUI -I$DIR_IMGUI_BACKEND -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o targets/imgui_tables.o $DIR_IMGUI/imgui_tables.cpp
g++ -std=c++11 -I$DIR_IMGUI -I$DIR_IMGUI_BACKEND -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o targets/imgui_widgets.o $DIR_IMGUI/imgui_widgets.cpp

# Compile ImGui backends
g++ -std=c++11 -I$DIR_IMGUI -I$DIR_IMGUI_BACKEND -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o targets/imgui_impl_glfw.o $DIR_IMGUI_BACKEND/imgui_impl_glfw.cpp
g++ -std=c++11 -I$DIR_IMGUI -I$DIR_IMGUI_BACKEND -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o targets/imgui_impl_opengl3.o $DIR_IMGUI_BACKEND/imgui_impl_opengl3.cpp

# Compile ImPlot
g++ -std=c++17 -I$DIR_IMPLOT -I$DIR_IMGUI -g -Wall -c -o targets/implot.o $DIR_IMPLOT/implot.cpp
g++ -std=c++17 -I$DIR_IMPLOT -I$DIR_IMGUI -g -Wall -c -o targets/implot_items.o $DIR_IMPLOT/implot_items.cpp
g++ -std=c++17 -I$DIR_IMPLOT -I$DIR_IMGUI -g -Wall -c -o targets/implot_demo.o $DIR_IMPLOT/implot_demo.cpp

# Main compilation command
g++ -Ofast -ffast-math -std=c++17 targets/*.o \
    -o targets/orange -I ./src/ src/orange.cpp src/network_base.cpp src/FFmpegWriter.cpp \
    src/camera.cpp src/video_capture.cpp src/acquire_frames.cpp src/offthreadmachine.cpp \
    src/opengldisplay.cpp src/threadworker.cpp src/gpu_video_encoder.cpp src/yolov8_det.cpp \
    $DIR_FILEBROWSER/ImGuiFileDialog.cpp \
    -I$DIR_IMGUI \
    -I$DIR_IMGUI_BACKEND \
    -I$DIR_IMPLOT \
    -I$DIR_FILEBROWSER \
    -I$DIR_ICONFONT \
    -I./src/NvEncoder/ ./src/NvEncoder/*.cpp \
    -I./nvenc_api/include -I/opt/EVT/eSDK/include/ -I$CUDA_DIR/include \
    -L/opt/EVT/eSDK/lib/ -lEmergentCamera -lEmergentGenICam -lEmergentGigEVision \
    -lm \
    -lpthread \
    -I./third_party/flatbuffers/include \
    -lenet -I/usr/local/include/ \
    -L$CUDA_DIR/lib64/ -lcudart -lcuda -lnppicc -lnppidei -lnvidia-encode -lnppc -lnppig -lnppial \
    -lGLEW -lGL \
    -I$FFMPEG_DIR/include/ \
    -L$FFMPEG_DIR/lib/ -lavformat -lswscale -lswresample -lavutil -lavcodec \
    -I$ORANGE_ROOT/opencv/include/opencv4 \
    -L$ORANGE_ROOT/opencv/lib -lopencv_sfm -lopencv_core -lopencv_imgcodecs -lopencv_imgproc \
    -I$TENSORRT_DIR/include \
    -L$TENSORRT_DIR/lib/ -lnvinfer -lnvinfer_plugin \
    `pkg-config --static --libs glfw3`