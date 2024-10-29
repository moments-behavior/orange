#!/bin/bash
mkdir -p targets;
rm -f targets/orange;
nvcc -c src/kernel.cu -arch=sm_80 -o targets/kernel.o

DIR_IMGUI="third_party/imgui"
DIR_IMGUI_BACKEND="third_party/imgui/backends"
DIR_IMPLOT="third_party/implot"
DIR_FILEBROWSER="third_party/imgui-filebrowser"
DIR_ICONFONT="third_party/IconFontCppHeaders"

# g++ -std=c++11 -I./third_party/imgui -I./third_party/imgui/backends -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o targets/imgui.o ./third_party/imgui/imgui.cpp
# g++ -std=c++11 -I./third_party/imgui -I./third_party/imgui/backends -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o targets/imgui_demo.o ./third_party/imgui/imgui_demo.cpp
# g++ -std=c++11 -I./third_party/imgui -I./third_party/imgui/backends -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o targets/imgui_draw.o ./third_party/imgui/imgui_draw.cpp
# g++ -std=c++11 -I./third_party/imgui -I./third_party/imgui/backends -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o targets/imgui_tables.o ./third_party/imgui/imgui_tables.cpp
# g++ -std=c++11 -I./third_party/imgui -I./third_party/imgui/backends -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o targets/imgui_widgets.o ./third_party/imgui/imgui_widgets.cpp
# g++ -std=c++11 -I./third_party/imgui -I./third_party/imgui/backends -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o targets/imgui_impl_glfw.o ./third_party/imgui/backends/imgui_impl_glfw.cpp
# g++ -std=c++11 -I./third_party/imgui -I./third_party/imgui/backends -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o targets/imgui_impl_opengl3.o ./third_party/imgui/backends/imgui_impl_opengl3.cpp

# g++ -std=c++17 -I$DIR_IMPLOT -I$DIR_IMGUI -g -Wall -c -o targets/implot.o $DIR_IMPLOT/implot.cpp
# g++ -std=c++17 -I$DIR_IMPLOT -I$DIR_IMGUI -g -Wall -c -o targets/implot_items.o $DIR_IMPLOT/implot_items.cpp
# g++ -std=c++17 -I$DIR_IMPLOT -I$DIR_IMGUI -g -Wall -c -o targets/implot_demo.o $DIR_IMPLOT/implot_demo.cpp

g++ -ggdb -Ofast -ffast-math -std=c++17 targets/*.o \
    -o targets/orange -I ./src/ src/orange.cpp src/network_base.cpp src/FFmpegWriter.cpp src/camera_driver_helper.cpp src/camera.cpp src/video_capture.cpp src/acquire_frames.cpp src/offthreadmachine.cpp src/opengldisplay.cpp src/threadworker.cpp src/gpu_video_encoder.cpp src/yolov8_det.cpp \
    -I$DIR_IMGUI \
    -I$DIR_IMGUI_BACKEND \
    -I$DIR_IMPLOT \
    -I$DIR_FILEBROWSER \
    -I$DIR_ICONFONT \
    -I./src/NvEncoder/ ./src/NvEncoder/*.cpp \
    -I./nvenc_api/include -I/opt/EVT/eSDK/include/ -I/usr/local/cuda/include \
    -L/opt/EVT/eSDK/lib/ -lEmergentCamera  -lEmergentGenICam  -lEmergentGigEVision \
    -lm \
    -lpthread \
    -I./third_party/flatbuffers/include \
    -lenet -I/usr/local/include/ \
    -L/usr/local/cuda/lib64/ -lcudart -lcuda -lnppicc -lnppidei -lnvidia-encode -lnppc -lnppig -lnppial \
    -lGLEW -lGL \
    -I$HOME/nvidia/ffmpeg/build/include/ \
    -L$HOME/nvidia/ffmpeg/build/lib/ -lavformat -lswscale -lswresample -lavutil -lavcodec \
    -I/usr/local/include/opencv4 \
    -lopencv_sfm -lopencv_core -lopencv_bgsegm -lopencv_imgcodecs -lopencv_imgproc -lopencv_video -lopencv_highgui -lopencv_videoio -lopencv_calib3d -lopencv_dnn -lopencv_features2d \
    -I/home/user/build/TensorRT-8.6.1.6/include \
    -L/home/user/build/TensorRT-8.6.1.6/lib/ -lnvinfer -lnvinfer_plugin \
    `pkg-config --static --libs glfw3`
