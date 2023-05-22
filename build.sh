#!/bin/bash
mkdir -p targets;
rm -f targets/orange;
nvcc -c src/cuda_line_reorder.cu -arch=sm_80 -o targets/cuda_line_reorder.o
nvcc -c src/color_conversion_gpu.cu -arch=sm_80 -o targets/color_conversion_gpu.o


DIR_IMGUI="third_party/imgui"
DIR_IMPLOT="third_party/implot"


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



g++ -Ofast -ffast-math -std=c++17 \
    targets/color_conversion_gpu.o \
    -o targets/*.o \
    -o targets/orange -I ./src/ src/orange.cpp src/camera_driver_helper.cpp src/camera.cpp src/video_capture_gpu.cpp src/buffer_utils.cpp \
    -I./third_party/imgui \
    -I./third_party/imgui/backends \
    -I$DIR_IMPLOT \
    -Ithird_party/imgui-filebrowser \
    -I./third_party/IconFontCppHeaders \
    -I./src/NvEncoder/ ./src/NvEncoder/*.cpp \
    -I./nvenc_api/include -I/opt/EVT/eSDK/include/ -I/usr/local/cuda/include \
    -L/opt/EVT/eSDK/lib/ -lEmergentCamera  -lEmergentGenICam  -lEmergentGigEVision \
    -lm \
    -lpthread \
    -L/usr/local/cuda/lib64/ -lcudart -lcuda -lnppicc -lnppidei -lnvidia-encode -lnppc \
    -lGLEW -lGLU -lGL \
    -I$HOME/nvidia/ffmpeg/build/include/ \
    -L$HOME/nvidia/ffmpeg/build/lib/ -lavformat -lswscale -lswresample -lavutil -lavcodec \
    -I/usr/local/include/opencv4 \
    -lopencv_sfm -lopencv_core -lopencv_bgsegm -lopencv_imgcodecs -lopencv_imgproc -lopencv_video -lopencv_highgui -lopencv_videoio -lopencv_calib3d -lopencv_dnn -lopencv_features2d \
    -I/usr/local/include/aruco \
    -laruco \
    `pkg-config --static --libs glfw3`

sudo ./targets/orange;