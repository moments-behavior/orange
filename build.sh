#!/bin/bash
mkdir -p targets;
rm -f targets/orange;

g++ -std=c++11 -I./third_party/imgui -I./third_party/imgui/backends -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o targets/imgui.o ./third_party/imgui/imgui.cpp
g++ -std=c++11 -I./third_party/imgui -I./third_party/imgui/backends -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o targets/imgui_demo.o ./third_party/imgui/imgui_demo.cpp
g++ -std=c++11 -I./third_party/imgui -I./third_party/imgui/backends -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o targets/imgui_draw.o ./third_party/imgui/imgui_draw.cpp
g++ -std=c++11 -I./third_party/imgui -I./third_party/imgui/backends -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o targets/imgui_tables.o ./third_party/imgui/imgui_tables.cpp
g++ -std=c++11 -I./third_party/imgui -I./third_party/imgui/backends -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o targets/imgui_widgets.o ./third_party/imgui/imgui_widgets.cpp
g++ -std=c++11 -I./third_party/imgui -I./third_party/imgui/backends -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o targets/imgui_impl_glfw.o ./third_party/imgui/backends/imgui_impl_glfw.cpp
g++ -std=c++11 -I./third_party/imgui -I./third_party/imgui/backends -g -Wall -Wformat `pkg-config --cflags glfw3` -c -o targets/imgui_impl_opengl3.o ./third_party/imgui/backends/imgui_impl_opengl3.cpp



g++ -Ofast -ffast-math -std=c++11 \
    -o targets/*.o \
    -o targets/orange -I ./src/ src/*.cpp \
    -I./third_party/imgui \
    -I./third_party/imgui/backends \
    -I./third_party/NvEncoder/ ./third_party/NvEncoder/*.cpp \
    -I./third_party/NvEncoder/include -I/usr/include/opencv4 -I/opt/EVT/eSDK/include/ -I/usr/lib/x86_64-linux-gnu/gstreamer-1.0 -I/usr/local/cuda-11.4/include \
    -L/opt/EVT/eSDK/lib/ -lEmergentCamera  -lEmergentGenICam  -lEmergentGigEVision \
    -lm \
    -lpthread \
    -lopencv_core -lopencv_imgcodecs -lopencv_bgsegm -lopencv_imgproc -lopencv_video -lopencv_highgui -lopencv_videoio \
    -lgstreamer-1.0 \
    -L/usr/local/cuda-11.4/lib64/ -lcudart -lcuda -lnppicc -lnvidia-encode \
    -lGLEW -lGLU \
    `pkg-config --cflags glfw3`  -lGL `pkg-config --static --libs glfw3` \
    `pkg-config --cflags --libs x11` \
    `pkg-config --cflags libavformat libswscale libswresample libavutil libavcodec` \
    `pkg-config --libs libavformat libswscale libswresample libavutil libavcodec`
sudo ./targets/orange;