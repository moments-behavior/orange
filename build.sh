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



g++ -Ofast -ffast-math -std=c++14 \
    -o targets/*.o \
    -o targets/orange -I ./src/ src/*.cpp \
    -I./third_party/imgui \
    -I./third_party/imgui/backends \
    -I./src/NvEncoder/ ./src/NvEncoder/*.cpp \
    -I./nvenc_api/include -I/opt/EVT/eSDK/include/ -I/usr/local/cuda/include \
    -L/opt/EVT/eSDK/lib/ -lEmergentCamera  -lEmergentGenICam  -lEmergentGigEVision \
    -lm \
    -lpthread \
    -L/usr/local/cuda/lib64/ -lcudart -lcuda -lnppicc -lnvidia-encode \
    -lGLEW -lGLU -lGL \
    -I/home/user/nvidia/ffmpeg/build/include/ \
    -L/home/user/nvidia/ffmpeg/build/lib/ -lavformat -lswscale -lswresample -lavutil -lavcodec \
    `pkg-config --static --libs glfw3` \
    `pkg-config --cflags --libs x11`
sudo ./targets/orange;