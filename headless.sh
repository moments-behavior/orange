#!/bin/bash
mkdir -p targets2;
rm -f targets2/orange_headless;
nvcc -c src/kernel.cu -arch=sm_80 -o targets2/kernel.o

g++ -Ofast -ffast-math -std=c++17 targets2/*.o \
    -o targets2/orange_headless -I ./src/ src/orange_headless.cpp src/camera_driver_helper.cpp src/camera.cpp src/video_capture_gpu.cpp \
    -I./src/NvEncoder/ ./src/NvEncoder/*.cpp \
    -I./nvenc_api/include -I/opt/EVT/eSDK/include/ -I/usr/local/cuda/include \
    -L/opt/EVT/eSDK/lib/ -lEmergentCamera  -lEmergentGenICam  -lEmergentGigEVision \
    -lm \
    -lpthread \
    -L/usr/local/cuda/lib64/ -lcudart -lcuda -lnppicc -lnppidei -lnvidia-encode -lnppc \
    -I$HOME/nvidia/ffmpeg/build/include/ \
    -L$HOME/nvidia/ffmpeg/build/lib/ -lavformat -lswscale -lswresample -lavutil -lavcodec
