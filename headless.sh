#!/bin/bash
mkdir -p targets2;
rm -f targets/orange_headless_slave;
nvcc -c src/cuda_line_reorder.cu -arch=sm_80 -o targets2/cuda_line_reorder.o

g++ -Ofast -ffast-math -std=c++17 \
    targets2/cuda_line_reorder.o \
    -o targets2/*.o \
    -o targets2/orange_headless_slave -I ./src/ src/orange_headless_slave.cpp src/camera_driver_helper.cpp src/camera.cpp src/video_capture_gpu.cpp \
    -I./src/NvEncoder/ ./src/NvEncoder/*.cpp \
    -I./nvenc_api/include -I/opt/EVT/eSDK/include/ -I/usr/local/cuda/include \
    -L/opt/EVT/eSDK/lib/ -lEmergentCamera  -lEmergentGenICam  -lEmergentGigEVision \
    -lm \
    -lpthread \
    -L/usr/local/cuda/lib64/ -lcudart -lcuda -lnppicc -lnppidei -lnvidia-encode \
    -I$HOME/nvidia/ffmpeg/build/include/ \
    -L$HOME/nvidia/ffmpeg/build/lib/ -lavformat -lswscale -lswresample -lavutil -lavcodec
