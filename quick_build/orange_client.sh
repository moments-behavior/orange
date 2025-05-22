#!/bin/bash
DIR="$(dirname "$(realpath "$0")")"
PARENT_DIR="$(dirname "$DIR")"
cd $PARENT_DIR
targets_folder=$PARENT_DIR/targets

mkdir -p $targets_folder
rm -f $targets_folder/orange_client
nvcc -c src/kernel.cu -arch=sm_80 -o $targets_folder/kernel.o

g++ -Ofast -ffast-math -std=c++17 $targets_folder/kernel.o \
    -o $targets_folder/orange_client -I ./src/ src/orange_headless_client.cpp src/network_base.cpp src/FFmpegWriter.cpp src/camera.cpp src/video_capture.cpp src/offthreadmachine.cpp src/acquire_frames_headless.cpp src/threadworker.cpp src/gpu_video_encoder.cpp \
    -I./src/NvEncoder/ ./src/NvEncoder/*.cpp \
    -I./nvenc_api/include -I/opt/EVT/eSDK/include/ -I/usr/local/cuda/include \
    -L/opt/EVT/eSDK/lib/ -lEmergentCamera  -lEmergentGenICam  -lEmergentGigEVision \
    -lm \
    -lpthread \
    -I./third_party/flatbuffers/include \
    -lenet -I/usr/local/include/ \
    -L/usr/local/cuda/lib64/ -lcudart -lcuda -lnppicc -lnppidei -lnvidia-encode -lnppc \
    -I$HOME/nvidia/ffmpeg/build/include/ \
    -L$HOME/nvidia/ffmpeg/build/lib/ -lavformat -lswscale -lswresample -lavutil -lavcodec \
    -I/usr/local/include/opencv4 \
    -lopencv_core -lopencv_imgcodecs -lopencv_imgproc
