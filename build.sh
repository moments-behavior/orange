#!/bin/bash
mkdir -p targets;
rm -f targets/orange;
g++ -Ofast -ffast-math -std=c++14 \
    -o targets/*.o \
    -o targets/orange -I ./src/ src/*.cpp \
    -I./src/NvEncoder/ ./src/NvEncoder/*.cpp \
    -I./nvenc_api/include -I/opt/EVT/eSDK/include/ -I/usr/local/cuda/include \
    -L/opt/EVT/eSDK/lib/ -lEmergentCamera  -lEmergentGenICam  -lEmergentGigEVision \
    -lm \
    -lpthread \
    -L/usr/local/cuda/lib64/ -lcudart -lcuda -lnppicc -lnvidia-encode \
    -lGLEW -lGLU -lGL \
    -I/home/user/nvidia/ffmpeg/build/include/ \
    -L/home/user/nvidia/ffmpeg/build/lib/ -lavformat -lswscale -lswresample -lavutil -lavcodec \
    `pkg-config --cflags --libs x11`
sudo ./targets/orange;