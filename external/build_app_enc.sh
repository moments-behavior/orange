#!/bin/bash
mkdir -p targets_nvenc;
rm -f targets_nvenc/AppEncCuda;
g++ -Ofast -ffast-math -std=c++14 \
    -o targets_nvenc/NvEncoder.o NvEncoder/NvEncoder.cpp \
    -o targets_nvenc/NvEncoder/NvEncoderCuda.o NvEncoder/NvEncoderCuda.cpp \
    -o targets_nvenc/NvEncoder/NvEncoderOutputInVidMemCuda.o NvEncoder/NvEncoderOutputInVidMemCuda.cpp \
    -o targets_nvenc/AppEncCuda.o AppEncCuda.cpp \
    -I/usr/local/cuda-11.4/include -I/home/ash/nvenc_api/include -I/home/ash/src/orange/external/Utils -I/home/ash/src/orange/external/NvEncoder \
    -lm \
    -lpthread;
sudo ./targets_nvenc/AppEncCuda;