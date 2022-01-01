#!/bin/bash
mkdir -p targets_nvenc;
rm -f targets_nvenc/AppEncCuda;
nvcc -c -arch=sm_80 Utils/*.cu -odir targets_nvenc
g++ -Ofast -ffast-math -std=c++14 -o targets_nvenc/*.o -o targets_nvenc/AppEncCuda AppEncCuda.cpp -I ./NvEncoder/ ./NvEncoder/*.cpp \
    -I/usr/local/cuda-11.4/include -I/home/ash/src/orange/external/NvEncoder -I/home/ash/nvenc_api/include \
    -L/usr/local/cuda-11.4/lib64/ \
    -lcudart -lcuda -lnvcuvid -lnvidia-encode
rm -f targets_nvenc/*.o