#!/bin/bash
mkdir -p targets_nvenc;
rm -f targets_nvenc/AppEncCuda;
#nvcc -c -arch=sm_80 Utils/*.cu -odir targets_nvenc
g++ -Ofast -ffast-math -std=c++14 -o targets_nvenc/*.o -o targets_nvenc/gpu_pipeline gpu_pipeline.cpp -I./NvEncoder/ ./NvEncoder/*.cpp \
    -I/usr/local/cuda-11.4/include -I/home/ash/nvenc_api/include \
    -L/usr/local/cuda-11.4/lib64/ -lpthread -lcudart -lcuda -lnvidia-encode -lnppicc \
    `PKG_CONFIG_PATH=/path/ffmpeg/lib/pkgconfig/ pkg-config --cflags libavformat libswscale libswresample libavutil libavcodec` `PKG_CONFIG_PATH=/path/ffmpeg/lib/pkgconfig/ pkg-config --libs libavformat libswscale libswresample libavutil libavcodec`
rm -f targets_nvenc/*.o