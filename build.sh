#!/bin/bash
mkdir -p targets;
rm -f targets/orange;
g++ -Ofast -ffast-math -std=c++14 \
    -o targets/*.o \
    -o targets/orange -I ./src/ src/*.cpp \
    -I./src/NvEncoder/ ./src/NvEncoder/*.cpp \
    -I./nvenc_api/include -I/usr/include/opencv4 -I/opt/EVT/eSDK/include/ -I/usr/lib/x86_64-linux-gnu/gstreamer-1.0 -I/usr/local/cuda-11.4/include \
    -L/opt/EVT/eSDK/lib/ -lEmergentCamera  -lEmergentGenICam  -lEmergentGigEVision \
    -lm \
    -lpthread \
    -lopencv_core -lopencv_imgcodecs -lopencv_bgsegm -lopencv_imgproc -lopencv_video -lopencv_highgui -lopencv_videoio \
    -lgstreamer-1.0 \
    -L/usr/local/cuda-11.4/lib64/ -lcudart -lcuda -lnppicc -lnvidia-encode \
    -lGLEW -lGLU -lGL \
    `pkg-config --cflags --libs x11` \
    `PKG_CONFIG_PATH=/path/ffmpeg/lib/pkgconfig/ pkg-config --cflags libavformat libswscale libswresample libavutil libavcodec` `PKG_CONFIG_PATH=/path/ffmpeg/lib/pkgconfig/ pkg-config --libs libavformat libswscale libswresample libavutil libavcodec`
sudo ./targets/orange;