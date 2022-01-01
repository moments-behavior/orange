#!/bin/bash
mkdir -p targets;
rm -f targets/orange;
g++ -Ofast -ffast-math -std=c++14 \
    -o targets/orange src/orange.cpp -I ./src/ ./src/*.cpp \
    -I/usr/include/opencv4 -I/opt/EVT/eSDK/include/ -I/usr/lib/x86_64-linux-gnu/gstreamer-1.0 \
    -L/opt/EVT/eSDK/lib/ -lEmergentCamera  -lEmergentGenICam  -lEmergentGigEVision \
    -lm \
    -lpthread \
    -lopencv_core -lopencv_imgcodecs -lopencv_bgsegm -lopencv_imgproc -lopencv_video -lopencv_highgui -lopencv_videoio \
    -lgstreamer-1.0; 
sudo ./targets/orange;