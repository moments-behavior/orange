#!/bin/bash
mkdir -p targets;
rm -f targets/orange;
g++ -Ofast -ffast-math -std=c++14 -o targets/video_capture.o src/video_capture.cpp -o targets/camera_driver_helper.o src/camera_driver_helper.cpp -o targets/camera.o src/camera.cpp -o targets/orange src/orange.cpp -I/usr/include/opencv4 -I/opt/EVT/eSDK/include/ -L/opt/EVT/eSDK/lib/ -lEmergentCamera  -lEmergentGenICam  -lEmergentGigEVision -lm -lpthread -lopencv_core -lopencv_imgcodecs -lopencv_bgsegm -lopencv_imgproc -lopencv_video -lopencv_highgui; 
sudo ./targets/orange;
