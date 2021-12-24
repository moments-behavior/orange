#!/bin/bash
mkdir -p targets;
rm -f targets/orange;
g++ -Ofast -ffast-math -std=c++14 -o targets/camera.o src/camera.cpp -o targets/orange src/orange.cpp -I/opt/EVT/eSDK/include/ -L/opt/EVT/eSDK/lib/ -lEmergentCamera  -lEmergentGenICam  -lEmergentGigEVision -lm -lpthread;
./targets/orange;
