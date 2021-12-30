#!/bin/bash
mkdir -p targets_nvenc;
rm -f targets/AppEncCuda;
g++ -Ofast -ffast-math -std=c++14 \
    -o targets/AppEncCuda.o src/AppEncCuda.cpp \
    -lm \
    -lpthread;
sudo ./targets_nvenc/AppEncCuda;