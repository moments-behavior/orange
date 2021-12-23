#!/bin/bash

mkdir -p targets;
rm -f targets/orange_debug;
clang++ -O0 -g -ffast-math -std=c++14 -lm -pthread -o targets/orange_debug src/orange.cpp;
./targets/yellow_debug;
