#!/bin/bash

mkdir -p targets;
rm -f targets/orange;
clang++ -Ofast -ffast-math -std=c++14 -lm -pthread -o targets/orange src/orange.cpp;
./targets/orange;
