#!/bin/bash
DIR="$(dirname "$(realpath "$0")")"
PARENT_DIR="$(dirname "$DIR")"
cd $PARENT_DIR
targets_folder=$PARENT_DIR/targets

mkdir -p $targets_folder
rm -f $targets_folder/yolo_offline
nvcc -c src/kernel.cu -arch=sm_80 -o $targets_folder/kernel.o
DIR_TENSORRT=$HOME/nvidia/TensorRT

g++ -Ofast -ffast-math -std=c++17 $targets_folder/kernel.o -o $targets_folder/yolo_offline \
    -I./src/ benchmark/yolo_offline.cpp src/yolov8_det.cpp \
    -I/usr/local/include/opencv4 \
    -lopencv_sfm -lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_highgui -lopencv_video \
    -I$DIR_TENSORRT/include -L$DIR_TENSORRT/lib/ -lnvinfer -lnvinfer_plugin \
    -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcudart -lcuda -lnppicc -lnppidei -lnvidia-encode -lnppc -lnppig -lnppial
