mkdir -p tools;
nvcc -c src/kernel.cu -arch=sm_80 -o tools/kernel.o

DIR_TENSORRT=$HOME/nvidia/TensorRT

g++ -Ofast -ffast-math -std=c++17 tools/*.o -o tools/yolo_tester \
    -I./src/ src/yolo_offline_tester.cpp src/yolov8_det.cpp \
    -I/usr/local/include/opencv4 \
    -lopencv_sfm -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_videoio -lopencv_highgui -lopencv_video -lopencv_calib3d \
    -I$DIR_TENSORRT/include -L$DIR_TENSORRT/lib/ -lnvinfer -lnvinfer_plugin \
    -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcudart -lcuda -lnppicc -lnppidei -lnvidia-encode -lnppc -lnppig -lnppial
