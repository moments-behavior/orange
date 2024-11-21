DIR_FFMPEG=$HOME/nvidia/ffmpeg
DIR_TENSORRT=$HOME/nvidia/TensorRT
sudo LD_LIBRARY_PATH=/usr/local/cuda/lib64:$DIR_FFMPEG/build/lib:$DIR_TENSORRT/lib ./targets/orange