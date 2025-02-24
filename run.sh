DIR_FFMPEG=$HOME/build/FFmpeg
DIR_TENSORRT=$HOME/build/TensorRT-10.6.0.26
sudo LD_LIBRARY_PATH=/usr/local/cuda/lib64:$DIR_FFMPEG/build/lib:$DIR_TENSORRT/lib ./targets/orange