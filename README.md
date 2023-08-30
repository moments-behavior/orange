# orange :orange: 
A GUI-based C/C++ library for emergent cameras

![gui](images/gui.png)



## Features 
1. Multiple cameras streaming 
2. PTP synchronization 
3. GPU accelerated encoding (h264, h265)
3. Support 7MP, 65MP, 100G, mono or color Emergent cameras

## Benchmark
Encoding performance using GPU A6000 with 7MP Emergent camera

![encoding_benchmark](images/encoding_benchmark.png)

## Build instructions 
1. Install CUDA 11.7 fowllow instructions from Nvidia Cuda install instructions.
2. Install Emergent camera SDK
3. Install FFmpeg 4.4 as shared library
```
./configure --prefix=$(pwd)/build --disable-static --enable-shared --enable-nonfree --enable-cuda-nvcc --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64
```
The project build file `build.sh` assume you install FFmepg at `$HOME/nvidia/ffmpeg/build/include/` and `$HOME/nvidia/ffmpeg/build/lib/`, if you install it at a different location, please change the `build.sh` accordingly. 

3. Install OpenGL and GLEW
```
sudo apt-get install libglfw3
sudo apt-get install libglfw3-dev
sudo apt-get install libglew-dev
```

4. Install CUDNN 

5. Install OPENCV with CUDNN support

```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D WITH_TBB=ON \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D WITH_CUBLAS=1 \
-D WITH_CUDA=ON \
-D BUILD_opencv_cudacodec=OFF \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=ON \
-D CUDA_ARCH_BIN=8.0 \
-D WITH_V4L=ON \
-D WITH_OPENGL=ON \
-D WITH_GSTREAMER=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_PC_FILE_NAME=opencv.pc \
-D OPENCV_ENABLE_NONFREE=ON \
-D OPENCV_EXTRA_MODULES_PATH=~/Build/opencv_contrib-4.6.0/modules \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D INSTALL_C_EXAMPLES=ON \
-D BUILD_EXAMPLES=ON ..
```
6. orange depends on github repos like `Dear Imgui`, `Implot` etc. Once cloned the repo, use the following command to pull submodules
```
git submodule init
git submodule update
```

7. If you are building the project the first time, uncomment line 11 ~ line 21 for building `ImGui` and `Implot` obejct files. Run
```
./build.sh
```
You can comment out line 11 ~ line 21 to reducing building time afterwards. 

## Contribute to the project 
If you wish to contribute to the project, please make changes to your local branch, and create a pull request before pushing.  
