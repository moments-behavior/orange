## Build OpenCV

1. Download and upzip `opencv-4.8.0.zip` and `opencv_contrib-4.8.0.zip`. Unzip the folders to `~/build/`, for instance. 

2. To build OpenCV with opencv sfm, please follow instructions from: https://docs.opencv.org/4.x/db/db8/tutorial_sfm_installation.html first to install sfm dependency. Ceres solver is optional. If you wish to install ceres solver, a more detailed installation instruction can be found at: http://ceres-solver.org/installation.html#linux. For simplicity, turns off CUDA as follow,
   
```
git clone --recurse-submodules https://github.com/ceres-solver/ceres-solver
cd ceres-solver
mkdir build && cd build
cmake -DUSE_CUDA=OFF ..
make -j $(nproc)
make test
sudo make install
```

3. Build OpenCV

If you are also going to use `red`, the labeling tool, please intall with `cuDNN` first, and config OpenCV differently with `cuDNN`. 

### Install cuDNN (depends on CUDA installation)
- download the cudnn install files (we use `cudnn 8.9.3` with `driver 525.105.17` and `cuda 12.0` )
- you may run the commands below for the exact version or download a TAR file for `cudnn-linux-x86_64-8.9.3.28_cuda12-archive.tar.xz` from the [cudnn version archives](https://developer.nvidia.com/rdp/cudnn-archive)
- extract the file
- copy cudnn files to where your `cuda` is installed -- we assume it is installed at `/usr/local/cuda` 
  ```
  sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
  sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
  sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
  ```
- verify installation and cudnn version
  ```
  source ~/.bashrc
  cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
  ```
  you can expect an output like:
  ```
  #define CUDNN_MAJOR 8
  #define CUDNN_MINOR 9
  #define CUDNN_PATCHLEVEL 3
  --
  #define CUDNN_VERSION (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)
  ```

### Install OpenCV
- download and upzip `opencv-4.8.0.zip` and `opencv_contrib-4.8.0.zip`. Unzip the folders to `~/build/`, for instance. Note, if you are using cuda 12.2, please download opencv-4.10 instead.

- to build OpenCV with opencv sfm, please follow instructions from: https://docs.opencv.org/4.x/db/db8/tutorial_sfm_installation.html first to install sfm dependency. Ceres solver is optional. If you wish to install ceres solver, a more detailed installation instruction can be found at: http://ceres-solver.org/installation.html#linux. At the time of test, one need to set CMake flag USE_CUDA=OFF for ceres.  

- build OpenCV using

```
cd opencv-4.8.0/ 
mkdir build
cd build 
```

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
-D CUDA_ARCH_BIN=7.5 \
-D WITH_V4L=ON \
-D WITH_QT=ON \
-D WITH_OPENGL=ON \
-D WITH_GSTREAMER=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_PC_FILE_NAME=opencv.pc \
-D OPENCV_ENABLE_NONFREE=ON \
-D OPENCV_EXTRA_MODULES_PATH=~/build/opencv_contrib-4.8.0/modules \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D INSTALL_C_EXAMPLES=ON \
-D BUILD_EXAMPLES=ON ..
```

```
make -j $(nproc) 
sudo make install
```

### Otherwise, you can simply do 

```
cd opencv-4.8.0/ 
mkdir build
cd build 
```

```
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D WITH_TBB=ON \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D WITH_CUDA=OFF \
-D BUILD_opencv_cudacodec=OFF \
-D WITH_OPENGL=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_PC_FILE_NAME=opencv.pc \
-D OPENCV_ENABLE_NONFREE=ON \
-D OPENCV_EXTRA_MODULES_PATH=~/build/opencv_contrib-4.8.0/modules \
-D INSTALL_PYTHON_EXAMPLES=OFF ..
```

This instruction assumes the `opencv_contrib-4.8.0` is at `~/Build/opencv_contrib-4.8.0`. This will install OpenCV at `/usr/local`. 

```
make -j $(nproc)
sudo make install
```
