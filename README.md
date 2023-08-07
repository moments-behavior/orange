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

4. orange depends on github repos like `Dear Imgui`, `Implot` etc. Once cloned the repo, use the following command to pull submodules
```
git submodule init
git submodule update
```

5. If you are building the project the first time, uncomment line 11 ~ line 21 for building `ImGui` and `Implot` obejct files. Run
```
./build.sh
```
You can comment out line 11 ~ line 21 to reducing building time afterwards. 

## Contribute to the project 
If you wish to contribute to the project, please make changes to your local branch, and create a pull request before pushing.  
