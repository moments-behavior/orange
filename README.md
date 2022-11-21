# orange :orange: 
A light weight C++ library for emergent camera


## Benchmark
Encoding performance using GPU A6000 with 7MP Emergent camera

![encoding_benchmark](images/encoding_benchmark.png)
Run `build.sh` in local folder

## Build instructions (Windows)

1. Update submodules
```
git submodule update --init
```
2. Install Visual Studio 2022
3. Install CMake
4. Install CUDA 11.6
5. Install Emergent camera SDK 2.36.01
6. Download Video Codec SDK 11.1.5
7. Copy `nvcuvid.lib` and `nvencodeapi.lib` to `nvcodec\lib`.
8. Run `cmake` to configure the project. An easy way is to use the [CMake Tools extension for Visual Studio Code](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools).

## Build instructions 
1. Install Emergent camera SDK
2. Install FFmpeg as shared library
```
./configure --prefix=$(pwd)/build --disable-static --enable-shared --enable-nonfree --enable-cuda-nvcc --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64
```

3. Install OpenGL and GLEW, GLM
```
sudo apt-get install libglfw3
sudo apt-get install libglfw3-dev
sudo apt-get install libglew-dev
sudo apt-get install libglm-dev
```

4. PTP setting for synchronization

```
sudo apt install linuxptp
```
```
sudo vim /etc/ptp4l.conf
```
Put following in the file
```
[global]
verbose 1
boundary_clock_jbod 1 
```

```
sudo ptp4l -i enp1s0f0 -i enp1s0f1 -f /etc/ptp4l.conf
```
