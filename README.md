# orange :orange: 
A light weight C++ library for emergent camera


## Benchmark
Encoding performance using GPU A6000 with 7MP Emergent camera

![encoding_benchmark](images/encoding_benchmark.png)
Run `build.sh` in local folder


## Build instructions 
1. Install Emergent camera SDK
2. Install FFmpeg as shared library
```
./configure --prefix=$(pwd)/build --disable-static --enable-shared --enable-nonfree --enable-cuda-nvcc --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64
```

3. Install OpenGL and GLEW
