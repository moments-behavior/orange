# orange :orange: 
A multi-camera capture, streaming and recording GUI application for emergent cameras in C++

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
1. Install CUDA (the software has been tested with version 11.7 - 12.1) fowllow instructions from Nvidia Cuda install instructions.

2. Install Emergent camera SDK:
Make sure you can stream all cameras individually with Emergent `eCapture`.  

3. Install FFmpeg 4.4
Refer to `docs/install_ffmpeg.md` for detailed instruction for building FFmpeg 4.4. 

The project build file `build.sh` assume you install FFmpeg at `$HOME/nvidia/ffmpeg/build/include/` and `$HOME/nvidia/ffmpeg/build/lib/`, if you install it at a different location, please change the `build.sh` `DIR_FFMPEG` to match your install directory. 

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
