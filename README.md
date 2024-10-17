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

## Dependencies
1. Emergent SDK
2. CUDA Toolkit
3. FFmpeg 
4. OpenCV 
5. OpenGL and GLEW 
6. DearImGUI and related repos
7. TensorRT 
8. ENET

## Build instructions 
1. Install CUDA (the software has been tested with version 11.7 - 12.1): https://docs.nvidia.com/cuda/cuda-installation-guide-linux/.

2. Install Emergent camera SDK:
Make sure you can stream all cameras individually with Emergent `eCapture`.  

3. Install FFmpeg 4.4
Refer to `docs/install_ffmpeg.md` for detailed instruction for building FFmpeg 4.4. 

The project build file `build.sh` assumes FFmpeg is installed at `$HOME/nvidia/ffmpeg/build/include/` and `$HOME/nvidia/ffmpeg/build/lib/`, if you install it at a different location, please change the `build.sh` `DIR_FFMPEG` to match your install directory. 

4. Install OpenGL and GLEW
```
sudo apt-get install libglfw3
sudo apt-get install libglfw3-dev
sudo apt-get install libglew-dev
```

5. Install OpenCV
Refer to `docs/install_opencv.md` for detailed instruction for building OpenCV. 

6. Install TensorRT 
The repo has been tested with TensorRT-8.6.1.6. Download and install TensorRT in `~/build/`, followings instruction: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html. For instance, look for Tar File Installation. 

7. Install ENET
Follow instruction: http://enet.bespin.org/Installation.html. 

8. Clone the repo and submodules

```
git clone https://github.com/JohnsonLabJanelia/orange.git
git submodule init
git submodule update
```

9. If you are building the project for the first time, uncomment line 13 ~ line 23 for building `ImGui` and `ImPlot` object files. Run
```
./build.sh
```
You can comment out line 13 ~ line 23 to reduce compiling time afterwards. 

Once built, it will make a folder called `targets`. The executable `orange` is the application. Start the program using the run script. 

```
./run.sh
```

## Use the Program
Create a `config` folder in the home directory. Create subdirectories `local` and `network` in the config folder. If there is only one server being used, create folders with camera configs in the `local` directory. For instance, here is an example directory tree of `~/config` folder. 

```
.
├── local
│   ├── 5cam
│   ├── 65
│   ├── 65_full
│   ├── 65_light
│   ├── center_ceiling
│   └── laser
└── network
    ├── 180_gpu_direct
    ├── 180_laser
    └── 180_light
```  

In the node folder (like 5cam folder), it contains 1 or more camera configs `[camera serial].json`. An example config file is in the `config` folder. Please name the file after the serial number of cameras and set the config according to your camera specifications. To enable `gpu_direct`, set `gpu_direct` to true, and set the `gpu_id` to select which gpu to use for image processing of the camera. 

## Contribute to the project 
If you wish to contribute to the project, please make changes to your local branch, and create a pull request before pushing.  


