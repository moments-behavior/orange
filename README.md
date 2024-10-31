# orange :orange: 
A multi-camera capture, streaming and recording GUI application for emergent cameras in C++

![gui](images/gui.png)


## Features 
1. Multiple cameras streaming 
2. PTP synchronization 
3. GPU accelerated encoding (h264, h265)
3. Support 7MP, 65MP, 100G, mono or color Emergent cameras
4. Multiple servers communication

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
1. Install CUDA (the software has been tested with version 12.x) and Emergent camera SDK. Follow instructions in `docs/install_linux_cuda_eSDK.md`. Make sure you can stream all cameras individually with Emergent `eCapture`.  

2. Install FFmpeg 4.4

Refer to [`docs/install_ffmpeg.md`](docs/install_ffmpeg.md) for detailed instruction for building FFmpeg 4.4. 

The project build file [`build.sh`](build.sh) assumes FFmpeg is installed at `$HOME/nvidia/ffmpeg`, if you installed it at a different location, please change the `build.sh` `DIR_FFMPEG` to match your install directory. 

3. Install OpenGL and GLEW
```
sudo apt-get install libglfw3
sudo apt-get install libglfw3-dev
sudo apt-get install libglew-dev
```

4. Install OpenCV
Refer to [`docs/install_opencv.md`](docs/install_opencv.md) for detailed instruction for building OpenCV. 

5. Install TensorRT 
The repo has been tested with TensorRT-8.6.1.6. Followings instruction: [`docs/install_tensorrt.md`](docs/install_tensorrt.md). The project build assumes TensorRT installed at `$HOME/nvidia/TensorRT-8.6.1.6`. If you installed in at a different location, please change the [`build.sh`](build.sh) `DIR_TENSORRT` to match your install directory.

6. Install ENET
Follow instruction: http://enet.bespin.org/Installation.html. 

7. Clone the repo and submodules

```
git clone https://github.com/JohnsonLabJanelia/orange.git
git submodule init
git submodule update
```

8. If you are building the project for the first time, uncomment [`line 15 ~ line 25`](https://github.com/JohnsonLabJanelia/orange/blob/5d7a1b9ec4738f8075895a2a0b27cff556aca834/build.sh#L15) for building `ImGui` and `ImPlot` object files. Run
```
./build.sh
```
Comment out Line 15 ~ line 25 to reduce compiling time afterwards. 

Once built, it will make a folder called `targets`. The executable `orange` is the application. Start the program using the run script. 

```
./run.sh
```

## Use the Application
When first time open the program, `orange` creates folders with the following structure

```
orange_data
├── config
│   ├── local
│   └── network
├── detect
├── exp
│   └── unsorted
└── pictures

```

The are two modes of using the application: local vs network. Local means all cameras are connected to one server, while network can support multiple servers. 

### Local mode
One could save preconfigued camera settings in a folder under `local`, for instance

```
orange_data
├── config
│   ├── local
│   │   ├── 5cam
│   │   │   ├── 2002488.json
│   │   │   ├── 2002489.json
│   │   │   ├── 2002490.json
│   │   │   ├── 2002496.json
│   │   │   └── 710038.json
│   │   └── center_ceiling
│   │       └── 710038.json
│   └── network
├── detect
│   └── rat_bbox.engine
├── exp
│   └── unsorted
│       └── 2024_10_31_13_09_41
│           ├── Cam710038_meta.csv
│           └── Cam710038.mp4
└── pictures
    └── 710038_0.tiff

```
In the node folder (like `5cam` folder), it contains 1 or more camera configs `[camera serial].json`. An example config file is in the `config` folder. Please name the file after the serial number of your cameras and set the config according to your camera specifications. To enable `gpu_direct`, set `gpu_direct` to true, and set the `gpu_id` to select which gpu to use for image processing of the camera. 

### Network mode
One can network multiple PCs to scale up to more cameras. 

The recorded videos are saved at `orange_data/exp/unsorted` by default. But it can be easily changed while using the app. 

## Contribute to the project 
If you wish to contribute to the project, please make changes to your local branch, and create a pull request before pushing.  


