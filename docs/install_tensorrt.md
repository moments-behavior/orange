- [tensor-rt (depends on nvidia-driver and CUDA)](#install-tensor-rt)

### install tensor-rt
this is based on [these instructions](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar) (has more details if needed)

**0. download and extract tensor-rt installation file**
  - we use `TensorRT-8.6.1.6` with `cuda 12.0`  -- you can directly download this (or from this page). But if you are using `cuda 12.2` and above, please use TensorRT 10, for instance `TensorRT-10.6.0.26.Linux.x86_64-gnu.cuda-12.6`. The installation steps are similar.
    ```
    cd /home/$USER/nvidia
    wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz
    tar -xzvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz
    ```
  - this should extract a folder `TensorRT-8.6.1.6` with following subdirectories
    ```
    bin  data  doc  include  lib  python  samples  targets
    ```
  - rename the folder `TensorRT`.

**1. add tensor-rt path in bashrc**
  - Add the absolute path to the TensorRT lib directory to the environment variable `LD_LIBRARY_PATH`:
    ```
    export LD_LIBRARY_PATH=/home/$USER/nvidia/TensorRT/lib:$LD_LIBRARY_PATH 
    source ~/.bashrc
    ```
**2. verify installation**
  - try to build one of the sample programs (say, `trtexec`) to verify installation
    ```
    cd /home/$USER/nvidia/TensorRT/samples/trtexec
    make
    ```
  - run the program built above
    ```
    cd home/$USER/nvidia/TensorRT/bin/
    ./trtexec
    ```

