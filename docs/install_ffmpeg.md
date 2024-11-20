## Install ffmpeg with gpu encoder support 
Make sure you have nvidia driver and cuda toolkit installed 

1. To compile ffmpeg with NVIDIA we need ffnvcodec too. Clone git repo:
```
mkdir ~/nvidia/ && cd ~/nvidia/  
git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
```

2. Install ffnvcodec on Ubuntu:
```
cd nv-codec-headers && sudo make install
```

3. Get ffmpeg source code, run: 
```
cd ~/nvidia/  
git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg/
```

4. Install GNU gcc compiler collection and libs, run:
```
sudo apt install build-essential yasm cmake libtool libc6 libc6-dev unzip wget libnuma1 libnuma-dev
```

Note, you may also need to install 
```
sudo apt-get install git make nasm pkg-config libx264-dev libxext-dev libxfixes-dev zlib1g-dev
```


### Install ffmpeg release 4.4.1 

This is for using ffmpeg API

```
cd ~/nvidia/ffmpeg
git checkout release/4.4
```

Rebuild the ffmpeg following above instruction. If you have trouble compiling it, it is likely that the gpu architecture is too low. Check config.log


### Build ffmpeg 4.4.1

Legacy 
```
./configure --prefix=$(pwd)/build --disable-static --enable-shared --enable-nonfree --enable-cuda-nvcc --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64 --nvccflags="-gencode arch=compute_75,code=sm_75 -O2"
```

or 

```
./configure --prefix=$(pwd)/build --disable-static --enable-shared --enable-nonfree --enable-cuda-nvcc --enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64 --nvccflags="-gencode arch=compute_90,code=sm_90 -O2"
```


If you followed the above steps, FFmpeg should be installed at `$HOME/nvidia/ffmpeg`
