## 1. Install Tool Chains 
```bash
sudo apt-get update 
sudo apt-get install build-essential 
sudo apt-get install git
sudo apt-get install gdb
sudo apt-get install cmake
```
```bash
sudo apt-get install pkg-config libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev 
sudo apt-get install libopencv-dev  
# pkg-config --modversion opencv
```
## 2. Install Nvidia Libs
### 2.1 install nvidia driver470
```bash
ubuntu-drivers devices
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-driver-470-server # for ubuntu18.04
nvidia-smi
```
### 2.2 install cuda11.3
- enter: https://developer.nvidia.com/cuda-toolkit-archive
- select：CUDA Toolkit 11.3.0(April 2021)
- select：[Linux] -> [x86_64] -> [Ubuntu] -> [18.04] -> [runfile(local)]<br>
You will see installation instructions on the web page like this：
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
sudo sh cuda_11.3.0_465.19.01_linux.run
```
The cuda installation process will have a window display.
- select：[continue] -> [accept] -> Press enter to cancel the first and second options like the following(<font color=#FFFF00 >**it is important!**</font>) -> [Install]<br>

```bash
CUDA Installer
[ ] Driver        # cancel the first
    [ ] 465.19.01 # cancel the second
[X] CUDA Toolkit 11.3 
[X] CUDA Samples 11.3 
[X] CUDA Demo Suite 11.3 
[X] CUDA Documentation 11.3 0tions 
```

The bash window prints the following， which means the installation is OK.
```bash
#===========
#= Summary =
#===========

#Driver:   Not Selected
#Toolkit:  Installed in /usr/local/cuda-11.3/
#......
```
add environment variables：
```bash
vim ~/.bashrc
```
Copy and paste the following into .bashrc
```bash
# cuda v11.3
export PATH=/usr/local/cuda-11.3/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-11.3
```
```bash
source ~/.bashrc
nvcc -V
```
The bash window prints the following content：<br>
<br>
nvcc: NVIDIA (R) Cuda compiler driver<br>
Copyright (c) 2005-2021 NVIDIA Corporation<br>
Built on Sun_Mar_21_19:15:46_PDT_2021<br>
Cuda compilation tools, release 11.3, V11.3.58<br>
Build cuda_11.3.r11.3/compiler.29745058_0<br>
<br>

### 2.3 install cudnn8.2
- enter：https://developer.nvidia.com/rdp/cudnn-archive
- select: Download cuDNN v8.2.0 (April 23rd, 2021), for CUDA 11.x
- select： cuDNN Library for Linux (x86_64)
- you will download file:  "cudnn-11.3-linux-x64-v8.2.0.53.tgz"
```bash
tar -zxvf cudnn-11.3-linux-x64-v8.2.0.53.tgz
```
copy cudnn  to cuda11.3's install dir
```bash
sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn.h
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
```
### 2.4 download tensorrt8.4.2.4
- enter： https://developer.nvidia.cn/nvidia-tensorrt-8x-download
- select： I Agree To the Terms of the NVIDIA TensorRT License Agreement
- select:   TensorRT 8.4 GA Update 1
- select:   TensorRT 8.4 GA Update 1 for Linux x86_64 and CUDA 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6 and 11.7 TAR Package
- you will download file:  "TensorRT-8.4.2.4.Linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz"
```bash
tar -zxvf TensorRT-8.4.2.4.Linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz
# test
cd TensorRT-8.4.2.4/samples/sampleMNIST
make
cd ../../bin/
```
Change the following path to your path!(<font color=#FFFF00 >**it is important!**</font>)
```bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/xxx/temp/TensorRT-8.4.2.4/lib
./sample_mnist
```
The bash window prints digit recognition task information， which indicats tensorrt8.4.2.4 is installed normally.
