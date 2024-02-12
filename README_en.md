# TensorRT-Alpha

<div align="center">

  [![Cuda](https://img.shields.io/badge/CUDA-11.3-%2376B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit-archive)
  [![](https://img.shields.io/badge/TensorRT-8.4.2.4-%2376B900.svg?style=flat&logo=tensorrt)](https://developer.nvidia.com/nvidia-tensorrt-8x-download)
  [![](https://img.shields.io/badge/ubuntu-18.04-orange.svg?style=flat&logo=ubuntu)](https://releases.ubuntu.com/18.04/)
  [![](https://img.shields.io/badge/windows-10-blue.svg?style=flat&logo=windows)](https://www.microsoft.com/)

  English | [简体中文](README.md)<br>



  <br>
  </div>

## Visualization
<div align='center'>
  <img src='.github/facemesh.jpg' width="143px">
  <img src='.github/people.gif' width="338px">
  <img src='.github/yolov8-snow.gif' height="190px" width="230px">
  <br>
  <img src='.github/yolov8-stree.gif'  width="260px">
  <img src='.github/u2net.gif'  width="190px">
  <img src='.github/libfacedet.gif'  width="260px">
  <br>
</div> 

## Introduce
This repository  provides accelerated deployment cases of deep learning CV popular models, and cuda c supports dynamic-batch image process, infer, decode, NMS.<br>
There are two ways to compile model(pth or onnx):<br>

<details>
<summary>pth -> trt</summary>
coming soon.
</details>

<details>
<summary>pth -> onnx -> trt:</summary>

  - [i]. According to the network disk provided by TensorRT-Alpha, download ONNX directly. [weiyun](https://share.weiyun.com/3T3mZKBm) or [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv?usp=sharing)
  - [ii]. Follow the instructions provided by TensorRT-Alpha to manually export ONNX from the relevant python source code framework.

</details>

## Update
- 2023.01.01  🔥 update yolov3, yolov4, yolov5, yolov6
- 2023.01.04  🍅 update yolov7, yolox, yolor
- 2023.01.05  🎉 update u2net, libfacedetection
- 2023.01.08  🚀 The whole network is the first to support yolov8 
- 2023.01.20  🍏 update efficientdet, pphunmanseg
- 2023.12.09  🍁 update yolov8-pose
- 2023.12.19  🍉 update yolov8-seg
- 2023.12.27  💖 update yolonas

## Installation
The following environments have been tested：<br>
<details>
<summary>Ubuntu18.04</summary>

- cuda11.3
- cudnn8.2.0
- gcc7.5.0
- tensorrt8.4.2.4
- opencv3.x or 4.x
- cmake3.10.2
</details>

<details>
<summary>Windows10</summary>

- cuda11.3 
- cudnn8.2.0
- visual studio 2017 or 2019 or 2022
- tensorrt8.4.2.4
- opencv3.x or 4.x
</details>

<details>
<summary>Python environment(Optional）</summary>

```bash
# install miniconda first
conda create -n tensorrt-alpha python==3.8 -y
conda activate tensorrt-alpha
git clone https://github.com/FeiYull/tensorrt-alpha
cd tensorrt-alpha
pip install -r requirements.txt  
```
</details>

Installation Tutorial：
- [Install For Ubuntu18.04](Install_For_Ubuntu18.04/Install_For_Ubuntu18.04.md)<br>
- [Docker For Linux](docker/README.md)<br>

## Quick Start
### Ubuntu18.04
set your TensorRT_ROOT path:
```bash
git clone https://github.com/FeiYull/tensorrt-alpha
cd tensorrt-alpha/cmake
vim common.cmake
# set var TensorRT_ROOT to your path in line 20, eg:
# set(TensorRT_ROOT /home/feiyull/TensorRT-8.4.2.4)
```
start to build project:
For example:[yolov8](yolov8/README.md)

## Onnx
At present, more than 30  models have been implemented, and some onnx files of them are organized as follows:

<div align='center'>

| model | tesla v100(32G) |weiyun |google driver |
  :-: | :-: | :-: | :-: |
|[yolov3](yolov3/README.md)| |[weiyun](https://share.weiyun.com/3T3mZKBm)| [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv?usp=sharing)|      
|[yolov4](yolov4/README.md)| |[weiyun](https://share.weiyun.com/3T3mZKBm)| [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv?usp=sharing)|
|[yolov5](yolov5/README.md)| |[weiyun](https://share.weiyun.com/3T3mZKBm)| [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv?usp=sharing)|     
|[yolov6](yolov6/README.md)| |[weiyun](https://share.weiyun.com/3T3mZKBm)| [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv?usp=sharing)|     
|[yolov7](yolov7/README.md)| |[weiyun](https://share.weiyun.com/3T3mZKBm)| [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv?usp=sharing)|     
|[yolov8](yolov8/README.md)| |[weiyun](https://share.weiyun.com/3T3mZKBm)| [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv?usp=sharing)|     
|[yolox](yolox/README.md)| |[weiyun](https://share.weiyun.com/3T3mZKBm)| [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv?usp=sharing)|     
|[yolor](yolor/README.md)| |[weiyun](https://share.weiyun.com/3T3mZKBm)| [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv?usp=sharing)|     
|[u2net](u2net/README.md)| |[weiyun](https://share.weiyun.com/3T3mZKBm)| [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv?usp=sharing)|     
|[libfacedetection](libfacedetection/README.md)| |[weiyun](https://share.weiyun.com/3T3mZKBm)| [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv?usp=sharing)|     
|[facemesh](facemesh/README.md)| |[weiyun](https://share.weiyun.com/3T3mZKBm)| [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv?usp=sharing)|  
|[pphumanseg](pphumanseg/README.md)| |[weiyun](https://share.weiyun.com/3T3mZKBm)| [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv?usp=sharing)|  
|[efficientdet](efficientdet/README.md)| |[weiyun](https://share.weiyun.com/3T3mZKBm)| [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv?usp=sharing)|     
|[yolov8-pose](yolov8-pose/README.md)| |[weiyun](https://share.weiyun.com/3T3mZKBm)| [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv?usp=sharing)| 
|[yolov8-seg](yolov8-seg/README.md)| |[weiyun](https://share.weiyun.com/3T3mZKBm)| [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv?usp=sharing)| 
|[yolonas](yolonas/README.md)| |[weiyun](https://share.weiyun.com/3T3mZKBm)| [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv?usp=sharing)| 
|more...(🚀: I will be back soon!)    |      |          |
</div>  

🍉We will test the time of all models on tesla v100 and A100! Now let's preview the performance of yolov8n on RTX2070m(8G)：
<div align='center'>

| model | video resolution | model input size |GPU Memory-Usage |GPU-Util|
  :-: | :-: | :-: | :-: | :-: |
|yolov8n|1920x1080|8x3x640x640|1093MiB/7982MiB| 14%| 

 <center>	<!--将图片和文字居中-->
<img src=".github/yolov8n-b8-1080p-to-640.jpg"
     alt="无法显示图片时显示的文字"
     style="zoom:40%"/>
<br>		<!--换行-->
<center>cost time per frame	<!--标题--></center>
    <br>		<!--换行-->

</div>
<br>

## Some Precision Alignment Renderings Comparison
<br>
<div align='center'>			<!--块级封装-->
     <center>	<!--将图片和文字居中-->
    <img src=".github/yolov8n-Offical(left)vsOurs(right).jpg"
         alt="无法显示图片时显示的文字"
         style="zoom:80%"/>
    <br>		<!--换行-->
    <center>yolov8n : Offical( left ) vs Ours( right )	<!--标题--></center>
    <br>		<!--换行-->
    <br>		<!--换行-->
    <center>	<!--将图片和文字居中-->
    <img src=".github/yolov7-tiny-Offical(left)vsOurs(right).jpg"
         alt="无法显示图片时显示的文字"
         style="zoom:80%"/>
    <br>		<!--换行-->
    <center>yolov7-tiny : Offical( left ) vs Ours( right )	<!--标题--></center>
    <br>		<!--换行-->
    <br>		<!--换行-->
    <img src=".github/yolov6s-v6.3-Offical(left)vsOurs(right).jpg"
         alt="无法显示图片时显示的文字"
         style="zoom:80%"/>
    <br>		<!--换行-->
    <center>yolov6s : Offical( left ) vs Ours( right )	<!--标题--></center>
    <br>		<!--换行-->
    <br>		<!--换行-->
    <img src=".github/yolov5s-v5.7-Offical(left)vsOurs(right)-img2.jpg"
         alt="无法显示图片时显示的文字"
         style="zoom:80%"/>
    <br>		<!--换行-->
    <center>yolov5s : Offical( left ) vs Ours( right )	<!--标题--></center>
    <br>		<!--换行-->
    <br>		<!--换行-->
    <img src=".github/yolov5s-v5.7-Offical(left)vsOurs(right)-img1.jpg"
         alt="无法显示图片时显示的文字"
         style="zoom:80%"/>
    <br>		<!--换行-->
    <center>yolov5s : Offical( left ) vs Ours( right )	<!--标题--></center>
    <br>		<!--换行-->
    <br>		<!--换行-->
    <img src=".github/libfacedet-Offical(left)vsOurs(right-topk-2000).jpg"
         alt="无法显示图片时显示的文字"
         style="zoom:100%"/>
    <br>		<!--换行-->
    <center>libfacedetection : Offical( left ) vs Ours( right topK:2000)	<!--标题--></center>
    <br>		<!--换行-->
    <br>		<!--换行-->
    </center>
</div>


## Citation
```bash
@misc{FeiYull_TensorRT-Alpha,  
  author = {FeiYull},  
  title = {TensorRT-Alpha},  
  year = {2023},  
  publisher = {GitHub},  
  journal = {GitHub repository},  
  howpublished = {https://github.com/FeiYull/tensorrt-alpha}
}
```