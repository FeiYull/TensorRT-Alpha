## 0. install eigen
eigen3.4.0 has been tested and passed!
```bash
# for linux
sudo apt-get install libeigen3-dev

# for windows
# download from https://eigen.tuxfamily.org/index.php?title=Main_Page
# decompressing the package
# Just manually add the include directory in the vs project
```

## 1. get onnx 
download directly at [weiyun](https://share.weiyun.com/3T3mZKBm) or [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv) or export onnx:
```bash
# 🔥 yolov8 offical repo: https://github.com/ultralytics/ultralytics
# 🔥 yolov8 quickstart: https://docs.ultralytics.com/quickstart/
# 🚀TensorRT-Alpha will be updated synchronously as soon as possible!

# install yolov8
conda create -n yolov8 python==3.8 -y # for Linux
# conda create -n yolov8 python=3.9 -y # for Windows10
conda activate yolov8
pip install ultralytics==8.0.200
pip install onnx==1.12.0

# download offical weights(".pt" file)
https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt
https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt
https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt
https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt
https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt
```

export onnx:
```bash
yolo mode=export model=yolov8n-seg.pt format=onnx dynamic=True opset=12
yolo mode=export model=yolov8s-seg.pt format=onnx dynamic=True opset=12    
yolo mode=export model=yolov8m-seg.pt format=onnx dynamic=True opset=12    
yolo mode=export model=yolov8l-seg.pt format=onnx dynamic=True opset=12    
yolo mode=export model=yolov8x-seg.pt format=onnx dynamic=True opset=12
```

## 2.edit and save onnx
```bash
# note: If you have obtained onnx by downloading, this step can be ignored
ignore
```

## 3.compile onnx
```bash
# put your onnx file in this path:tensorrt-alpha/data/yolov8-seg
cd tensorrt-alpha/data/yolov8-seg
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/feiyull/TensorRT-8.4.2.4/lib
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolov8n-seg.onnx  --saveEngine=yolov8n-seg.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolov8s-seg.onnx  --saveEngine=yolov8s-seg.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolov8m-seg.onnx  --saveEngine=yolov8m-seg.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolov8l-seg.onnx  --saveEngine=yolov8l-seg.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolov8x-seg.onnx  --saveEngine=yolov8x-seg.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
```
## 4.run
```bash
git clone https://github.com/FeiYull/tensorrt-alpha
cd tensorrt-alpha/yolov8-seg
mkdir build
cd build
cmake ..
make -j10
# note: the dstImage will be saved in tensorrt-alpha/yolov8-seg/build by default

## 640
# infer image
./app_yolov8_seg  --model=../../data/yolov8/yolov8n-seg.trt --size=640 --batch_size=1  --img=../../data/6406407.jpg   --show --savePath=../

# infer video
./app_yolov8_seg  --model=../../data/yolov8/yolov8n-seg.trt     --size=640 --batch_size=1  --video=../../data/people.mp4  --show 

# infer camera
./app_yolov8_seg  --model=../../data/yolov8/yolov8n-seg.trt     --size=640 --batch_size=1  --cam_id=0  --show

```
## 5. appendix
ignore