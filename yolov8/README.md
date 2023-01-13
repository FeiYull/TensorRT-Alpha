## 1. get onnx 
download directly at [weiyun](https://share.weiyun.com/3T3mZKBm) or [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv) or export onnx:
```bash
# 🔥 yolov8 offical repo: https://github.com/ultralytics/ultralytics
# 🔥 yolov8 quickstart: https://docs.ultralytics.com/quickstart/
# 🚀TensorRT-Alpha will be updated synchronously as soon as possible!

# install yolov8
conda create -n yolov8 python==3.8 -y
conda activate yolov8
pip install ultralytics==8.0.5

# download offical weights(".pth" file)
https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt
https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt
```

export onnx:
```bash
yolo mode=export model=yolov8n.pt format=onnx dynamic=True    #simplify=True
yolo mode=export model=yolov8s.pt format=onnx dynamic=True    #simplify=True
yolo mode=export model=yolov8m.pt format=onnx dynamic=True    #simplify=True
yolo mode=export model=yolov8l.pt format=onnx dynamic=True    #simplify=True
yolo mode=export model=yolov8x.pt format=onnx dynamic=True    #simplify=True
```

## 2.edit and save onnx
```bash
# note: If you have obtained onnx by downloading, this step can be ignored
ignore
```

## 3.compile onnx
```bash
# put your onnx file in this path:tensorrt-alpha/data/yolov8
cd tensorrt-alpha/data/yolov8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/TensorRT-8.4.2.4/lib
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolov8n.onnx  --saveEngine=yolov8n.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolov8s.onnx  --saveEngine=yolov8s.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolov8m.onnx  --saveEngine=yolov8m.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolov8l.onnx  --saveEngine=yolov8l.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolov8x.onnx  --saveEngine=yolov8x.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640
```
## 4.run
```bash
git clone https://github.com/FeiYull/tensorrt-alpha
cd tensorrt-alpha/yolov8
mkdir build
cd build
cmake ..
make -j10
# note: the dstImage will be saved in tensorrt-alpha/yolov8/build by default

## 640
# infer image
./app_yolov8  --model=../../data/yolov8/yolov8n.trt --size=640 --batch_size=1  --img=../../data/6406407.jpg   --show --savePath
./app_yolov8  --model=../../data/yolov8/yolov8n.trt --size=640 --batch_size=8  --video=../../data/people.mp4  --show --savePath

# infer video
./app_yolov8  --model=../../data/yolov8/yolov8n.trt     --size=640 --batch_size=8  --video=../../data/people.mp4  --show --savePath=../

# infer camera
./app_yolov8  --model=../../data/yolov8/yolov8n.trt     --size=640 --batch_size=2  --cam_id=0  --show
```
## 5. appendix