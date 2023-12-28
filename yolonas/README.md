## 1. get onnx 
download directly at [weiyun](https://share.weiyun.com/3T3mZKBm) or [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv) or export onnx:
```bash
pip install super-gradients==3.3.1
cd super-gradients
# copy the python script provided in this repository to your workspace
# note:The weight file is downloaded automatically
cp TensorRT-Alpha/yolonas/alpha_export_dynamic.py YOUR_WORKSPACE

# for YOLO_NAS_S
# Changing lines 9-11 of the code allows you to switch to other models, eg:YOLO_NAS_M
python alpha_export_dynamic.py
```

## 2.edit and save onnx
```bash
# note: If you have obtained onnx by downloading, this step can be ignored
ignore
```

## 3.compile onnx
```bash
# put your onnx file in this path:tensorrt-alpha/data/yolonas
cd tensorrt-alpha/data/yolonas
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/feiyull/TensorRT-8.4.2.4/lib
# 640
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolonas_s.onnx  --saveEngine=yolonas_s.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
```
## 4.run
```bash
git clone https://github.com/FeiYull/tensorrt-alpha
cd tensorrt-alpha/yolonas
mkdir build
cd build
cmake ..
make -j10
# note: the dstImage will be saved in tensorrt-alpha/yolonas/build by default

## 640
# infer image
./app_yolo_nas  --model=../../data/yolo_nas/yolonas_s.trt --size=640 --batch_size=1  --img=../../data/6406407.jpg   --show --savePath=../

# infer video
./app_yolo_nas  --model=../../data/yolo_nas/yolonas_s.trt --size=640 --batch_size=2  --video=../../data/people.mp4  --show 

# infer camera
./app_yolo_nas  --model=../../data/yolo_nas/yolonas_s.trt --size=640 --batch_size=2  --cam_id=0  --show
```
## 5. appendix
ignore