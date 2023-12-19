## 1. get onnx 

download directly at [weiyun]:[weiyun](https://share.weiyun.com/3T3mZKBm) or [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv?usp=sharing) or export onnx:
```bash
git clone https://github.com/WongKinYiu/yolov7
git checkout  072f76c72c641c7a1ee482e39f604f6f8ef7ee92
# 640
python export.py --weights yolov7-tiny.pt  --dynamic  --grid
python export.py --weights yolov7.pt  --dynamic  --grid
python export.py --weights yolov7x.pt  --dynamic  --grid
# 1280
python export.py --weights yolov7-w6.pt  --dynamic  --grid --img-size 1280
```
## 2.edit and save onnx 
```bash
# note: If you have obtained onnx by downloading, this step can be ignored
ignore
```

## 3.compile onnx
```bash
# put your onnx file in this path:tensorrt-alpha/data/yolov7
cd tensorrt-alpha/data/yolov7
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/feiyull/TensorRT-8.4.2.4/lib
# 640
../../../../TensorRT-8.4.2.4/bin/trtexec    --onnx=yolov7-tiny.onnx  --saveEngine=yolov7-tiny.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
../../../../TensorRT-8.4.2.4/bin/trtexec    --onnx=yolov7.onnx   	--saveEngine=yolov7.trt       --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
../../../../TensorRT-8.4.2.4/bin/trtexec    --onnx=yolov7x.onnx   	--saveEngine=yolov7x.trt      --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
# 1280
../../../../TensorRT-8.4.2.4/bin/trtexec    --onnx=yolov7-w6.onnx    --saveEngine=yolov7-w6.trt    --buildOnly --minShapes=images:1x3x1280x1280 --optShapes=images:2x3x1280x1280 --maxShapes=images:4x3x1280x1280

# note:if report an error（Error Code 1: Cuda Runtime (an illegal memory access was encountered "bool context = m_context->executeV2((void**)bindings)" returns false） 
when running the model(yolov7-w6), just lower the batch_size.
```
## 4.run
```bash
git clone https://github.com/FeiYull/tensorrt-alpha
cd tensorrt-alpha/yolov7
mkdir build
cd build
cmake ..
make -j10
# note: the dstImage will be saved in tensorrt-alpha/yolov7/build by default

## 640
# infer image
./app_yolov7  --model=../../data/yolov7/yolov7-tiny.trt --size=640  --batch_size=1  --img=../../data/6406401.jpg  --show --savePath
./app_yolov7  --model=../../data/yolov7/yolov7-w6.trt   --size=1280 --batch_size=1  --img=../../data/6406401.jpg  --show --savePath=../

# infer video
./app_yolov7  --model=../../data/yolov7/yolov7-tiny.trt     --size=640 --batch_size=2  --video=../../data/people.mp4  --show 

# infer camera
./app_yolov7  --model=../../data/yolov7/yolov7-tiny.trt     --size=640 --batch_size=2  --cam_id=0  --show
```
## 5. appendix
ignore