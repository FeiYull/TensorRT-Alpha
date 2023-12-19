## 1. get onnx
download directly at [weiyun](https://share.weiyun.com/3T3mZKBm) or [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv)

or export onnx:
```bash
# Please refer to following site, it is tensorrt's offical doc, and it lead you to export onnx from efficientdet's offical weights.
# TensorRT-Alpha converts python to cuda c.
https://github.com/NVIDIA/TensorRT/blob/release/8.4/samples/python/efficientdet/README.md
```
## 2.edit and save onnx
```bash
# note: If you have obtained onnx by downloading, this step can be ignored
ignored
```
## 3.compile onnx 
```bash
# put your onnx file in this path:tensorrt-alpha/data/efficientdet
cd tensorrt-alpha/data/efficientdet
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/feiyull/TensorRT-8.4.2.4/lib
../../../../TensorRT-8.4.2.4/bin/trtexec --onnx=efficientdet0.onnx   --saveEngine=efficientdet0.trt   --buildOnly --minShapes=input:1x512x512x3 --optShapes=input:2x512x512x3 --maxShapes=input:4x512x512x3
../../../../TensorRT-8.4.2.4/bin/trtexec --onnx=efficientdet1.onnx   --saveEngine=efficientdet1.trt   --buildOnly --minShapes=input:1x640x640x3 --optShapes=input:2x640x640x3 --maxShapes=input:4x640x640x3
../../../../TensorRT-8.4.2.4/bin/trtexec --onnx=efficientdet2.onnx   --saveEngine=efficientdet2.trt   --buildOnly --minShapes=input:1x768x768x3 --optShapes=input:2x768x768x3 --maxShapes=input:4x768x768x3
../../../../TensorRT-8.4.2.4/bin/trtexec --onnx=efficientdet3.onnx   --saveEngine=efficientdet3.trt   --buildOnly --minShapes=input:1x896x896x3 --optShapes=input:2x896x896x3 --maxShapes=input:4x896x896x3

```
## 4.run
```bash
git clone https://github.com/FeiYull/tensorrt-alpha
cd tensorrt-alpha/efficientdet
mkdir build
cd build
cmake ..
make -j10
# note: the dstImage will be saved in tensorrt-alpha/efficientdet/build by default

# infer image
./app_efficientdet  --model=../../data/efficientdet/efficientdet0.trt --img=../../data/road0.png  --size=512 --batch_size=1 --show --savePath
./app_efficientdet  --model=../../data/efficientdet/efficientdet1.trt --img=../../data/road0.png  --size=640 --batch_size=1 --show --savePath
./app_efficientdet  --model=../../data/efficientdet/efficientdet2.trt --img=../../data/road0.png  --size=768 --batch_size=1 --show --savePath
./app_efficientdet  --model=../../data/efficientdet/efficientdet3.trt --img=../../data/road0.png  --size=896 --batch_size=1 --show --savePath


# infer video
./app_efficientdet  --model=../../data/efficientdet/efficientdet0.trt  --size=512 --batch_size=2  --video=../../data/people.mp4  --show

# infer camera
./app_efficientdet  --model=../../data/efficientdet/efficientdet0.trt  --size=512 --batch_size=2  --cam_id=0  --show
```
## 5. appendix
ignore