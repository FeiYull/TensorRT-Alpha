## 特别说明
🚀TensorRT-Alpha现在支持yolov5.6.0和最新的yolov5.7.0 <br>
🍅TensorRT-Alpha support yolov5.6.0 and yolov5.7.0<br>
## 1. get onnx
download directly at:[weiyun](https://share.weiyun.com/3T3mZKBm) or [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv?usp=sharing) or export onnx:
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
#---------------
# for yolov5.6.0
#---------------
git checkout  v6.0
# 640
python export.py --weights=yolov5s.pt  --dynamic  
python export.py --weights=yolov5m.pt  --dynamic  
python export.py --weights=yolov5l.pt  --dynamic  
python export.py --weights=yolov5x.pt  --dynamic  
# 1280
python export.py --weights=yolov5s6.pt  --dynamic 
python export.py --weights=yolov5m6.pt  --dynamic 
python export.py --weights=yolov5l6.pt  --dynamic 
#---------------
# for yolov5.7.0
#---------------
git checkout v7.0
# 640
python export.py --weights=yolov5n.pt  --dynamic --include=onnx 
python export.py --weights=yolov5s.pt  --dynamic --include=onnx
python export.py --weights=yolov5m.pt  --dynamic --include=onnx
python export.py --weights=yolov5l.pt  --dynamic --include=onnx
python export.py --weights=yolov5x.pt  --dynamic --include=onnx
# 1280
python export.py --weights=yolov5n6.pt  --dynamic --include=onnx
python export.py --weights=yolov5s6.pt  --dynamic --include=onnx
python export.py --weights=yolov5m6.pt  --dynamic --include=onnx
python export.py --weights=yolov5l6.pt  --dynamic --include=onnx
python export.py --weights=yolov5x6.pt  --dynamic --include=onnx
```
## 2.edit and save onnx
```bash
# note: If you have obtained onnx by downloading, this step can be ignored
#---------------
# for yolov5.6.0
#---------------
git clone https://github.com/FeiYull/tensorrt-alpha
cd tensorrt-alpha/yolov5
conda activate tensorrt-alpha
python alpha_edit.py --mode=p5 --net_name=yolov5s  --model_path=../data/yolov5/yolov5s.onnx # 640
python alpha_edit.py --mode=p6 --net_name=yolov5s6  --model_path=../data/yolov5/yolov5s6.onnx # 1280
#---------------
# for yolov5.7.0
#---------------
ignore,nice!
```
## 3.compile onnx
```bash
# put your onnx file in this path:tensorrt-alpha/data/yolov5
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/feiyull/TensorRT-8.4.2.4/lib
cd tensorrt-alpha/data/yolov5

#---------------
# for yolov5.6.0
#---------------
# 640
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=alpha_yolov5s.onnx   --saveEngine=alpha_yolov5s.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=alpha_yolov5m.onnx   --saveEngine=alpha_yolov5m.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=alpha_yolov5l.onnx   --saveEngine=alpha_yolov5l.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=alpha_yolov5x.onnx   --saveEngine=alpha_yolov5x.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
# 1280
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=alpha_yolov5s6.onnx   --saveEngine=alpha_yolov5s6.trt  --buildOnly --minShapes=images:1x3x1280x1280 --optShapes=images:2x3x1280x1280 --maxShapes=images:4x3x1280x1280

#---------------
# for yolov5.7.0
#---------------
# 640
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolov5n.onnx   --saveEngine=yolov5n.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolov5s.onnx   --saveEngine=yolov5s.trt   --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolov5m.onnx   --saveEngine=yolov5m.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolov5l.onnx   --saveEngine=yolov5l.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolov5x.onnx   --saveEngine=yolov5x.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
# 1280
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolov5n6.onnx   --saveEngine=yolov5n6.trt  --buildOnly --minShapes=images:1x3x1280x1280 --optShapes=images:2x3x1280x1280 --maxShapes=images:4x3x1280x1280
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolov5s6.onnx   --saveEngine=yolov5s6.trt  --buildOnly --minShapes=images:1x3x1280x1280 --optShapes=images:2x3x1280x1280 --maxShapes=images:4x3x1280x1280
```
## 4.run
```bash
git clone https://github.com/FeiYull/tensorrt-alpha
cd tensorrt-alpha/yolov5
mkdir build
cd build
cmake ..
make -j10
# note: the dstImage will be saved in tensorrt-alpha/yolov5/build by default

#---------------
# yolov5.6.0
#---------------
# 640
# infer an image
./app_yolov5  --version=v560 --model=../../data/yolov5/alpha_yolov5s.trt   --size=640  --batch_size=1 --img=../../data/6406401.jpg   --show --savePath
# infer video
./app_yolov5  --version=v560 --model=../../data/yolov5/alpha_yolov5s.trt   --size=640  --batch_size=2 --video=../../data/people.mp4  --show --savePath=../
# infer web camera
./app_yolov5  --version=v560 --model=../../data/yolov5/alpha_yolov5s.trt   --size=640  --batch_size=2 --cam_id=0                     --show --savePath

# 1280
./app_yolov5  --version=v560 --model=../../data/yolov5/alpha_yolov5m6.trt  --size=1280 --batch_size=1 --img=../../data/6406401.jpg   --show --savePath

#---------------
# yolov5.7.0
#---------------
# 640
# infer an image
./app_yolov5  --version=v570 --model=../../data/yolov5/yolov5n.trt   --size=640  --batch_size=1  --img=../../data/6406401.jpg   --show --savePath=../
# infer video
./app_yolov5  --version=v570 --model=../../data/yolov5/yolov5n.trt   --size=640  --batch_size=2  --video=../../data/people.mp4  --show 
# infer web camera
./app_yolov5  --version=v570 --model=../../data/yolov5/yolov5n.trt   --size=640  --batch_size=2  --show  --cam_id=0

# 1280
./app_yolov5  --version=v570 --model=../../data/yolov5/yolov5s6.trt  --size=1280 --batch_size=1 --img=../../data/6406401.jpg   --show --savePath

```
## 5. appendix
offical weights for yolov5.6.0<br>
[yolov5s]   |   https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt<br>
[yolov5m]   |   https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5m.pt<br>
[yolov5l]   |   https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5l.pt<br>
[yolov5x]   |   https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5x.pt<br>
[yolov5s6]  |   https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s6.pt<br>
[yolov5m6]  |   https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5m6.pt<br>
[yolov5l6]  |   https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5l6.pt<br>
[yolov5x6]  |   https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5x6.pt<br>

offical weights for yolov5.7.0<br>
[yolov5s]   |   https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt<br>
[yolov5m]   |   https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt<br>
[yolov5l]   |   https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt<br>
[yolov5x]   |   https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt<br>
[yolov5s6]   |   https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s6.pt<br>
[yolov5m6]   |   https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m6.pt<br>
[yolov5l6]   |   https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l6.pt<br>
[yolov5x6]   |   https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x6.pt<br>