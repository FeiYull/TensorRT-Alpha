## 1. get onnx
download directly at:[weiyun](https://share.weiyun.com/3T3mZKBm) or [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv?usp=sharing) or export onnx:
```bash
git clone https://github.com/ultralytics/yolov5
git checkout  v6.0
python export.py --weights=yolov5s.pt  --dynamic  # 640
python export.py --weights=yolov5s6.pt  --dynamic # 1280
```
## 2.edit and save onnx
```bash
# note: If you have obtained onnx by downloading, this step can be ignored
git clone https://github.com/FeiYull/tensorrt-alpha
cd tensorrt-alpha/yolov5
conda activate tensorrt-alpha
python yolov5.py --mode=p5 --net_name=yolov5s  --model_path=../data/yolov5/yolov5s.onnx # 640
python yolov5.py --mode=p6 --net_name=yolov5s6  --model_path=../data/yolov5/yolov5s6.onnx # 1280
```
## 3.compile onnx
```bash
# put your onnx file in this path:tensorrt-alpha/data/yolov5
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/TensorRT-8.4.2.4/lib
cd tensorrt-alpha/data/yolov5
# 640
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=alpha_yolov5s.onnx   --saveEngine=alpha_yolov5s.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640
# 1280
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=alpha_yolov5s6.onnx   --saveEngine=alpha_yolov5s6.trt  --buildOnly --minShapes=images:1x3x1280x1280 --optShapes=images:8x3x1280x1280 --maxShapes=images:8x3x1280x1280
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

# 640
# infer an image
./app_yolov5  --model=../../data/yolov5/alpha_yolov5s.trt   --size=640  --batch_size=1 --img=../../data/6406401.jpg   --show --savePath
# infer video
./app_yolov5  --model=../../data/yolov5/alpha_yolov5s.trt   --size=640  --batch_size=8 --video=../../data/people.mp4  --show --savePath
# infer web camera
./app_yolov5  --model=../../data/yolov5/alpha_yolov5s.trt   --size=640  --batch_size=4 --cam_id=0                     --show --savePath

# 1280
./app_yolov5  --model=../../data/yolov5/alpha_yolov5m6.trt  --size=1280 --batch_size=1 --img=../../data/6406401.jpg   --show --savePath
```
## 5. appendix
[yolov5s]   |   https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt<br>
[yolov5m]   |   https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5m.pt<br>
[yolov5l]   |   https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5l.pt<br>
[yolov5x]   |   https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5x.pt<br>
[yolov5s6]  |   https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s6.pt<br>
[yolov5m6]  |   https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5m6.pt<br>
[yolov5l6]  |   https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5l6.pt<br>
[yolov5x6]  |   https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5x6.pt<br>
