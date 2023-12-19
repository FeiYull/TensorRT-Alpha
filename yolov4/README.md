## 1. get onnx
download directly at [weiyun](https://share.weiyun.com/3T3mZKBm) or [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv)
or export onnx:
```bash
git clone https://github.com/Tianxiaomo/pytorch-YOLOv4
git checkout  a65d219f9066bae4e12003bd7cdc04531860c672
git clone https://github.com/FeiYull/tensorrt-alpha
cd tensorrt-alpha/yolov4
# PLease use the "alpha_export.py" file provided by TensorRT-Alpha to export onnx
cp alpha_export.py Pytorch_YOLOV4/
cd Pytorch_YOLOV4/
# 608
python alpha_export.py cfg/yolov4.cfg yolov4.weights --batch_size=-1 --onnx_file_path=alpha_yolov4_-1_3_608_608_dynamic.onnx
```
## 2.edit and save onnx
```bash
# note: If you have obtained onnx by downloading, this step can be ignored
ignore
```
## 3.compile onnx
```bash
# put your onnx file in this path:tensorrt-alpha/data/yolov4
cd tensorrt-alpha/data/yolov4
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/feiyull/TensorRT-8.4.2.4/lib
# 608
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolov4_-1_3_608_608_dynamic.onnx   --saveEngine=yolov4_-1_3_608_608_dynamic.trt  --buildOnly --minShapes=input:1x3x608x608 --optShapes=input:2x3x608x608 --maxShapes=input:4x3x608x608
```
## 4.run
```bash
git clone https://github.com/FeiYull/tensorrt-alpha
cd tensorrt-alpha/yolov4
mkdir build
cd build
cmake ..
make -j10
# note: the dstImage will be saved in tensorrt-alpha/yolov4/build by default

## 608
# infer image
./app_yolov4  --model=../../data/yolov4/alpha_yolov4_-1_3_608_608_dynamic.trt --size=608  --batch_size=1  --img=../../data/6406402.jpg  --show --savePath=../

# infer video
./app_yolov4  --model=../../data/yolov4/alpha_yolov4_-1_3_608_608_dynamic.trt --size=608 --batch_size=2  --video=../../data/people.mp4  --show 

# infer camera
./app_yolov4  --model=../../data/yolov4/alpha_yolov4_-1_3_608_608_dynamic.trt --size=608 --batch_size=2  --cam_id=0  --show
```
## 5. appendix
ignore