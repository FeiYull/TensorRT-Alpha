## 说明
- 0、请使用本仓库提供的导出脚本“alpha_export.py：
- 1、使用torch1.7+onnx1.8.0时候，导出onnx的时候会报错：
“RuntimeError: Exporting the operator silu to ONNX opset version 11 is not supported. Please open a bug to request ONNX export support for the missing operator.”
- 2、将环境改为：torch1.9+onnx1.11.0，上述不支持的op问题就解决了导出onnx问题。


## 1. get onnx 
download directly at [weiyun](https://share.weiyun.com/3T3mZKBm) or [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv)

or export onnx:
```bash
git clone https://github.com/WongKinYiu/yolor
git checkout  462858e8737f56388f812cfe381a69c4ffca0cc7
# PLease use the "alpha_export.py" file provided by TensorRT-Alpha to export onnx
cd yolor-main
cp  alpha_export.py yolor-main

# 1280
python alpha_export.py --net=yolor_p6
# 640
python alpha_export.py --net=yolor_csp
python alpha_export.py --net=yolor_csp_star
python alpha_export.py --net=yolor_csp_x
python alpha_export.py --net=yolor_csp_x_star
```
## 2.edit and save onnx 
```bash
# note: If you have obtained onnx by downloading, this step can be ignored
ignore
```
## 3.compile onnx
```bash
# put your onnx file in this path:tensorrt-alpha/data/yolor
cd tensorrt-alpha/data/yolor
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/feiyull/TensorRT-8.4.2.4/lib

#1280
../../../../TensorRT-8.4.2.4/bin/trtexec  --onnx=yolor_p6.onnx   --saveEngine=yolor_p6.trt  --buildOnly   --minShapes=images:1x3x1280x1280 --optShapes=images:2x3x1280x1280 --maxShapes=images:4x3x1280x1280

# 640
../../../../TensorRT-8.4.2.4/bin/trtexec  --onnx=yolor_csp.onnx          --saveEngine=yolor_csp.trt          --buildOnly   --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
../../../../TensorRT-8.4.2.4/bin/trtexec  --onnx=yolor_csp_star.onnx     --saveEngine=yolor_csp_star.trt     --buildOnly   --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
../../../../TensorRT-8.4.2.4/bin/trtexec  --onnx=yolor_csp_x.onnx        --saveEngine=yolor_csp_x.trt        --buildOnly   --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
../../../../TensorRT-8.4.2.4/bin/trtexec  --onnx=yolor_csp_x_star.onnx   --saveEngine=yolor_csp_x_star.trt   --buildOnly   --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
```
## 4.run
```bash
git clone https://github.com/FeiYull/tensorrt-alpha
cd tensorrt-alpha/yolor
mkdir build
cd build
cmake ..
make -j10
# note: the dstImage will be saved in tensorrt-alpha/yolor/build by default

## 640
# infer image
./app_yolor  --model=../../data/yolor/yolor_csp.trt --size=640  --batch_size=1  --img=../../data/6406401.jpg  --show --savePath=../

# infer video
./app_yolor  --model=../../data/yolor/yolor_csp.trt --size=640 --batch_size=2  --video=../../data/people.mp4  --show 

# infer camera
./app_yolor  --model=../../data/yolor/yolor_csp.trt --size=640 --batch_size=2  --cam_id=0  --show


## 1280
./app_yolor  --model=../../data/yolor/yolor_p6.trt  --size=1280 --batch_size=1  --img=../../data/6406401.jpg  --show --savePath
```
## 5. appendix
ignore