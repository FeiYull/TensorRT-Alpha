## 1. get onnx
download directly at [weiyun](https://share.weiyun.com/3T3mZKBm) or [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv)

or export onnx:
```bash
git clone https://github.com/thepowerfuldeez/facemesh.pytorch
# Please use the alpha_export.py script provided by this repo to export onnx
cp alpha_export.py facemesh.pytorch
cd facemesh.pytorch
python alpha_export.py  --weights=facemesh.pth
```
## 2.edit and save onnx
```bash
# note: If you have obtained onnx by downloading, this step can be ignored
ignore
```
## 3.compile onnx
```bash
# put your onnx file in this path:tensorrt-alpha/data/facemesh
cd tensorrt-alpha/data/facemesh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/TensorRT-8.4.2.4/lib
../../../../TensorRT-8.4.2.4/bin/trtexec --onnx=facemesh.onnx   --saveEngine=facemesh.trt   --minShapes=image:2x3x192x192 --optShapes=image:2x3x192x192 --maxShapes=image:8x3x192x192
```
## 4.run
```bash
git clone https://github.com/FeiYull/tensorrt-alpha
cd tensorrt-alpha/facemesh
mkdir build
cd build
cmake ..
make -j10
# note: the dstImage will be saved in tensorrt-alpha/facemesh/build by default
./app_facemesh  --model=../../data/facemesh/facemesh.trt --size=192 --imgs_dir=../../data/  --show --savePath
```
## 5. appendix
ignore