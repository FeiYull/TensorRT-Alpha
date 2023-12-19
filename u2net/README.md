## 1. get onnx
download directly at [weiyun](https://share.weiyun.com/3T3mZKBm) or [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv)

or export onnx:
```bash
git clone https://github.com/xuebinqin/U-2-Net
cd U-2-Net-master
# Use the script alpha_export.py provided by this repo to export onnx
cp alpha_export.py U-2-Net-master
python alpha_export.py --net=u2net --weights=saved_models/u2net/u2net.pth
python alpha_export.py --net=u2netp --weights=saved_models/u2netp/u2netp.pth
```
## 2.edit and save onnx
```bash
# note: If you have obtained onnx by downloading, this step can be ignored
ignore
```
## 3.compile onnx
```bash
# put your onnx file in this path:tensorrt-alpha/data/u2net
cd tensorrt-alpha/data/u2net
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/feiyull/TensorRT-8.4.2.4/lib

../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=u2net.onnx   --saveEngine=u2net.trt   --buildOnly --minShapes=images:1x3x320x320 --optShapes=images:4x3x320x320 --maxShapes=images:8x3x320x320
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=u2netp.onnx  --saveEngine=u2netp.trt  --buildOnly --minShapes=images:1x3x320x320 --optShapes=images:4x3x320x320 --maxShapes=images:8x3x320x320
```
## 4.run
```bash
git clone https://github.com/FeiYull/tensorrt-alpha
cd tensorrt-alpha/u2net
mkdir build
cd build
cmake ..
make -j10
# note: the dstImage will be saved in tensorrt-alpha/u2net/build by default

## 320
# infer image
./app_u2net  --model=../../data/u2net/u2net.trt --size=320  --batch_size=1  --img=../../data/sailboat3.jpg  --show --savePath

# infer video
./app_u2net  --model=../../data/u2net/u2net.trt --size=320 --batch_size=2  --video=../../data/people.mp4  --show

# infer camera
./app_u2net  --model=../../data/u2net/u2net.trt --size=320 --batch_size=2  --cam_id=0  --show
```
## 5. appendix
ignore