## 1. get onnx
download directly at [weiyun](https://share.weiyun.com/3T3mZKBm) or [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv)

or export onnx:
```bash
git clone https://github.com/ShiqiYu/libfacedetection.train
git checkout  a3bc97c7e85bb206c9feca97fbd541ce82cfa3a9

# note：The official repository gives the following three models:
yunet_yunet_final_320_320_simplify.onnx
yunet_yunet_final_640_640_simplify.onnx
yunet_yunet_final_dynamic_simplify.onnx
choose the third model here.
```
## 2.edit and save onnx
```bash
# note: If you have obtained onnx by downloading, this step can be ignored
conda activate tensorrt-alpha
# put your onnx file in this path:tensorrt-alpha/data/libfacedetection
cd  tensorrt-alpha/data/libfacedetction
python alpha_edit.py --onnx=yunet_yunet_final_dynamic_simplify.onnx
```
## 3.compile onnx 
```bash
# put your onnx file in this path:tensorrt-alpha/data/libfacedetection
cd tensorrt-alpha/data/libfacedetection
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/feiyull/TensorRT-8.4.2.4/lib
../../../../TensorRT-8.4.2.4/bin/trtexec --onnx=alpha_yunet_yunet_final_dynamic_simplify.onnx --saveEngine=alpha_yunet_yunet_final_dynamic_simplify.trt --buildOnly --minShapes=input:1x3x120x120 --optShapes=input:4x3x320x320 --maxShapes=input:8x3x2000x2000
```
## 4.run
```bash
git clone https://github.com/FeiYull/tensorrt-alpha
cd tensorrt-alpha/libfacedetction
mkdir build
cd build
cmake ..
make -j10
# note: the dstImage will be saved in tensorrt-alpha/libfacedetction/build by default

# dynamic [b w h] 
# infer image
./app_libfacedetction  --model=../../data/libfacedetction/alpha_yunet_yunet_final_dynamic_simplify.trt  --batch_size=1  --img=../../data/6406401.jpg  --show --savePath

# infer video
./app_libfacedetction  --model=../../data/libfacedetction/alpha_yunet_yunet_final_dynamic_simplify.trt  --batch_size=8  --video=../../data/people.mp4  --show

# infer camera
./app_libfacedetction  --model=../../data/libfacedetction/alpha_yunet_yunet_final_dynamic_simplify.trt  --batch_size=2  --cam_id=0  --show
```
## 5. appendix
ignore