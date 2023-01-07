+------------+
|1. get onnx |
+------------+
download directly at [weiyun](https://share.weiyun.com/3T3mZKBm) or [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv)

or export onnx:

git clone https://github.com/ShiqiYu/libfacedetection.train
git checkout  a3bc97c7e85bb206c9feca97fbd541ce82cfa3a9

The official repository gives the following three models:
yunet_yunet_final_320_320_simplify.onnx
yunet_yunet_final_640_640_simplify.onnx
yunet_yunet_final_dynamic_simplify.onnx

choose the third model here.


+---------------------+
|2.edit and save onnx |
+---------------------+
# note: If you have obtained onnx by downloading, this step can be ignored
conda activate tensorrt-alpha
python alpha_edit.py --onnx=../data/libfacedetction/yunet_yunet_final_dynamic_simplify.onnx

+----------------+
| 3.compile onnx |
+----------------+
# put your onnx file in this path:tensorrt-alpha/data/libfacedetection
cd tensorrt-alpha/data/libfacedetection
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/TensorRT-8.4.2.4/lib
../../../../TensorRT-8.4.2.4/bin/trtexec --onnx=alpha_yunet_yunet_final_dynamic_simplify.onnx --saveEngine=alpha_yunet_yunet_final_dynamic_simplify.trt --workspace=1 --buildOnly --minShapes=input:1x3x120x120 --optShapes=input:4x3x320x320 --maxShapes=input:8x3x2000x2000

+------+
|4.run |
+------+
git clone https://github.com/FeiYull/tensorrt-alpha
cd tensorrt-alpha/libfacedetction
mkdir build
cd build
cmake ..
make -j10
# note: the dstImage will be saved in tensorrt-alpha/libfacedetction/build by default
./app_libfacedetction  --model=../../data/libfacedetction/alpha_yunet_yunet_final_dynamic_simplify.trt  --batch_size=4  --img=../../data/6406401.jpg  --show --savePath
./app_libfacedetction  --model=../../data/libfacedetction/alpha_yunet_yunet_final_dynamic_simplify.trt  --batch_size=4  --video=../../data/people.mp4  --show


+-----------+
|5. appendix|
+-----------+
ignore