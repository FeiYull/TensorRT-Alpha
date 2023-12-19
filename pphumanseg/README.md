## 1. get onnx
download directly at [weiyun](https://share.weiyun.com/3T3mZKBm) or [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv)

or export onnx:
```bash
# Install git-lfs from https://git-lfs.github.com/
git clone https://github.com/opencv/opencv_zoo && cd opencv_zoo
git checkout  ae1d754a3ea14e4244fbea7d781cca2e18584035
git lfs install
git lfs pull
# note：The official onnx is in this path:opencv_zoo/models/human_segmentation_pphumanseg.
```
## 2.edit and save onnx
```bash
# note: If you have obtained onnx by downloading, this step can be ignored
conda activate tensorrt-alpha
# put your onnx file in this path:tensorrt-alpha/data/pphumanseg
cd  tensorrt-alpha/data/pphumanseg
python alpha_edit.py --onnx=../data/pphumanseg/human_segmentation_pphumanseg_2021oct.onnx
```
## 3.compile onnx 
```bash
# put your onnx file in this path:tensorrt-alpha/data/pphumanseg
cd tensorrt-alpha/data/pphumanseg
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/feiyull/TensorRT-8.4.2.4/lib
../../../../TensorRT-8.4.2.4/bin/trtexec --onnx=human_segmentation_pphumanseg_2021oct_dynamic.onnx   --saveEngine=human_segmentation_pphumanseg_2021oct_dynamic.trt  --buildOnly  --minShapes=x:1x3x192x192 --optShapes=x:2x3x192x192 --maxShapes=x:4x3x192x192
```
## 4.run
```bash
git clone https://github.com/FeiYull/tensorrt-alpha
cd tensorrt-alpha/pphumanseg
mkdir build
cd build
cmake ..
make -j10
# note: the dstImage will be saved in tensorrt-alpha/pphumanseg/build by default

# infer image
./app_pphunmanseg  --model=../../data/pphumanseg/human_segmentation_pphumanseg_2021oct_dynamic.trt --img=../../data/6.jpg  --size=192 --batch_size=1 --show -savePath

# infer video
./app_pphunmanseg  --model=../../data/pphumanseg/human_segmentation_pphumanseg_2021oct_dynamic.trt  --batch_size=2  --video=../../data/people.mp4  --show

# infer camera
./app_pphunmanseg  --model=../../data/pphumanseg/human_segmentation_pphumanseg_2021oct_dynamic.trt  --batch_size=2  --cam_id=0  --show
```
## 5. appendix
ignore