## 特别说明
这里yolox官方在图像预处理的时候，其中resize是带有padding，只不过是在图像右边界、下边界进行padding
，本仓库的是核函数做法是：将图像进行等比例缩放插值，效果图存在m_input_resize_without_padding_device中，
然后将上述效果图像 copy 到m_input_resize_device(申请内存大小为：
416 * 416 * 3 * batch_size or 640 * 640 * 3 * batch_size,初始值为：{114, 114, 114})
另外，由于插值始终和opencv严格对齐，略有差异，但最终检测结果几乎一样，框的位置一样，置信度小数点后第二位才有差异。
最后，模型支持固定batch size

## 1. get onnx 
download onnx(default:batch_size=2) directly at [weiyun](https://share.weiyun.com/3T3mZKBm) or [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv?usp=sharing)
or export onnx:
```bash
git clone https://github.com/Megvii-BaseDetection/YOLOX
git checkout  0.3.0

## batch_size=1
# 640 for image
python tools/export_onnx.py --output-name=yolox_s.onnx  --exp_file=exps/default/yolox_s.py --ckpt=yolox_s.pth --decode_in_inference --batch-size=1
python tools/export_onnx.py --output-name=yolox_m.onnx  --exp_file=exps/default/yolox_m.py --ckpt=yolox_m.pth --decode_in_inference --batch-size=1
python tools/export_onnx.py --output-name=yolox_x.onnx  --exp_file=exps/default/yolox_x.py --ckpt=yolox_x.pth --decode_in_inference --batch-size=1
python tools/export_onnx.py --output-name=yolox_s.onnx  --exp_file=exps/default/yolox_s.py --ckpt=yolox_s.pth --decode_in_inference --batch-size=1
# 416 for image
python tools/export_onnx.py --output-name=yolox_nano.onnx --exp_file=exps/default/yolox_nano.py --ckpt=yolox_nano.pth --decode_in_inference --batch-size=1
python tools/export_onnx.py --output-name=yolox_tiny.onnx --exp_file=exps/default/yolox_tiny.py --ckpt=yolox_tiny.pth --decode_in_inference --batch-size=1

## batch_size > 1
# 例如将上述6条编译onnx的指令中，设置参数--batch-size=2，也是可行，但需注意：最后运行的时候，需要设置一样参数：--batch_size=2
```

## 2.edit and save onnx
```bash
# note: If you have obtained onnx by downloading, this step can be ignored
ignore
```

## 3.compile onnx
```bash
# put your onnx file in this path:tensorrt-alpha/data/yolox
cd tensorrt-alpha/data/yolox
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/feiyull/TensorRT-8.4.2.4/lib

# 640
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolox_s.onnx   --saveEngine=yolox_s.trt  --buildOnly
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolox_m.onnx   --saveEngine=yolox_m.trt  --buildOnly
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolox_x.onnx   --saveEngine=yolox_x.trt  --buildOnly

# 416
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolox_nano.onnx   --saveEngine=yolox_nano.trt  --buildOnly
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolox_tiny.onnx   --saveEngine=yolox_tiny.trt  --buildOnly
```

## 4.run
```bash
git clone https://github.com/FeiYull/tensorrt-alpha
cd tensorrt-alpha/yolox
mkdir build
cd build
cmake ..
make -j10
# note: the dstImage will be saved in tensorrt-alpha/yolox/build by default
# only support static multi-batch inference!
# the values of batch_size are different, and onnx needs to be compiled additionally

## 640
# infer image
./app_yolox  --model=../../data/yolox/yolox_s.trt     --size=640 --batch_size=1  --img=../../data/6406401.jpg  --show --savePath=../

# infer video
./app_yolox  --model=../../data/yolox/yolox_s.trt     --size=640 --batch_size=1  --video=../../data/people.mp4  --show 

# infer camera
./app_yolox  --model=../../data/yolox/yolox_s.trt     --size=640 --batch_size=1  --cam_id=0  --show

# 416
./app_yolox  --model=../../data/yolox/yolox_nano.trt  --size=416 --batch_size=1  --img=../../data/6406401.jpg  --show --savePath
```
## 5. appendix
ignore