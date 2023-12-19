🚀TensorRT-Alpha现在支持yolov6.2.0和最新的yolov6.3.0
🍅TensorRT-Alpha support yolov6.2.0 and yolov6.3.0

# 1. get onnx
download directly at:[weiyun](https://share.weiyun.com/3T3mZKBm) or [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv?usp=sharing) or export onnx:
```bash
git clone https://github.com/meituan/YOLOv6
#---------------
# for yolov6.2.0
#---------------
git checkout  0.2.0
```
<details>
<summary>edit script export_onnx.py of yolov6.2.0</summary>

In order to support dynamic axes in yolo6.2.0, the official onnx export script export_onnx.py, we needs to be modified, and the following modifications are made:
```bash
torch.onnx.export(model, img, f, verbose=False, opset_version=12,
                              training=torch.onnx.TrainingMode.EVAL,
                              do_constant_folding=True,
                              input_names=['images'],
                              output_names=['num_dets', 'det_boxes', 'det_scores', 'det_classes']
                              if args.end2end and args.max_wh is None else ['outputs'],
                            #-----------------------------------------
                              # TensorRT-alpha
                              dynamic_axes={
                                'images': {
                                    0: 'batch',
                                    2: 'height',
                                    3: 'width'},  # shape(1,3,640,640)
                                'outputs': {
                                    0: 'batch',
                                    1: 'anchors'}  # shape(1,25200,85)
                            } 
                            #-----------------------------------------
                              )
# 640
python ./deploy/ONNX/export_onnx.py   --weights  yolov6n.pt
python ./deploy/ONNX/export_onnx.py   --weights  yolov6s.pt
python ./deploy/ONNX/export_onnx.py   --weights  yolov6m.pt
# 1280
python ./deploy/ONNX/export_onnx.py   --weights  yolov6n6.pt
python ./deploy/ONNX/export_onnx.py   --weights  yolov6s6.pt
python ./deploy/ONNX/export_onnx.py   --weights  yolov6m6.pt

```
</details>


```bash
#---------------
# for yolov6.3.0
#---------------
git checkout  0.3.0
# 640
python ./deploy/ONNX/export_onnx.py --weights yolov6n.pt  --img 640 --dynamic-batch  --simplify
python ./deploy/ONNX/export_onnx.py --weights yolov6s.pt  --img 640 --dynamic-batch  --simplify
python ./deploy/ONNX/export_onnx.py --weights yolov6m.pt  --img 640 --dynamic-batch  --simplify
python ./deploy/ONNX/export_onnx.py --weights yolov6l.pt  --img 640 --dynamic-batch  --simplify
# 1280
python ./deploy/ONNX/export_onnx.py --weights yolov6n6.pt  --img 1280 --dynamic-batch  --simplify
python ./deploy/ONNX/export_onnx.py --weights yolov6s6.pt  --img 1280 --dynamic-batch  --simplify
python ./deploy/ONNX/export_onnx.py --weights yolov6m6.pt  --img 1280 --dynamic-batch  --simplify
python ./deploy/ONNX/export_onnx.py --weights yolov6l6.pt  --img 1280 --dynamic-batch  --simplify
```
## 2.edit and save onnx
```bash
# note: If you have obtained onnx by downloading, this step can be ignored
ignore
```
## 3.compile onnx
```bash
# put your onnx file in this path:tensorrt-alpha/data/yolov6
cd tensorrt-alpha/data/yolov6
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/feiyull/TensorRT-8.4.2.4/lib
# 640
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolov6n.onnx   --saveEngine=yolov6n.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolov6s.onnx   --saveEngine=yolov6s.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolov6m.onnx   --saveEngine=yolov6m.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolov6l.onnx   --saveEngine=yolov6l.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
# 1280
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolov6n6.onnx   --saveEngine=yolov6n6.trt  --buildOnly --minShapes=images:1x3x1280x1280 --optShapes=images:2x3x1280x1280 --maxShapes=images:4x3x1280x1280
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolov6s6.onnx   --saveEngine=yolov6s6.trt  --buildOnly --minShapes=images:1x3x1280x1280 --optShapes=images:2x3x1280x1280 --maxShapes=images:4x3x1280x1280
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolov6m6.onnx   --saveEngine=yolov6m6.trt  --buildOnly --minShapes=images:1x3x1280x1280 --optShapes=images:2x3x1280x1280 --maxShapes=images:4x3x1280x1280
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolov6l6.onnx   --saveEngine=yolov6l6.trt  --buildOnly --minShapes=images:1x3x1280x1280 --optShapes=images:2x3x1280x1280 --maxShapes=images:4x3x1280x1280
```
## 4.run 
```bash
git clone https://github.com/FeiYull/tensorrt-alpha
cd tensorrt-alpha/yolov6
mkdir build
cd build
cmake ..
make -j10
# note: the dstImage will be saved in tensorrt-alpha/yolov6/build by default
## 640
# infer image
./app_yolov6  --model=../../data/yolov6/yolov6n.trt --size=640  --batch_size=1  --img=../../data/6406401.jpg  --show --savePath=../

# infer video
./app_yolov6  --model=../../data/yolov6/yolov6n.trt     --size=640 --batch_size=2  --video=../../data/people.mp4  --show 

# infer camera
./app_yolov6  --model=../../data/yolov6/yolov6n.trt     --size=640 --batch_size=2  --cam_id=0  --show

## 1280
# infer video
./app_yolov6  --model=../../data/yolov6/yolov6s6.trt --size=1280  --batch_size=2  --video=../../data/people.mp4  --savePath=../
```
## 5. appendix
ignore