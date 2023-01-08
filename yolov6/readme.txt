+------------+
|1. get onnx |
+------------+
download directly at [weiyun](https://share.weiyun.com/3T3mZKBm) or [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv)

or export onnx:

git clone https://github.com/meituan/YOLOv6
git checkout  0.2.0
# In order to support dynamic axes, the official onnx export script export_onnx.py 
needs to be modified, and the following modifications are made
#*****************************************************************************************************************
torch.onnx.export(model, img, f, verbose=False, opset_version=12,
                              training=torch.onnx.TrainingMode.EVAL,
                              do_constant_folding=True,
                              input_names=['images'],
                              output_names=['num_dets', 'det_boxes', 'det_scores', 'det_classes']
                              if args.end2end and args.max_wh is None else ['outputs'],
	              #***************************************************************
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
	             #***************************************************************

                              )
#*****************************************************************************************************************

python export_onnx.py   --weights  yolov6n.pt

+---------------------+
|2.edit and save onnx |
+---------------------+
# note: If you have obtained onnx by downloading, this step can be ignored
ignore

+----------------+
| 3.compile onnx |
+----------------+
# put your onnx file in this path:tensorrt-alpha/data/yolov6
cd tensorrt-alpha/data/yolov6
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/TensorRT-8.4.2.4/lib
# 640
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=yolov6n.onnx   --saveEngine=yolov6n.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640

+------+
|4.run |
+------+
git clone https://github.com/FeiYull/tensorrt-alpha
cd tensorrt-alpha/yolov6
mkdir build
cd build
cmake ..
make -j10
# note: the dstImage will be saved in tensorrt-alpha/yolov6/build by default
## 640
# infer image
./app_yolov6  --model=../../data/yolov6/yolov6n.trt --size=640  --batch_size=1  --img=../../data/6406401.jpg  --show --savePath

# infer video
./app_yolov6  --model=../../data/yolov6/yolov6n.trt     --size=640 --batch_size=8  --video=../../data/people.mp4  --show --savePath=../

# infer camera
./app_yolov6  --model=../../data/yolov6/yolov6n.trt     --size=640 --batch_size=4  --cam_id=0  --show

+-----------+
|5. appendix|
+-----------+
ignore