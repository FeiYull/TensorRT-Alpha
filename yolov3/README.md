## 1. get onnx 
download directly at [weiyun](https://share.weiyun.com/3T3mZKBm) or [google driver](https://drive.google.com/drive/folders/1-8phZHkx_Z274UVqgw6Ma-6u5AKmqCOv)
or export onnx:
```bash
git clone https://github.com/ultralytics/yolov3
git checkout  dd838e25863169d0de4f10631a609350658efb69
```
```bash
# note: When using the official export.py to export onnx, you need to comment the following two lines：
#---------------------------------------------------------------------------------------------------------
if simplify:
            try:
                check_requirements(('onnx-simplifier',))
                import onnxsim

                LOGGER.info(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(
                    model_onnx,
	    #-------------------------------------------------------------------------------
                    #dynamic_input_shape=dynamic, 
                    #input_shapes={'images': list(im.shape)} if dynamic else None
	    #-------------------------------------------------------------------------------
                    )
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                LOGGER.info(f'{prefix} simplifier failure: {e}')
        LOGGER.info(f'{prefix} export success, saved as {f} ({file_size(f):.1f} MB)')
        LOGGER.info(f"{prefix} run --dynamic ONNX model inference with: 'python detect.py --weights {f}'")
#---------------------------------------------------------------------------------------------------------
```
```bash
cd yolov3
python export.py  --weights  yolov3-tiny.pt --dynamic --simplify
python export.py  --weights  yolov3.pt      --dynamic --simplify
python export.py  --weights  yolov3-spp.pt  --dynamic
```
## 2.edit and save onnx
```bash
# note: If you have obtained onnx by downloading, this step can be ignored
git clone https://github.com/FeiYull/tensorrt-alpha
cd tensorrt-alpha/yolov3
conda activate tensorrt-alpha
#edit alpha_edit.py on line21 & line25.
python alpha_edit.py
```
## 3.compile onnx
```bash
# put your onnx file in this path:tensorrt-alpha/data/yolov3
cd tensorrt-alpha/data/yolov3
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/feiyull/TensorRT-8.4.2.4/lib
# 640
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=alpha_yolov3.onnx        --saveEngine=alpha_yolov3.trt      --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=alpha_yolov3-spp.onnx    --saveEngine=alpha_yolov3-spp.trt  --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640
../../../../TensorRT-8.4.2.4/bin/trtexec   --onnx=alpha_yolov3-tiny.onnx   --saveEngine=alpha_yolov3-tiny.trt --buildOnly --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640

# note: When compiling the alpha_yolov3-tiny model, if the following error may occur
# error: Error Code 4: Internal Error (/model.11/Reshape: IShuffleLayer applied to shape tensor must have 0 or 1 #reshape dimensions: dimensions were [-1,2])
# solve：Add the parameter --simplify when exporting onnx (opset defaults to 13, which is high enough)
```
## 4.run
```bash
git clone https://github.com/FeiYull/tensorrt-alpha
cd tensorrt-alpha/yolov3
mkdir build
cd build
cmake ..
make -j10
# note: the dstImage will be saved in tensorrt-alpha/yolov3/build by default

## 640
# infer image
./app_yolov3  --model=../../data/yolov3/alpha_yolov3-tiny.trt --size=640  --batch_size=1  --img=../../data/6406403.jpg  --show --savePath

# infer video
./app_yolov3  --model=../../data/yolov3/alpha_yolov3-tiny.trt --size=640 --batch_size=2  --video=../../data/people.mp4  --show

# infer camera
./app_yolov3  --model=../../data/yolov3/alpha_yolov3-tiny.trt --size=640 --batch_size=2  --cam_id=0  --show

# note:yolov3-tiny has obvious missed detection on the image 6406401.jpg, don't worry, the effect is consistent with the official
```
## 5. appendix
ignore