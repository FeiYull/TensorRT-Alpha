1. preprocess(100%)
2.infer(100%)
3.postprocess(wating to update)

Todo: concat the output0 output1 output2 output3 by torch

# compily onnx 
./trtexec.exe   --onnx=culane_dynamic.onnx  --saveEngine=culane_dynamic.trt  --buildOnly --minShapes=images:1x3x320x1600 --optShapes=images:4x3x320x1600 --maxShapes=images:8x3x320x1600
./trtexec.exe   --onnx=tusimple_dynamic.onnx  --saveEngine=tusimple_dynamic.trt  --buildOnly --minShapes=images:1x3x320x800 --optShapes=images:4x3x320x800 --maxShapes=images:8x3x320x800


# Command line parameters
--model=D:/ThirdParty/TensorRT-8.4.2.4/bin/culane_dynamic.trt   --batch_size=1  --img=D:/TensorRT-Alpha/Ultra-Fast-Lane-Detection-v2/009.jpg
--model=D:/ThirdParty/TensorRT-8.4.2.4/bin/culane_dynamic.trt   --batch_size=1  --img=D:/TensorRT-Alpha/Ultra-Fast-Lane-Detection-v2/road0.png

--model=D:/ThirdParty/TensorRT-8.4.2.4/bin/tusimple_dynamic.trt   --batch_size=1  --img=D:/TensorRT-Alpha/Ultra-Fast-Lane-Detection-v2/009.jpg
--model=D:/ThirdParty/TensorRT-8.4.2.4/bin/tusimple_dynamic.trt   --batch_size=1  --img=D:/TensorRT-Alpha/Ultra-Fast-Lane-Detection-v2/road0.png
--model=D:/ThirdParty/TensorRT-8.4.2.4/bin/tusimple_dynamic.trt   --batch_size=8  --video=D:/TensorRT-Alpha/data/people.mp4


# export onnx by alpha_export.py:
python alpha_export.py configs/culane_res18.py      --test_model culane_res18.pth
python alpha_export.py configs/tusimple_res18.py  --test_model tusimple_res18.pth

# inference onnx by onnx_inference_temp.py:
python onnx_inference_temp.py # Note:Tempora on 29 lines and 40 lines hand -switch configuration parameters