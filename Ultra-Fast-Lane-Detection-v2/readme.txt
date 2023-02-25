./trtexec.exe   --onnx=culane_dynamic.onnx  --saveEngine=culane_dynamic.trt  --buildOnly --minShapes=images:1x3x320x1600 --optShapes=images:4x3x320x1600 --maxShapes=images:8x3x320x1600

./trtexec.exe   --onnx=tusimple_dynamic.onnx  --saveEngine=tusimple_dynamic.trt  --buildOnly --minShapes=images:1x3x320x800 --optShapes=images:4x3x320x800 --maxShapes=images:8x3x320x800


--model=D:/ThirdParty/TensorRT-8.4.2.4/bin/culane_dynamic.trt   --batch_size=1  --img=D:/TensorRT-Alpha/Ultra-Fast-Lane-Detection-v2/009.jpg
--model=D:/ThirdParty/TensorRT-8.4.2.4/bin/culane_dynamic.trt   --batch_size=1  --img=D:/TensorRT-Alpha/Ultra-Fast-Lane-Detection-v2/road0.png

--model=D:/ThirdParty/TensorRT-8.4.2.4/bin/tusimple_dynamic.trt   --batch_size=1  --img=D:/TensorRT-Alpha/Ultra-Fast-Lane-Detection-v2/009.jpg
--model=D:/ThirdParty/TensorRT-8.4.2.4/bin/tusimple_dynamic.trt   --batch_size=1  --img=D:/TensorRT-Alpha/Ultra-Fast-Lane-Detection-v2/road0.png
--model=D:/ThirdParty/TensorRT-8.4.2.4/bin/tusimple_dynamic.trt   --batch_size=8  --video=D:/TensorRT-Alpha/data/people.mp4