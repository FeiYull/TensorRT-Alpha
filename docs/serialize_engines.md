# trtexec commands

## yolov26

```bash
trtexec --onnx=yolo26n.onnx \
        --saveEngine=engines/yolov26n.trt \
        --noTF32
```

## yolov26_cls

```bash
trtexec --onnx=yolo26n-cls.onnx \
        --saveEngine=engines/yolov26n_cls.trt \
        --noTF32
```

## yolov26_depth

```bash
trtexec --onnx=yolo26n-depth.onnx \
        --saveEngine=engines/yolov26n_depth.trt \
        --noTF32
```

## yolov26_obb

```bash
trtexec --onnx=yolo26n-obb.onnx \
        --saveEngine=engines/yolov26n_obb.trt \
        --noTF32
```

## yolov26_pose

```bash
trtexec --onnx=yolo26n-pose.onnx \
        --saveEngine=engines/yolov26n_pose.trt \
        --noTF32
```

## yolov26_seg

```bash
trtexec --onnx=yolo26n-seg.onnx \
        --saveEngine=engines/yolov26n_seg.trt \
        --noTF32
```

## yolov26_sem

```bash
trtexec --onnx=yolo26n-sem.onnx \
        --saveEngine=engines/yolov26n_sem.trt \
        --noTF32
```

## efficientdet

```bash
trtexec --onnx=efficientdet0.onnx \
        --saveEngine=engines/efficientdet0.trt \
        --minShapes=input:1x512x512x3 \
        --optShapes=input:2x512x512x3 \
        --maxShapes=input:4x512x512x3
```

## u2net

```bash
trtexec --onnx=u2net.onnx \
        --saveEngine=engines/u2net.trt \
        --minShapes=images:1x3x320x320 \
        --optShapes=images:2x3x320x320 \
        --maxShapes=images:4x3x320x320
```

## yolo_nas

```bash
trtexec --onnx=yolonas_s.onnx \
        --saveEngine=engines/yolonas_s.trt \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:2x3x640x640 \
        --maxShapes=images:4x3x640x640
```

## yolor

```bash
trtexec --onnx=yolor_csp.onnx \
        --saveEngine=engines/yolor_csp.trt \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:2x3x640x640 \
        --maxShapes=images:4x3x640x640
```

## yolov3

```bash
trtexec --onnx=alpha_yolov3-spp.onnx \
        --saveEngine=engines/alpha_yolov3_spp.trt \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:2x3x640x640 \
        --maxShapes=images:4x3x640x640
```

## yolov4

```bash
trtexec --onnx=alpha_yolov4_-1_3_608_608_dynamic.onnx \
        --saveEngine=engines/alpha_yolov4_1_3_608_608_dynamic.trt \
        --minShapes=input:1x3x608x608 \
        --optShapes=input:2x3x608x608 \
        --maxShapes=input:4x3x608x608
```

## yolov5

```bash
trtexec --onnx=yolov5n.onnx \
        --saveEngine=engines/yolov5n.trt \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:2x3x640x640 \
        --maxShapes=images:4x3x640x640
```

## yolov6

```bash
trtexec --onnx=yolov6n.onnx \
        --saveEngine=engines/yolov6n.trt \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:2x3x640x640 \
        --maxShapes=images:4x3x640x640
```

## yolov7

```bash
trtexec --onnx=yolov7-tiny.onnx \
        --saveEngine=engines/yolov7_tiny.trt \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:2x3x640x640 \
        --maxShapes=images:4x3x640x640
```

## yolov8

```bash
trtexec --onnx=yolov8n.onnx \
        --saveEngine=engines/yolov8n.trt \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:2x3x640x640 \
        --maxShapes=images:4x3x640x640
```

## yolov8_pose

```bash
trtexec --onnx=yolov8n-pose.onnx \
        --saveEngine=engines/yolov8n_pose.trt \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:2x3x640x640 \
        --maxShapes=images:4x3x640x640
```

## yolov8_seg

```bash
trtexec --onnx=yolov8n-seg.onnx \
        --saveEngine=engines/yolov8n_seg.trt \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:2x3x640x640 \
        --maxShapes=images:4x3x640x640
```

## yolox

```bash
trtexec --onnx=yolox_s.onnx \
        --saveEngine=engines/yolox_s.trt
```

## yunet

```bash
trtexec --onnx=alpha_yunet_yunet_final_dynamic_simplify.onnx \
        --saveEngine=engines/alpha_yunet_yunet_final_dynamic_simplify.trt \
        --minShapes=input:1x3x120x120 \
        --optShapes=input:2x3x320x320 \
        --maxShapes=input:4x3x2000x2000
```
