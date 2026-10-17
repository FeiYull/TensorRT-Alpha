# Quantization

## Three precision tiers

| Tier | Engine | Serialization flag | Purpose |
|---|---|---|---|
| FP32 | `engines/yolov26n.trt` | `--noTF32` | Accuracy-alignment baseline (TF32 off, so fp32 does not drift) |
| FP16 | `engines/yolov26n_fp16.trt` | `--fp16` | Default speed-up tier |
| INT8 | `engines/yolov26n_int8.trt` | `--int8` | Needs the Q/DQ quantized ONNX exported by the Python-side PTQ |

## Building

```bash
# FP32: the command this repository uses for accuracy validation (TF32 off, to avoid fp32 drift)
trtexec --onnx=sample.onnx --saveEngine=sample.trt --noTF32

# FP16: --fp16 is mandatory, otherwise TensorRT computes in FP32
trtexec --onnx=yolo26n_fp16.onnx --saveEngine=yolov26n_fp16.trt --fp16

# INT8: use the Q/DQ quantized ONNX produced by the Python-side PTQ, plus --int8
trtexec --onnx=yolo26n_int8.onnx --saveEngine=yolov26n_int8.trt --int8

# Let TensorRT pick the best precision combination on its own
trtexec --onnx=sample.onnx --saveEngine=sample.trt --best
```

## Caveats

- `--noTF32` is **only for FP32 engines used as the accuracy baseline**. Do not add it to FP16 / INT8 engines: it disables the TF32 tensor cores and, on an already-quantized graph, only throws away speed.
- Never pass `--fp16` and `--int8` together. TensorRT 10.16 fails to build weakly-typed quantized networks that way.
- **Only object detection has dedicated ONNX files**: `onnx/yolo26n_fp16.onnx` and `onnx/yolo26n_int8.onnx` (the latter is a Q/DQ quantized graph). The ONNX files for the other tasks are full-precision graphs; adding `--int8` to them triggers dynamic calibration, which inflates the engine by 2~7.7x and makes it slower, so **do not build them that way**.
- Changing the resolution means rebuilding the engine at that size: det / pose / seg use 640, depth uses 768, obb / sem use 1024, cls uses 224.

## Performance and size

YOLO26n, `batch = 2`, Ubuntu 24.04 + RTX 5060 Ti, `trtexec --loadEngine` with 200 iterations (averaged over 100):

| Engine | Serialization flag | GPU latency | Throughput | Size |
|---|---|---|---|---|
| `yolov26n.trt` (FP32) | `--noTF32` | 2.401 ms | 302.4 qps | 13.03 MB |
| **`yolov26n_fp16.trt` (FP16)** | **`--fp16`** | **1.708 ms** | **383.4 qps** | 8.20 MB |

> The `trtexec` latency covers **inference only**. Per-frame end-to-end stage costs are in the README's [performance showcase](../README.md#部分性能展示).

## Accuracy

The alignment procedure used before and after quantization is described in the README's [strict accuracy alignment](../README.md#严格精度对齐): reference values must come from **onnxruntime (square letterbox) plus the official postprocess operators**. Do not use `.pt` `predict()` as the baseline, because both the preprocess and the postprocess branch per backend.
