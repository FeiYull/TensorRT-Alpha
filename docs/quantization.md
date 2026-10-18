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

## yolo11: PTQ for all five tasks

Every yolo11 task ships a dedicated quantized ONNX pair, so INT8 no longer goes
through TensorRT's dynamic calibration. The export is the stock ultralytics PTQ
(`quantize=16` / `quantize=8`); the one deviation is the symmetric Q/DQ
quantizer in `sry_export.py`, required because TensorRT rejects a non-zero zero
point.

```bash
# FP16: onnxruntime float16 conversion, graph I/O stays fp32, no calibration
python sry_export.py 16 --weights yolo11n.pt     --imgsz 640
python sry_export.py 16 --weights yolo11n-seg.pt --imgsz 640
python sry_export.py 16 --weights yolo11n-pose.pt --imgsz 640
python sry_export.py 16 --weights yolo11n-cls.pt --imgsz 224
python sry_export.py 16 --weights yolo11n-obb.pt --imgsz 1024

# INT8: Q/DQ graph, only Conv / Gemm / MatMul quantized (the head stays float)
python sry_export.py 8 --weights yolo11n.pt      --imgsz 640  --data coco8.yaml
python sry_export.py 8 --weights yolo11n-seg.pt  --imgsz 640  --data <seg_calib.yaml>
python sry_export.py 8 --weights yolo11n-pose.pt --imgsz 640  --data <pose_calib.yaml>
python sry_export.py 8 --weights yolo11n-cls.pt  --imgsz 224  --data <cls_dir>
python sry_export.py 8 --weights yolo11n-obb.pt  --imgsz 1024 --data <obb_calib.yaml>
```

Calibration only ever reads the images, so the labels can be empty. That is also
the way around two exporter constraints: a segment dataset must have
`len(segments) == len(boxes)` (an empty label file satisfies both counts, a
detect dataset does not), and a pose dataset must carry `kpt_shape`. `cls` takes
a directory with `train/<class>/` and `val/<class>/` instead of a yaml.

```bash
trtexec --onnx=yolo11n_fp16.onnx --saveEngine=engines/yolo11n_fp16.trt --fp16
trtexec --onnx=yolo11n_int8.onnx --saveEngine=engines/yolo11n_int8.trt --int8
# ... same for yolo11n_{seg,pose,cls,obb}, see serialize_engines.md
```

Engine sizes, Windows + TensorRT 10.16, `batch = 2`:

| Task | FP32 (`--noTF32`) | FP16 (`--fp16`) | INT8 (`--int8`) |
|---|---|---|---|
| det | 14.95 MB | 8.59 MB | 6.25 MB |
| seg | 15.86 MB | 9.31 MB | 10.62 MB |
| pose | 16.22 MB | 9.41 MB | 7.49 MB |
| cls | 12.67 MB | 6.87 MB | 4.59 MB |
| obb | 15.73 MB | 11.36 MB | 8.29 MB |

INT8 is not always the smallest: `seg` grows by 1.3 MB over FP16 because the
mask-prototype path keeps its own layers out of the quantized set, and every
Q/DQ boundary adds a conversion kernel. Smallest is not the same as fastest --
measure before picking a tier for segmentation.

Same box, `yolo11n` detection, `batch = 2`, 200 iterations, mean per batch:
FP32 **2.441 ms** / FP16 **1.698 ms** / INT8 **1.993 ms**. FP16 is the sweet
spot here. The Q/DQ graph leaves the whole head in FP32 and pays a conversion at
every quantized/dequantized boundary, so INT8 ends up ~17% slower than FP16 even
though its file is smaller.

## Caveats

- `--noTF32` is **only for FP32 engines used as the accuracy baseline**. Do not add it to FP16 / INT8 engines: it disables the TF32 tensor cores and, on an already-quantized graph, only throws away speed.
- Never pass `--fp16` and `--int8` together. TensorRT 10.16 fails to build weakly-typed quantized networks that way.
- **yolo11 has a dedicated ONNX pair for all five tasks** (`yolo11n*_{fp16,int8}.onnx`). For `yolov26` only object detection does; the other yolov26 task ONNX files are full-precision graphs, and adding `--int8` to those triggers dynamic calibration, which inflates the engine by 2~7.7x and makes it slower, so **do not build them that way**.
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
