# Exporting and serializing

Two steps: ultralytics exports the ONNX, `trtexec` turns it into an engine.

## 1. Export the ONNX (ultralytics CLI)

```bash
pip install ultralytics
yolo export model=yolo26n.pt format=onnx imgsz=640 batch=2 opset=17 simplify=True
```

The defaults land next to the weights as `yolo26n.onnx`. Swap `model=` for another task file (`yolo26n-seg.pt`, `yolo26n-pose.pt`, `yolo26n-obb.pt`, `yolo26n-cls.pt`, `yolo26n-depth.pt`) and adjust `imgsz`:

| Task | Native `imgsz` |
|---|---|
| det / pose / seg | 640 |
| depth | 768 |
| obb / sem | 1024 |
| cls | 224 |

These are the resolutions the engines in `engines/` were built at. Changing the resolution means rebuilding the engine at that size.

The CLI only exposes the common flags. For the full set use the Python API:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(
    format="onnx",
    batch=2,          # static batch; omit or -1 for a dynamic axis
    imgsz=640,        # native resolution of the task
    dynamic=False,    # True only if you intend to feed variable shapes
    simplify=True,
    opset=17,
    nms=None,         # keep NMS out of the graph, the CUDA kernels do it
)
```

`nms=None` matters: this repository runs decode + NMS in its own fused CUDA kernels, so the graph must stay raw. Keep the input name `images` and the output `output0` unless you also change `input_output_names` in `configs/<model>.ini`.

With `dynamic=False` the exported tensor is `[batch, 3, imgsz, imgsz]` and the output stays `[batch, 4 + nc, anchors]` — untouched postprocess, exactly what the kernels expect.

## 2. Serialize the engine (`trtexec`)

One command per model, all 21 registered ones: [serialize_engines.md](serialize_engines.md)

```bash
trtexec --onnx=yolo26n.onnx --saveEngine=engines/yolov26n.trt --noTF32
```

`--noTF32` is what the FP32 accuracy baseline uses (TF32 off, so fp32 does not drift). Precision tiers and their measured latency/throughput/size are in [quantization](quantization.md) — the short version: `--fp16` for the default speed-up tier, `--int8` only with the Q/DQ ONNX, and never both at once.

For a dynamic batch, give the profiles explicitly:

```bash
trtexec --onnx=yolo26n.onnx --saveEngine=yolov26n_dyn.trt \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:2x3x640x640 \
        --maxShapes=images:4x3x640x640
```

The INI's `batch_size` must then fall inside `[min, max]`; anything outside aborts at startup instead of being clamped.

## 3. Point the model at the engine

Engines live in `engines/<net>.trt`, and `configs/<net>.ini` carries `engine = engines/<net>.trt`. Either drop the file under the expected name, edit that key, or override per run:

```bash
trt_alpha run --net yolov26 --source data/bus.jpg --engine engines/yolov26n_fp16.trt --show
```

A mismatch between the INI's `batch_size` and a static engine fails loudly at startup, which is the intended behaviour — see [configuration](config.md).
