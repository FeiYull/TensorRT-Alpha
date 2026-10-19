# Exporting and serializing

Every model needs the same two steps:

1. **Export** — ultralytics (or, for the legacy families, the model's own repo)
   produces the ONNX.
2. **Serialize** — `trtexec` turns that ONNX into `engines/<name>.trt`.

Below is one section per registered model: the export command, then the
serialize command. `configs/<net>.ini` already points at the engine path used
here, so nothing else needs wiring.

## Before you start

```bash
pip install ultralytics
```

- **Static batch = 2.** Every fused family (yolov8 / 9 / 10 / 11 / 26) is
  exported with the batch baked into the graph. A static engine refuses any
  other batch, and the INI's `batch_size` must equal it — a mismatch aborts at
  startup instead of being clamped.
- **Keep NMS out of the graph.** This repository runs decode + NMS in its own
  fused CUDA kernels. From the Python API pass `nms=None`; the CLI default
  already leaves it out.
- **Keep the tensor names for consistency.** Input is `images`, outputs are
  `output0` / `output1`. Nothing on the C++ side reads a name from the INI: each
  model takes the first input and the first N outputs of the engine as declared
  (the engine is the single source of truth), so a rename will not break it --
  but keeping the upstream names makes cross-checking against Python simpler.
- **`--noTF32` on every FP32 build.** TF32 is on by default from Ampere on, and
  it drifts enough to reshuffle NMS; the accuracy baseline in this repo is
  TF32-off.
- **Naming.** Upstream weights and the ONNX derived from them keep the
  ultralytics spelling `yolo11n*` (that is the real filename on disk); engines
  and registrations use `yolov11n*`. So `yolo11n.pt` → `yolo11n.onnx` →
  `engines/yolov11n.trt`.

### Full flag set (Python API)

The CLI only exposes the common flags; for the rest use the API:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(
    format="onnx",
    batch=2,          # static batch; -1 / dynamic=True for a dynamic axis
    imgsz=640,        # native resolution of the task
    dynamic=False,    # True only if you intend to feed variable shapes
    simplify=True,
    opset=17,
    nms=None,         # keep NMS out of the graph, the CUDA kernels do it
)
```

With `dynamic=False` the input is `[batch, 3, imgsz, imgsz]` and the output
stays `[batch, 4 + nc, anchors]` — postprocess untouched, exactly what the
kernels expect.

### Native resolution per task

These are the resolutions the shipped engines were built at. Building at
another size means exporting and serializing at that size — the INI does not
carry spatial dimensions, the engine's declared shape is the single source of
truth. The legacy families (yolov3 … u2net) each state their own size below.

| task | `imgsz` |
|---|---|
| det / pose / seg | 640 |
| cls | 224 (short-side resize + centre crop, not a letterbox) |
| depth | 768 |
| obb / sem | 1024 |

## yolov26

det · 640 · static batch = 2

```bash
# export
yolo export model=yolo26n.pt format=onnx imgsz=640 batch=2 opset=17

# serialize
trtexec --onnx=yolo26n.onnx --saveEngine=engines/yolov26n.trt --noTF32
```

## yolov26_cls

cls · 224 · static batch = 2

```bash
# export
yolo export model=yolo26n-cls.pt format=onnx imgsz=224 batch=2 opset=17

# serialize
trtexec --onnx=yolo26n-cls.onnx --saveEngine=engines/yolov26n_cls.trt --noTF32
```

## yolov26_depth

depth · 768 · static batch = 2

```bash
# export
yolo export model=yolo26n-depth.pt format=onnx imgsz=768 batch=2 opset=17

# serialize
trtexec --onnx=yolo26n-depth.onnx --saveEngine=engines/yolov26n_depth.trt --noTF32
```

## yolov26_obb

obb · 1024 (DOTA training resolution) · static batch = 2 · nc = 15

```bash
# export
yolo export model=yolo26n-obb.pt format=onnx imgsz=1024 batch=2 opset=17

# serialize
trtexec --onnx=yolo26n-obb.onnx --saveEngine=engines/yolov26n_obb.trt --noTF32
```

## yolov26_pose

pose · 640 · static batch = 2 · 17 keypoints

```bash
# export
yolo export model=yolo26n-pose.pt format=onnx imgsz=640 batch=2 opset=17

# serialize
trtexec --onnx=yolo26n-pose.onnx --saveEngine=engines/yolov26n_pose.trt --noTF32
```

## yolov26_seg

seg · 640 · static batch = 2 · two outputs (`output0` predictions, `output1` mask prototypes)

```bash
# export
yolo export model=yolo26n-seg.pt format=onnx imgsz=640 batch=2 opset=17

# serialize
trtexec --onnx=yolo26n-seg.onnx --saveEngine=engines/yolov26n_seg.trt --noTF32
```

## yolov26_sem

sem · 1024 · static batch = 2

```bash
# export
yolo export model=yolo26n-sem.pt format=onnx imgsz=1024 batch=2 opset=17

# serialize
trtexec --onnx=yolo26n-sem.onnx --saveEngine=engines/yolov26n_sem.trt --noTF32
```

## yolov11

det · 640 · static batch = 2

```bash
# export
yolo export model=yolo11n.pt format=onnx imgsz=640 batch=2 opset=17

# serialize
trtexec --onnx=yolo11n.onnx --saveEngine=engines/yolov11n.trt --noTF32

# quantized tiers (Q/DQ ONNX, see quantization.md)
trtexec --onnx=yolo11n_fp16.onnx --saveEngine=engines/yolov11n_fp16.trt --fp16
trtexec --onnx=yolo11n_int8.onnx --saveEngine=engines/yolov11n_int8.trt --int8
```

## yolov11_cls

cls · 224 · static batch = 2

```bash
# export
yolo export model=yolo11n-cls.pt format=onnx imgsz=224 batch=2 opset=17

# serialize
trtexec --onnx=yolo11n-cls.onnx --saveEngine=engines/yolov11n_cls.trt --noTF32

# quantized tiers
trtexec --onnx=yolo11n-cls_fp16.onnx --saveEngine=engines/yolov11n_cls_fp16.trt --fp16
trtexec --onnx=yolo11n-cls_int8.onnx --saveEngine=engines/yolov11n_cls_int8.trt --int8
```

## yolov11_obb

obb · 1024 · static batch = 2 · nc = 15

```bash
# export
yolo export model=yolo11n-obb.pt format=onnx imgsz=1024 batch=2 opset=17

# serialize
trtexec --onnx=yolo11n-obb.onnx --saveEngine=engines/yolov11n_obb.trt --noTF32

# quantized tiers
trtexec --onnx=yolo11n-obb_fp16.onnx --saveEngine=engines/yolov11n_obb_fp16.trt --fp16
trtexec --onnx=yolo11n-obb_int8.onnx --saveEngine=engines/yolov11n_obb_int8.trt --int8
```

## yolov11_pose

pose · 640 · static batch = 2 · 17 keypoints

```bash
# export
yolo export model=yolo11n-pose.pt format=onnx imgsz=640 batch=2 opset=17

# serialize
trtexec --onnx=yolo11n-pose.onnx --saveEngine=engines/yolov11n_pose.trt --noTF32

# quantized tiers
trtexec --onnx=yolo11n-pose_fp16.onnx --saveEngine=engines/yolov11n_pose_fp16.trt --fp16
trtexec --onnx=yolo11n-pose_int8.onnx --saveEngine=engines/yolov11n_pose_int8.trt --int8
```

## yolov11_seg

seg · 640 · static batch = 2 · two outputs

```bash
# export
yolo export model=yolo11n-seg.pt format=onnx imgsz=640 batch=2 opset=17

# serialize
trtexec --onnx=yolo11n-seg.onnx --saveEngine=engines/yolov11n_seg.trt --noTF32

# quantized tiers
trtexec --onnx=yolo11n-seg_fp16.onnx --saveEngine=engines/yolov11n_seg_fp16.trt --fp16
trtexec --onnx=yolo11n-seg_int8.onnx --saveEngine=engines/yolov11n_seg_int8.trt --int8
```

## yolov8

det · 640 · static batch = 2

```bash
# export
yolo export model=yolov8n.pt format=onnx imgsz=640 batch=2 opset=17

# serialize
trtexec --onnx=yolov8n.onnx --saveEngine=engines/yolov8n.trt --noTF32
```

## yolov8_cls

cls · 224 · static batch = 2

```bash
# export
yolo export model=yolov8n-cls.pt format=onnx imgsz=224 batch=2 opset=17

# serialize
trtexec --onnx=yolov8n-cls.onnx --saveEngine=engines/yolov8n_cls.trt --noTF32
```

## yolov8_obb

obb · 1024 · static batch = 2 · nc = 15

```bash
# export
yolo export model=yolov8n-obb.pt format=onnx imgsz=1024 batch=2 opset=17

# serialize
trtexec --onnx=yolov8n-obb.onnx --saveEngine=engines/yolov8n_obb.trt --noTF32
```

## yolov8_pose

pose · 640 · static batch = 2 · 17 keypoints

```bash
# export
yolo export model=yolov8n-pose.pt format=onnx imgsz=640 batch=2 opset=17

# serialize
trtexec --onnx=yolov8n-pose.onnx --saveEngine=engines/yolov8n_pose.trt --noTF32
```

## yolov8_seg

seg · 640 · static batch = 2 · two outputs

```bash
# export
yolo export model=yolov8n-seg.pt format=onnx imgsz=640 batch=2 opset=17

# serialize
trtexec --onnx=yolov8n-seg.onnx --saveEngine=engines/yolov8n_seg.trt --noTF32
```

## yolov8_old

The legacy detection implementation, kept for A/B comparison. It consumes the
**dynamic** engine, not the static one above — and it exports to the same
`yolov8n.onnx` stem, so build one, move it aside, then build the other.

```bash
# export (dynamic batch)
yolo export model=yolov8n.pt format=onnx imgsz=640 batch=1 dynamic=True opset=17

# serialize
trtexec --onnx=yolov8n.onnx \
        --saveEngine=engines/yolov8n_old.trt \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:2x3x640x640 \
        --maxShapes=images:4x3x640x640
```

## yolov8_pose_old

Legacy pose implementation, dynamic engine (same ONNX stem as the static build).

```bash
# export (dynamic batch)
yolo export model=yolov8n-pose.pt format=onnx imgsz=640 batch=1 dynamic=True opset=17

# serialize
trtexec --onnx=yolov8n-pose.onnx \
        --saveEngine=engines/yolov8n_pose_old.trt \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:2x3x640x640 \
        --maxShapes=images:4x3x640x640
```

## yolov8_seg_old

Legacy segmentation implementation, dynamic engine (same ONNX stem as the static build).

```bash
# export (dynamic batch)
yolo export model=yolov8n-seg.pt format=onnx imgsz=640 batch=1 dynamic=True opset=17

# serialize
trtexec --onnx=yolov8n-seg.onnx \
        --saveEngine=engines/yolov8n_seg_old.trt \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:2x3x640x640 \
        --maxShapes=images:4x3x640x640
```

## yolov9

det · 640 · static batch = 2 · `yolov9t` is the smallest detection checkpoint ultralytics ships

```bash
# export
yolo export model=yolov9t.pt format=onnx imgsz=640 batch=2 opset=17

# serialize
trtexec --onnx=yolov9t.onnx --saveEngine=engines/yolov9t.trt --noTF32
```

## yolov9_seg

seg · 640 · static batch = 2 · two outputs — only `yolov9c-seg` and
`yolov9e-seg` exist, so the engine is named after the `c` checkpoint

```bash
# export
yolo export model=yolov9c-seg.pt format=onnx imgsz=640 batch=2 opset=17

# serialize
trtexec --onnx=yolov9c-seg.onnx --saveEngine=engines/yolov9c_seg.trt --noTF32
```

## yolov10

det · 640 · static batch = 2

> Only the **one2many** export works: `[B, 84, 8400]`. The official release ONNX
> is **end2end** (`[B, 300, 6]`, NMS inside the graph) and is not supported —
> the `expectC = 4 + nc + extra` check rejects it at startup (300 != 84).
> `yolo export` emits one2many, because `Detect.fuse()` drops `one2one_cv2` and
> `end2end` then stays False.

```bash
# export
yolo export model=yolov10n.pt format=onnx imgsz=640 batch=2 opset=17

# serialize
trtexec --onnx=yolov10n.onnx --saveEngine=engines/yolov10n.trt --noTF32
```

## yolov3

det · 640 · dynamic batch — the ONNX is not an ultralytics export: build it with
the model's own repo and place it here as `alpha_yolov3-spp.onnx`

```bash
# export: from the Darknet/YOLOv3 repo, not ultralytics

# serialize
trtexec --onnx=alpha_yolov3-spp.onnx \
        --saveEngine=engines/alpha_yolov3_spp.trt \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:2x3x640x640 \
        --maxShapes=images:4x3x640x640
```

## yolov4

det · 608 · dynamic batch — ONNX from the YOLOv4 repo, not ultralytics

```bash
# export: from the Darknet/YOLOv4 repo

# serialize
trtexec --onnx=alpha_yolov4_-1_3_608_608_dynamic.onnx \
        --saveEngine=engines/alpha_yolov4_1_3_608_608_dynamic.trt \
        --minShapes=input:1x3x608x608 \
        --optShapes=input:2x3x608x608 \
        --maxShapes=input:4x3x608x608
```

## yolov5

det · 640 · dynamic batch — ONNX from the YOLOv5 repo, not the ultralytics CLI
(this repo's `yolov5` uses the legacy preprocess, not the fused one)

```bash
# export: from the YOLOv5 repo, or use a prebuilt yolov5n.onnx

# serialize
trtexec --onnx=yolov5n.onnx \
        --saveEngine=engines/yolov5n.trt \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:2x3x640x640 \
        --maxShapes=images:4x3x640x640
```

## yolov6

det · 640 · dynamic batch — ONNX from the YOLOv6 repo, not ultralytics

```bash
# export: from the YOLOv6 repo

# serialize
trtexec --onnx=yolov6n.onnx \
        --saveEngine=engines/yolov6n.trt \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:2x3x640x640 \
        --maxShapes=images:4x3x640x640
```

## yolov7

det · 640 · dynamic batch — ONNX from the YOLOv7 repo, not ultralytics

```bash
# export: from the YOLOv7 repo

# serialize
trtexec --onnx=yolov7-tiny.onnx \
        --saveEngine=engines/yolov7_tiny.trt \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:2x3x640x640 \
        --maxShapes=images:4x3x640x640
```

## yolox

det · 640 — ONNX from the YOLOX repo, not ultralytics (static shapes in the graph)

```bash
# export: from the YOLOX repo

# serialize
trtexec --onnx=yolox_s.onnx --saveEngine=engines/yolox_s.trt
```

## yolor

det · 640 · dynamic batch — ONNX from the YOLOR repo, not ultralytics

```bash
# export: from the YOLOR repo

# serialize
trtexec --onnx=yolor_csp.onnx \
        --saveEngine=engines/yolor_csp.trt \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:2x3x640x640 \
        --maxShapes=images:4x3x640x640
```

## yolo_nas

det · 640 · dynamic batch — ONNX from the SuperGradients repo, not ultralytics

```bash
# export: from SuperGradients (yolo_nas_s)

# serialize
trtexec --onnx=yolonas_s.onnx \
        --saveEngine=engines/yolonas_s.trt \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:2x3x640x640 \
        --maxShapes=images:4x3x640x640
```

## efficientdet

det · 512 · **NHWC** input (the only model in the repo that is not NCHW) ·
dynamic batch

```bash
# export: from the EfficientDet repo

# serialize -- note the input name `input` and the NHWC axis order
trtexec --onnx=efficientdet0.onnx \
        --saveEngine=engines/efficientdet0.trt \
        --minShapes=input:1x512x512x3 \
        --optShapes=input:2x512x512x3 \
        --maxShapes=input:4x512x512x3
```

## yunet

face detection · dynamic shapes (`1x3x120x120` … `4x3x2000x2000`) — ONNX from
the OpenCV Zoo, not ultralytics

```bash
# export: from the OpenCV Zoo (the `_dynamic_simplify` variant)

# serialize
trtexec --onnx=alpha_yunet_yunet_final_dynamic_simplify.onnx \
        --saveEngine=engines/alpha_yunet_yunet_final_dynamic_simplify.trt \
        --minShapes=input:1x3x120x120 \
        --optShapes=input:2x3x320x320 \
        --maxShapes=input:4x3x2000x2000
```

## u2net

salient-object segmentation · 320 · dynamic batch — ONNX from the U-2-Net repo

```bash
# export: from the U-2-Net repo

# serialize
trtexec --onnx=u2net.onnx \
        --saveEngine=engines/u2net.trt \
        --minShapes=images:1x3x320x320 \
        --optShapes=images:2x3x320x320 \
        --maxShapes=images:4x3x320x320
```

## Dynamic batch

The `*_old` and legacy sections above pass explicit profiles. The same form
works for any model whose ONNX has a dynamic batch axis:

```bash
trtexec --onnx=yolo26n.onnx --saveEngine=engines/yolov26n_dyn.trt \
        --minShapes=images:1x3x640x640 \
        --optShapes=images:2x3x640x640 \
        --maxShapes=images:4x3x640x640
```

The INI's `batch_size` must then fall inside `[min, max]`; anything outside
aborts at startup instead of being clamped.

## Precision tiers

`--noTF32` above is the FP32 accuracy baseline. The other two tiers are engine
flags on the **same** ONNX — except INT8, which needs a Q/DQ graph:

| flag | needs | note |
|---|---|---|
| `--fp16` | any ONNX | default speed-up tier |
| `--int8` | ONNX with `QuantizeLinear` / `DequantizeLinear` | PTQ/QAT in PyTorch, see [quantization](quantization.md) |

`--fp16` and `--int8` are mutually exclusive; passing both is an error.

**FP32 is the only accuracy contract in this repository.** Every "matches
Python" number, and every engine checked into the validation matrix, refers to
the `--noTF32` build. FP16 is a single flag on the same ONNX. INT8 additionally
needs a Q/DQ graph from the PyTorch-side PTQ; the yolov11 sections above show
the form, and every other family works the same way — but quantization is
governed by the calibration set and per-node scales, so it is deliberately kept
out of the regression matrix. On this repository's own measurements INT8 is also
**not** faster than FP16 for the fused families, which makes FP16 the practical
choice if you want a speed tier at all.

## Point the model at the engine

Engines live in `engines/<name>.trt` and `configs/<net>.ini` carries
`engine = engines/<name>.trt`. Drop the file under the expected name, edit that
key, or override per run:

```bash
trt_alpha run --net yolov26 --source data/bus.jpg --engine engines/yolov26n_fp16.trt --show
```

A mismatch between the INI's `batch_size` and a static engine fails loudly at
startup, which is the intended behaviour — see [configuration](config.md).

## Rebuilding the whole set

The same commands, scripted for Linux, are in
[`scripts/linux/serialize_engines_linux.sh`](../scripts/linux/serialize_engines_linux.sh)
(`--only <substring>` builds a subset).
