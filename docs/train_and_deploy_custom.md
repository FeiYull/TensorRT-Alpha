# Training and deploying a custom model

This page walks through the whole path for a model you trained yourself, using an
object detection model with **4 classes** as the example. Ultralytics is used for
training here; any other framework works the same way — only the export step in
section 2 differs.

The example produces `my_det_task_cls_num_4.{pt,onnx,trt,ini}` and ends with a
successful `trt_alpha run`.

## 1. Train

```python
from ultralytics import YOLO


def main():
    model = YOLO("yolo26n.pt")
    results = model.train(data="african-wildlife.yaml", epochs=100, imgsz=640)


if __name__ == '__main__':
    main()
```

When training finishes you get the weights file:

```
my_det_task_cls_num_4.pt
```

## 2. Export to ONNX

```python
from ultralytics import YOLO

model = YOLO(r"runs\detect\train-3\weights\my_det_task_cls_num_4.pt")
model.export(
    format="onnx",
    batch=2,        # this repository batches, so export with a static batch of 2
    imgsz=640,
    dynamic=False,
    simplify=True,
    opset=17,
    nms=None,       # keep the default one-to-many head: one-to-one loses a little
                    # accuracy, and this repository's decode + NMS is written in
                    # CUDA C, which is fast enough
)
```

This writes:

```
my_det_task_cls_num_4.onnx
```

## 3. Serialize with `trtexec`

```powershell
.\trtexec.exe --onnx=my_det_task_cls_num_4.onnx --saveEngine=my_det_task_cls_num_4.trt --noTF32
```

This writes:

```
my_det_task_cls_num_4.trt
```

## 4. Create the INI configuration

All paths below are relative to the repository root (`TensorRT-Alpha/`).

1. Copy `my_det_task_cls_num_4.trt` into `engines/`.

2. Create `data/classes/african-wildlife.txt` with the classes your model was
   trained on. This example has four:

   ```text
   buffalo 255 0 255
   elephant 255 0 0
   rhino 0 255 255
   zebra 0 255 0
   ```

   Each line is `name B G R`. The colour is **BGR**, i.e. it is used as-is when
   the result is drawn — `buffalo 255 0 255` means magenta.

3. Create `configs/my_det_task_cls_num_4.ini`:

   ```ini
   [model]
   engine = engines/my_det_task_cls_num_4.trt
   class_names_file = data/classes/african-wildlife.txt
   num_class = 4

   [input]
   layout = nchw
   batch_size = 2

   [postprocess]
   conf_thresh = 0.25
   iou_thresh = 0.7
   top_k = 300
   ```

   Adjust `engine`, `class_names_file`, `num_class` and `batch_size` to match your
   own model. `num_class` must equal the class count baked into the engine: a
   mismatch is rejected at startup (`output0 channel = N but 4+num_class+extra = M`),
   never silently clamped.

## 5. Run

> **Always pass `--config` when you load your own model.** `--net` only selects
> the model implementation and the *default* INI (`configs/<net>.ini`); it does not
> know about your class count. Everything else — the engine path included — can
> also live in the INI, so `--engine` is optional once `engine =` is set.

```powershell
.\build\bin\Release\trt_alpha.exe run --source E:/Code/datasets/african-wildlife/images/test --net yolov26 --config configs/my_det_task_cls_num_4.ini --engine engines/my_det_task_cls_num_4.trt --save
```

A successful run looks like this:

```text
(base) PS D:\VS2022_Project\TensorRT-Alpha> .\build\bin\Release\trt_alpha.exe run --source E:/Code/datasets/african-wildlife/images/test --net yolov26 --config configs/my_det_task_cls_num_4.ini --engine engines/my_det_task_cls_num_4.trt --save
[2026-10-19 19:21:36.219] [INFO ] [tid=7600] [paths.cpp:208 root] Paths: root = D:\VS2022_Project\TensorRT-Alpha  (source: executable location)
[2026-10-19 19:21:36.219] [INFO ] [tid=7600] [config.cpp:174 loadModelConfig] Config: loaded configs/my_det_task_cls_num_4.ini (with configs/base.ini) (engine=engines/my_det_task_cls_num_4.trt, batch=2, layout=NCHW, extras=34 entries)
[2026-10-19 19:21:36.220] [INFO ] [tid=7600] [config.cpp:264 loadClassNamesFile] Config: class names file data/classes/african-wildlife.txt (4 classes)
[2026-10-19 19:21:36.220] [INFO ] [tid=7600] [inference_pool.cpp:50 InferencePool] [InferencePool] constructing with 1 worker(s), engine=engines/my_det_task_cls_num_4.trt, batchSize=2, maxQueueSize=4
[2026-10-19 19:21:36.220] [INFO ] [tid=7600] [inference_pool.cpp:55 InferencePool] [InferencePool] loading shared engine: engines/my_det_task_cls_num_4.trt
[2026-10-19 19:21:36.220] [INFO ] [tid=7600] [engine.cpp:381 Engine] Engine: loading D:\VS2022_Project\TensorRT-Alpha\engines\my_det_task_cls_num_4.trt
[TRT INFO ] Loaded engine size: 12 MiB
[2026-10-19 19:21:36.331] [INFO ] [tid=7600] [engine.cpp:452 discoverIo] Engine: io[0] input  'images' fp32 [2, 3, 640, 640]
[2026-10-19 19:21:36.331] [INFO ] [tid=7600] [engine.cpp:462 discoverIo] Engine: io[0] input batch static, min/opt/max = 2/2/2
[2026-10-19 19:21:36.331] [INFO ] [tid=7600] [engine.cpp:452 discoverIo] Engine: io[1] output 'output0' fp32 [2, 8, 8400]
[2026-10-19 19:21:36.332] [INFO ] [tid=7600] [engine.cpp:410 Engine] Engine: loaded (2 io tensors)
[2026-10-19 19:21:36.332] [INFO ] [tid=7600] [yolov26.cpp:72 loadConfig] yolov26: config num_class=4 extra=0 conf=0.25 iou=0.7 top_k=300
[TRT INFO ] [MS] Running engine with multi stream info
[TRT INFO ] [MS] Number of aux streams is 3
[TRT INFO ] [MS] Number of total worker streams is 4
[TRT INFO ] [MS] The main stream provided by execute/enqueue calls is the first worker stream
[TRT INFO ] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +52, now: CPU 0, GPU 62 (MiB)
[2026-10-19 19:21:36.394] [INFO ] [tid=7600] [engine.cpp:372 applyInputShape] InputShape: 'images' layout=NCHW -> [2, 3, 640, 640]
[2026-10-19 19:21:36.395] [INFO ] [tid=7600] [yolov26.cpp:230 init] yolov26: initialized (batch=2, outC=8, anchors=8400, dst=640x640)
[2026-10-19 19:21:36.395] [INFO ] [tid=7600] [inference_pool.cpp:137 InferencePool] [InferencePool] worker[0] ready
[2026-10-19 19:21:36.395] [INFO ] [tid=7600] [inference_pool.cpp:147 InferencePool] [InferencePool] ready with 1 worker(s)
+===================================================================================================+
| [CONFIG] yolov26                                                configs/my_det_task_cls_num_4.ini |
| [model]                                                                                           |
|   engine            = engines/my_det_task_cls_num_4.trt                 CLI                       |
|   class_names_file  = data/classes/african-wildlife.txt                 my_det_task_cls_num_4.ini |
|   num_class         = 4                                                 my_det_task_cls_num_4.ini |
|                                                                                                   |
| [input]                                                                                           |
|   batch_size        = 2                                                 my_det_task_cls_num_4.ini |
|   layout            = nchw                                              my_det_task_cls_num_4.ini |
|                                                                                                   |
| [normalize]                                                                                       |
|   scale             = 255.0                                             base.ini                  |
|   mean              = 0.0,0.0,0.0                                       base.ini                  |
|   std               = 1.0,1.0,1.0                                       base.ini                  |
|   pad_value         = 114.0                                             base.ini                  |
|                                                                                                   |
| [postprocess]                                                                                     |
|   conf_thresh       = 0.25                                              my_det_task_cls_num_4.ini |
|   iou_thresh        = 0.7                                               my_det_task_cls_num_4.ini |
|   top_k             = 300                                               my_det_task_cls_num_4.ini |
|                                                                                                   |
| [pool]                                                                                            |
|   workers           = 1                                                 base.ini                  |
|   max_queue_size    = 4                                                 base.ini                  |
|   result_queue_size = 8                                                 base.ini                  |
|   render_threads    = 6                                                 base.ini                  |
|                                                                                                   |
| [output]                                                                                          |
|   jpeg_quality      = 95                                                base.ini                  |
|   save_dir          = save/yolov26                                      default                   |
|                                                                                                   |
| [resolved]   engine truth (shape = engine declaration; -1 = dynamic axis)                         |
|   batch             = 2                                                 (engine-confirmed)        |
|   in                = images   2 x 3 x 640 x 640   fp32 (4B)   linear                             |
|   out               = output0   2 x 8 x 8400   fp32 (4B)   linear                                 |
+===================================================================================================+
[2026-10-19 19:21:36.414] [INFO ] [tid=7600] [opencv_source.cpp:129 openImage] OpenCVSource: 'E:/Code/datasets/african-wildlife/images/test' is a directory -> images mode (single level)
[2026-10-19 19:21:36.415] [INFO ] [tid=7600] [opencv_source.cpp:165 openImages] OpenCVSource: found 227 image(s) in E:/Code/datasets/african-wildlife/images/test
[2026-10-19 19:21:36.415] [INFO ] [tid=7600] [opencv_source.cpp:96 OpenCVSource] OpenCVSource[images]: opened 'E:/Code/datasets/african-wildlife/images/test' batch=2 sourceId=0
[2026-10-19 19:21:36.422] [INFO ] [tid=7600] [pipeline.cpp:176 validateConfig] Pipeline: render_threads=6 (frame order on disk is not guaranteed)
[2026-10-19 19:21:36.422] [INFO ] [tid=7600] [pipeline.cpp:188 start] Pipeline: starting with 1 source(s) and 1 pool(s)
[2026-10-19 19:21:36.422] [INFO ] [tid=7600] [pipeline.cpp:200 start] Pipeline: launching 6 render thread(s)
[2026-10-19 19:21:36.423] [INFO ] [tid=7600] [pipeline.cpp:206 start] Pipeline: render thread(s) launched
[2026-10-19 19:21:36.423] [INFO ] [tid=7600] [pipeline.cpp:217 start] Pipeline: launching source thread 0
[2026-10-19 19:21:36.423] [INFO ] [tid=7600] [pipeline.cpp:220 start] Pipeline: all source threads launched
[2026-10-19 19:21:36.423] [INFO ] [tid=46752] [pipeline.cpp:338 renderLoop] Pipeline: render loop started, entering while
[2026-10-19 19:21:36.423] [INFO ] [tid=7600] [pipeline.cpp:225 start] Pipeline: started (1 source + 6 render = 7 threads)
[2026-10-19 19:21:36.423] [INFO ] [tid=48696] [pipeline.cpp:338 renderLoop] Pipeline: render loop started, entering while
[2026-10-19 19:21:36.423] [INFO ] [tid=31172] [pipeline.cpp:338 renderLoop] Pipeline: render loop started, entering while
[2026-10-19 19:21:36.423] [INFO ] [tid=49896] [pipeline.cpp:338 renderLoop] Pipeline: render loop started, entering while
[2026-10-19 19:21:36.423] [INFO ] [tid=49208] [pipeline.cpp:338 renderLoop] Pipeline: render loop started, entering while
[2026-10-19 19:21:36.423] [INFO ] [tid=34192] [pipeline.cpp:338 renderLoop] Pipeline: render loop started, entering while
[2026-10-19 19:21:36.423] [INFO ] [tid=24136] [pipeline.cpp:237 sourceLoop] Pipeline: source[0] thread entered
[2026-10-19 19:21:36.423] [INFO ] [tid=24136] [pipeline.cpp:243 sourceLoop] Pipeline: source[0] 'images' using pool[0]
[2026-10-19 19:21:36.434] [INFO ] [tid=46752] [opencv_renderer.cpp:979 save] OpenCVRenderer: saving into D:\VS2022_Project\TensorRT-Alpha\save\yolov26
[2026-10-19 19:21:36.434] [WARN ] [tid=46752] [opencv_renderer.cpp:1037 save] OpenCVRenderer::save: overwriting existing file(s), e.g. save/yolov26\1 (103).jpg (reported once per output dir)
[2026-10-19 19:21:37.188] [INFO ] [tid=24136] [pipeline.cpp:259 sourceLoop] Pipeline: source[0] next() returned false, breaking
[2026-10-19 19:21:37.188] [INFO ] [tid=24136] [pipeline.cpp:320 sourceLoop] Pipeline: source[0] exiting
[2026-10-19 19:21:37.194] [INFO ] [tid=49896] [pipeline.cpp:350 renderLoop] Pipeline: render loop pop() returned false, exiting
[2026-10-19 19:21:37.194] [INFO ] [tid=49896] [pipeline.cpp:494 renderLoop] Pipeline: render loop exiting
[2026-10-19 19:21:37.196] [INFO ] [tid=46752] [pipeline.cpp:350 renderLoop] Pipeline: render loop pop() returned false, exiting
[2026-10-19 19:21:37.196] [INFO ] [tid=46752] [pipeline.cpp:494 renderLoop] Pipeline: render loop exiting
[2026-10-19 19:21:37.198] [INFO ] [tid=48696] [pipeline.cpp:350 renderLoop] Pipeline: render loop pop() returned false, exiting
[2026-10-19 19:21:37.198] [INFO ] [tid=48696] [pipeline.cpp:494 renderLoop] Pipeline: render loop exiting
[2026-10-19 19:21:37.201] [INFO ] [tid=49208] [pipeline.cpp:350 renderLoop] Pipeline: render loop pop() returned false, exiting
[2026-10-19 19:21:37.201] [INFO ] [tid=49208] [pipeline.cpp:494 renderLoop] Pipeline: render loop exiting
[2026-10-19 19:21:37.204] [INFO ] [tid=31172] [pipeline.cpp:350 renderLoop] Pipeline: render loop pop() returned false, exiting
[2026-10-19 19:21:37.204] [INFO ] [tid=31172] [pipeline.cpp:494 renderLoop] Pipeline: render loop exiting
[2026-10-19 19:21:37.205] [INFO ] [tid=34192] [pipeline.cpp:350 renderLoop] Pipeline: render loop pop() returned false, exiting
[2026-10-19 19:21:37.205] [INFO ] [tid=34192] [pipeline.cpp:494 renderLoop] Pipeline: render loop exiting
[2026-10-19 19:21:37.205] [INFO ] [tid=7600] [pipeline.cpp:567 waitForCompletion] Pipeline: all threads joined
[2026-10-19 19:21:37.205] [INFO ] [tid=7600] [inference_pool.cpp:370 shutdown] [InferencePool] shutdown requested, queue=0, active=0
[2026-10-19 19:21:37.205] [INFO ] [tid=24208] [inference_pool.cpp:238 workerLoop] [InferencePool] worker[0] exiting
[2026-10-19 19:21:37.210] [INFO ] [tid=7600] [inference_pool.cpp:391 shutdown] [InferencePool] shutdown complete
(base) PS D:\VS2022_Project\TensorRT-Alpha>
```

The line that matters is `config num_class=4 ... outC=8` — `8 = 4 (xywh) + 4 (classes) + 0
(extra)`, so the engine and the INI agree. Results are written to `save/yolov26/`
(the directory comes from `--net`; override it with `[output] save_dir`).
