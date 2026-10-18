# Command-line usage

```
trt_alpha v1.0 -- TensorRT inference framework

trt_alpha list
trt_alpha run   [options] --source <path|url|id>
trt_alpha bench [options]
trt_alpha build                       (not implemented yet)
```

`trt_alpha --help` prints the same synopsis.

## `list`

Prints every registered model. The name it prints is exactly what `--net` accepts, and it also determines the default INI (`configs/<net>.ini`).

```bash
trt_alpha list
```

Current build registers 26 models: `efficientdet`, `u2net`, `yolo_nas`, `yolor`, `yolo11` and its task family (`yolo11_cls`, `_obb`, `_pose`, `_seg`), `yolov3`-`yolov8`, `yolov8_pose`, `yolov8_seg`, `yolov26` and its task family (`yolov26_cls`, `_depth`, `_obb`, `_pose`, `_seg`, `_sem`), `yolox`, `yunet`.

## `run`

### Source kinds

`--source` is the **only** source option: the kind is detected from the string, so there is no `--image` / `--video` / `--camera` flag any more (passing one is an error).

| Value | Detected as |
|---|---|
| `data/bus.jpg` | single image |
| `data/` | image directory (one level deep, every image inferred separately) |
| `data/people.mp4` | local video file |
| `rtsp://host:554/s1` | network stream (`rtsp` / `rtmp` / `http(s)` / `udp`) |
| `0` | camera device id |

### Options

| Option | Meaning |
|---|---|
| `--source <value>` | input source, see the table above. Required |
| `--net <name>` | model name, default `yolov8`. Also picks up `configs/<name>.ini` |
| `--config <ini>` | explicit model INI, replacing `configs/<net>.ini` |
| `--engine <trt>` | override the INI's `engine=`. Wins over `--config` |
| `--batch <n>` | override the INI's `batch_size`. A static engine rejects any value other than its own |
| `--workers <n>` | inference-pool workers over the INI's `[pool].workers` |
| `--save [dir]` | save rendered frames. Without a value it goes to `save/<net>`; with a value that directory is used verbatim |
| `--show` | open the result window (title `<net>`-scoped, `trt_alpha:<net>`) |
| `--root <dir>` | override the project root used to resolve relative paths |

### Examples

```bash
# the YOLO26 task family
trt_alpha run --net yolov26       --source data/bus.jpg    --show
trt_alpha run --net yolov26_seg   --source data/bus.jpg    --save
trt_alpha run --net yolov26_pose  --source data/zidane.jpg --show
trt_alpha run --net yolov26_obb   --source data/boats.mp4  --save
trt_alpha run --net yolov26_cls   --source data/dog.jpg    --show
trt_alpha run --net yolov26_depth --source data/road0.jpg  --save
trt_alpha run --net yolov26_sem   --source data/road1.jpg  --save MY_SAVE_PATH

# every source kind
trt_alpha run --net yolov26 --source data/                            # image directory
trt_alpha run --net yolov26 --source data/people.mp4 --show           # local video
trt_alpha run --net yolov26 --source 0 --show                         # camera 0
trt_alpha run --net yolov26 --source rtsp://192.168.1.10:554/stream1  # network stream

# swap the engine / tune the batch / add workers
trt_alpha run --net yolov26 --source data/ --engine engines/yolov26n_fp16.trt --save
trt_alpha run --net yolov26 --source data/ --engine engines/yolov26n_int8.trt --save
# note: The batch parameter only uses dynamic batch, which is already decided when exporting ONNX 
trt_alpha run --net yolov26 --source data/ --batch 2 --workers 4 --save
```

## `bench`

Measures latency and throughput without any source: the frames are synthetic, so the result isolates engine + preprocess + postprocess. It reports per-stage timings plus p50 / p90 / p99 and FPS.

| Option | Meaning |
|---|---|
| `--net <name>` | model name, default `yolov8`; pulls `configs/<name>.ini` |
| `--config <ini>` | explicit model INI |
| `--engine <trt>` | override the INI's engine path |
| `--batch <n>` | override the INI's `batch_size` |
| `--iters <n>` | measured iterations, default `100` |
| `--warmup <n>` | warmup iterations, default `10` (dropped from the statistics) |
| `--src <WxH>` | source frame size, default = the engine's input size |
| `--root <dir>` | override the project root |

```bash
trt_alpha bench --net yolov26 --iters 100 --warmup 10
trt_alpha bench --engine engines/yolov26n_fp16.trt --batch 2 --iters 200 --warmup 20
```

At least one of `--net` / `--config` / `--engine` must be given.

## `build`

`build` (ONNX → engine serialization) is a placeholder and currently reports TODO. Serialize with `trtexec` for now, see [exporting and serializing](export_and_serialize.md).

## Behaviour worth knowing

- **Image sources always infer one frame at a time.** `--source data/bus.jpg` or `--source data/` degenerates to `batch=1` regardless of the engine's `batch_size`, so a batch=2 engine will not be fed two different-sized images. Video and camera sources do batch normally.
- **`--show` forces the render path to a single thread.** `cv::imshow` / `cv::waitKey` must live on one thread, so `--show` and `--save` together will not parallelize encoding.
- **`--engine` beats everything**: it overrides the INI's `engine=`, and the engine's declared shapes then override the INI's spatial dims.
- **`--save <dir>` is used verbatim**, no `save/<net>` subdirectory is appended; `--save` without a value produces `save/<net>`.
- Option values are validated before anything is loaded: an unknown option, a missing value, or a non-positive `--batch` / `--workers` aborts with a message instead of silently falling back to the INI.

Configuration keys (thresholds, queues, workers, JPEG quality, ...) live in the two INI levels, see [configuration](config.md).
