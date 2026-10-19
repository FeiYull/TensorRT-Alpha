# Configuration

Two levels, **highest priority first**:

```
API call / command-line flags   >   configs/<model>.ini   >   configs/base.ini
```

- **`configs/base.ini`** — defaults shared by every model (normalization, postprocess thresholds, thread pool, output directory).
- **`configs/<model>.ini`** — per-model overrides (engine path, class file, batch, task-specific thresholds).
- **API call / command-line flags** — **highest priority**, applied at runtime and overriding both INI layers. `--engine` overrides `engine=`, `--batch` overrides `batch_size`, `--workers` overrides `[pool].workers`, `--config` swaps the whole model INI; the fields of the high-level `InferParams` behave the same way.

## base.ini

### `[input]`

| Key | Default | Meaning |
|---|---|---|
| `layout` | written explicitly in every `configs/<model>.ini` (`nchw`; `efficientdet` = `nhwc`); commented out here | Logical axis order of the input. Priority: `<model>.ini` > `base.ini` > the model's built-in canonical layout. Only change it when the same model is swapped onto an engine with a different layout |
| `batch_size` | `1` | Batch actually used for this run. A static engine requires it to **equal** the baked-in batch; a dynamic engine requires it to fall inside `[min, max]`. A mismatch is an **error and terminates** — never silently clamped |
| `max_batch_size` | not checked | Contract for the engine's batch upper bound; a mismatch fails at startup |
| `dst_h` / `dst_w` | — | **Normally leave these out**: the spatial dims come from the engine's declared shape. Nothing is printed when they are absent; a value that disagrees with a static engine dim only raises a warning and the engine shape wins (it never errors). They act as the intended value only when that engine dim is dynamic (`-1`) |

### `[normalize]`

| Key | Default | Meaning |
|---|---|---|
| `scale` | `255.0` | Normalization divisor. Honoured by every model |
| `mean` | `0.0,0.0,0.0` | Per-channel mean, comma separated. **Only the legacy preprocess honours it** (`yolov3`-`yolov7`, `yolov8_old`, `yolov8_pose_old`, `yolov8_seg_old`, `yolox`, `yolor`, `yolo_nas`, `u2net`). The fused preprocess (`yolov26`, `yolov11`, the new `yolov8` family — all tasks —, `yolov9`, `yolov10`, plus `depth` / `sem`) has no slot for it: a non-identity value fails at startup instead of being silently dropped |
| `std` | `1.0,1.0,1.0` | Per-channel std, same rule as `mean` |
| `pad_value` | `114.0` | Letterbox fill value (matches ultralytics' 114) |

### `[postprocess]`

| Key | Default | Meaning |
|---|---|---|
| `conf_thresh` | `0.25` | Confidence threshold (matches ultralytics `conf`) |
| `iou_thresh` | `0.45` | NMS IoU threshold (yolo26 uses `0.7`, see `configs/yolov26.ini`) |
| `top_k` | `300` | **Output cap** per image after NMS (matches ultralytics `max_det`). It only caps the output, it does not reduce the work |

### `[pool]`

| Key | Default | Meaning |
|---|---|---|
| `workers` | `1` | Inference-pool workers = `1 engine + N context`. Every extra worker costs another engine context and its device memory |
| `max_queue_size` | `16` | Inference task queue depth (one entry = one batch). When full it **blocks**, back-pressuring the source, so no task is dropped |
| `result_queue_size` | `32` | Result queue depth. When full it **drops the oldest** — a live view prefers dropping frames over falling behind |
| `render_threads` | `0` = auto | Render threads. auto = 1..8 from the CPU core count, and only when saving to disk; `--show` is always 1. Affects only `--save` encoding parallelism |

### `[output]`

| Key | Default | Meaning |
|---|---|---|
| `save_dir` | `save/<model>` | Output directory. `--save <dir>` wins (used verbatim, no model subdirectory appended) |
| `show_window` | `trt_alpha:<registered name>` | Used only with `--show`. The default carries the registered name so several models can be shown without fighting over one window |
| `jpeg_quality` | `95` | JPEG quality 1..100 for saved frames. q85 cuts size by 41% with no measurable time change |

### Reserved keys (writing them has no effect; the config box flags them `[unused]`)

`device_id` (multi-GPU; device 0 is always used), `policy` (queue policy is fixed per queue), and `block_1d` / `block_2d_x` / `block_2d_y` under `[kernels]`.

## `<model>.ini`

```ini
[model]
engine = engines/yolov26n.trt
class_names_file = data/classes/coco80.txt
num_class = 80
# extra: per-anchor extra channels (seg = 32 / pose = 51). 0 for det.
# extra = 0
# mask_orig: seg only. Also emit Segmentation::maskOrig (the source-sized mask).
# Off by default: it is one full-resolution bilinear resize per instance
# (~0.6 ms at 1912x1080) and the built-in renderer never reads it.
# mask_orig = false

[input]
batch_size = 2

[postprocess]
conf_thresh = 0.25
iou_thresh = 0.7
top_k = 300
```

Any field present in `<model>.ini` overrides the value from `base.ini`. Task-specific render parameters (`kpt_conf_thresh`, `mask_alpha`, ...) belong in the `[render]` section of `<model>.ini` and must **not** go into `base.ini`.

## Two traps

- **Keys are looked up as `section.key`** (`IniParser::makeKey`). `render_threads` must therefore live under `[pool]`; put it under another section and it is stored as, say, `output.render_threads` and **silently never read**. The startup config box flags unread keys as `[unused]`, which is how you notice. Note that `base.ini` keys an individual task has no use for (`conf_thresh` on `cls`, `top_k` on `depth`) show up there too — that is expected, only a key you wrote in `<model>.ini` is a mistake.
  The converse is guarded in code: a model whose preprocess cannot honour `mean` / `std` **fails at startup** instead of ignoring them (`core::requireScaleOnlyNorm`). A key that is accepted is a key that is used.
- **Do not write the spatial dims (H/W) into the INI.** The engine's declared shape is the single source of truth. Nothing is logged while `dst_h` / `dst_w` are absent; if you do write a value that disagrees with a static engine dim you get `config size ... != engine declared shape, using engine shape (...)` and the **engine** shape is used. That line is a warning, not an error — the run continues. Rebuild the engine if you need another resolution.
