# Disabling rectangular inference (rect=False)

Every accuracy comparison in this repository is run with **rectangular inference disabled**. Here is why, and how to reproduce it on the ultralytics side.

## What rect inference changes

`ultralytics` pads the input to a **stride-aligned rectangle** when rectangular inference is on: an 1912x1080 frame becomes something like 1920x1088 rather than a square. This repository always pads to a **square** `imgsz x imgsz` (fill value 114), because a statically built engine only accepts one fixed shape.

The two schemes put the image at a different scale and a different offset inside the network input, so the decoded boxes land at different coordinates. This is not a rounding difference: on the same frame the depth task measured a **19.7 m** difference in depth, purely from the letterbox shape.

## Reproducing the official run

Pass `rect=False` so the reference run letterboxes exactly like the engine does:

```python
from ultralytics import YOLO

# model = YOLO("yolo26n-cls.pt")
model = YOLO("yolo26n-obb.pt")
# model = YOLO("yolo26n-pose.pt")
# model = YOLO("yolo26n-seg.pt")
# model = YOLO("yolo26n-depth.pt")

results = model(
    r"Data/boats.jpg",
    rect=False,     # close rectangle inference!
    save=True,
    project="./save",
    name="output",
    exist_ok=True,
)
```

## Rules of thumb

- **Keep it off for anything you intend to compare.** Boxes, masks, keypoints and depth maps are all affected by the letterbox shape, so a `rect=True` reference will disagree with the engine everywhere, not just at the borders.
- **It does not make the engine rectangular.** The engine's input shape comes from the ONNX and never changes at runtime; `rect` is purely a property of how the PyTorch-side input is prepared.
- **Semantic tasks are the loudest.** Segmentation and depth expose the padding offset directly in the output values, so they are the quickest way to notice a mismatched letterbox.
- Changing the resolution is a separate matter: that needs a new engine built at that size, see [exporting and serializing](export_and_serialize.md).
