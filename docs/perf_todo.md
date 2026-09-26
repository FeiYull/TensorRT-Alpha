# 性能优化待办（v1.1+）

## opencv_source.cpp
- [ ] `copyIntoBatch` 加 `img.isContinuous()` 分支：
      连续且 `v.stride == img.cols*3` 时用 `memcpy` 一整块；
      否则逐行。

## yolov8.cpp
- [ ] `m_inputStaging` 白分配了，`setBatch` 里没用。
      要么删掉，要么 H2D 前先拷进它（pinned 带宽高）。
- [ ] `infer()` 里 `setTensorAddress` 每次调，可移到 `init()`。

## pipeline.cpp
- [ ] `renderLoop` 的 `result.views[i]` 诊断日志（for 循环那段）
      在 Release 下是 `TRT_LOG_INFO`，会刷屏。考虑降级成 DEBUG。