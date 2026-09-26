# test_pool 环境搭建与 InferencePool 设计变更

> 记录日期：2026-09-18
> 涉及范围：CMake 构建链、`trt_alpha_core` 库、`InferencePool` 设计、test_pool 测试
> 目的：留档"为什么这么配"，避免后来者（包括未来的自己）重踩坑

---

## 1. 背景：test_pool 从"编不过"到"7 场景全 PASS"

起点：`test/test_pool/` 目录存在但为空。`test/CMakeLists.txt` 里有
`add_subdirectory(test_pool)`，但该目录下无 `CMakeLists.txt`，cmake configure 直接失败。

终点：`test_pool_scenarios.exe` 7 个场景全部 PASS，`test_pool.exe` happy path
用真 yolov8 引擎跑通。

一路上踩的坑**不是 test_pool 独有**，而是整个项目的构建链缺了几个关键配置。
本文件把每个坑、每个修复都记下来。

---

## 2. 构建链修复（顶层 CMakeLists.txt）

### 2.1 `project(... LANGUAGES CXX)` 缺 CUDA

**症状**：`.cu` 文件里的 kernel 函数（`transposeAnchors` / `decodeYoloV8Head` /
`nmsFast` / `resizeLetterbox` / `bgrToNchwNormalized`）在链接期全部报
`LNK2019: unresolved external symbol`。

**根因**：`project()` 只声明了 `CXX`，CMake 不知道 `.cu` 该用 nvcc 编。
即便 GLOB 把 `.cu` 加进源列表，CMake 也会把它当普通 C++ 文件处理，导致符号
被 C++ 编译器 mangling 后与 `.cpp` 侧不匹配。

**修复**：
```cmake
project(TensorRT_Alpha LANGUAGES CXX CUDA)

```









### CUDA 架构自动探测（2026-09-18 收尾）

**问题**：CMake 在 `project(... CUDA)` 时隐式给 `CMAKE_CUDA_ARCHITECTURES`
设默认值（CUDA 12.9 下是 `52` = Maxwell），导致：
  * 用户显式指定无效（`if(NOT DEFINED ...)` 被隐式默认值骗过）
  * 编译出 sm_52 cubin，RTX 50 系运行时靠 PTX JIT

**修复**：用自定义变量 `TRT_ALPHA_CUDA_ARCH` 作为"用户是否指定"的唯一判据，
并 `set(... CACHE STRING "..." FORCE)` 强制覆盖 CMake 的隐式默认值。

**探测优先级**：
  1. `-DTRT_ALPHA_CUDA_ARCH=<arch>` 命令行
  2. `nvidia-smi --query-gpu=compute_cap` 探测本机
  3. 回退列表 `75;86;89;120`（Turing/Ampere 消费/Ada/Blackwell）

**验证**（RTX 5060 Ti）：
  * configure 输出：`-- [CUDA] arch detected from local GPU: 120`
  * cache：`CMAKE_CUDA_ARCHITECTURES:STRING=120`
  * 编译命令：`--generate-code=arch=compute_120,code=[compute_120,sm_120]`
  * test_pool happy path：**505 ms → 9.19 ms（55x）**

**收益**：消除了 PTX JIT 开销，这是 505 ms 里约 490 ms 的来源。

**注意**：
  * 用户显式指定用 `-DTRT_ALPHA_CUDA_ARCH=...`，不用 CMake 官方变量名
  * 删除了全局的 `set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)`，
    改由 `set_target_properties(trt_alpha_core ...)` 显式声明（target 级更精确）



## 10. fillResult 重构（2026-09-19）

### 目标
消除 InferencePool::runBatch 里的 dynamic_cast + 深拷贝。

### 改动
- 新建 `include/trt_alpha/core/batch_result.hpp`：
  BatchResult 从 inference_pool.hpp 独立出来，切断
  model.hpp ↔ inference_pool.hpp 的循环依赖。
- `IModel` 加纯虚 `fillResult(core::BatchResult&)`（非 const，move 语义）。
- det::IDetector / seg::ISegmentor / cls::IClassifier 各实现 fillResult，
  用 `std::move + clear()` 转移结果。
- `inference_pool.hpp`：
  * 删 dynamic_cast 两处
  * 删 `#include "det/detector.hpp"` 和 `seg/segmentor.hpp`
  * runBatch 末尾一行 `model.fillResult(result)`
- `BatchResult` 加 `classifications` 字段（为 cls 任务预留）。

### 收益
- 池跟 det/seg/cls 命名空间彻底解耦
- 加新任务类型（关键点等）不用改池
- batch=8/topK=300 时结果传递从 2400 次拷贝降为 1 次指针转移

### 验证
- Debug: 7/7 PASS
- (Release 待验证)

### 已知遗留
- `IModel::reset()` 现在多数情况是空操作（fillResult 已 move 走），
  保留作为通用契约，将来可考虑删除。