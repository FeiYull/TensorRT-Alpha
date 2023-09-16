## 1. download tensorrt8.4.2.4
- enter： https://developer.nvidia.cn/nvidia-tensorrt-8x-download
- select： I Agree To the Terms of the NVIDIA TensorRT License Agreement
- select:   TensorRT 8.4 GA Update 1
- select:   TensorRT 8.4 GA Update 1 for Linux x86_64 and CUDA 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6 and 11.7 TAR Package
- download file:  "TensorRT-8.4.2.4.Linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz"

```bash
cd TensorRT-Alpha/docker
cp TensorRT-8.4.2.4.Linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz .
```

## 2. build docker images
```bash
docker build -f ubuntu18.04-cu113.Dockerfile --network=host -t trta .
```