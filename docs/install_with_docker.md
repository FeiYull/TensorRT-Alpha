# Docker

`docker/Dockerfile` downloads and builds CUDA + TensorRT + OpenCV + FFmpeg + `trt_alpha` inside one image, so
nothing has to be installed on the host. Verified result: ~9.5 GB image (9.34 GB before FFmpeg was added),
`list` returns 21 models on a 5060 Ti.

The image is built with `-DTRT_ALPHA_WITH_FFMPEG=ON`, i.e. `--decode ffmpeg` and `--hwaccel nvdec` work out of the
box. FFmpeg is a **hard runtime dependency** of that build, which is why `/opt/ffmpeg` is copied into the run stage
and registered in `ld.so.conf`.

## 0. Prerequisites

```bash
# Docker Engine (USTC mirror; download.docker.com is usually unreachable in CN)
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://mirrors.ustc.edu.cn/docker-ce/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.asc
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.asc] \
https://mirrors.ustc.edu.cn/docker-ce/linux/ubuntu noble stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update && sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker "$USER"

# NVIDIA Container Toolkit (needed for --gpus all)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

docker run --rm --gpus all ubuntu:24.04 nvidia-smi
```

## 1. Build

Run from the repository root. `.dockerignore` has to be there too - Docker ignores the copy inside `docker/`.

```bash
cd /home/feiyull/workspace/TensorRT-Alpha
docker build -f docker/Dockerfile -t trt-alpha:latest .
```

~23 min from scratch (the 8 GB TensorRT download is most of it), ~30 s when only the source changed.

Build arguments (all optional):

| Argument | Default | Meaning |
|---|---|---|
| `OPENCV_VERSION` | `4.8.1` | OpenCV source tag |
| `TRT_VERSION` | `10.16.1.11` | TensorRT archive name |
| `TRT_DIR` | `10.16.1` | TensorRT download directory (three fields) |
| `CUDA_VERSION` | `12.9.1` | CUDA tag component |
| `UBUNTU_TAG` | `ubuntu24.04` | CUDA image tag suffix |
| `FFMPEG_BUILD` | `ffmpeg-n9.0-latest-linux64-gpl-shared-9.0` | BtbN asset name, pinned to the same `n9.0` as the Windows package |
| `JOBS` | `16` | parallel compile jobs |
| `WITH_BUILDER` | `0` | `1` keeps `libnvinfer_builder_resource_*.so` (+2.9 GB) |
| `CUDA_ARCH` | `75;86;89;120` | passed as `-DTRT_ALPHA_CUDA_ARCH` |

```bash
docker build -f docker/Dockerfile --build-arg JOBS=32 --build-arg WITH_BUILDER=1 -t trt-alpha:full .

# single stage, for debugging
docker build -f docker/Dockerfile --target opencv  -t trt-alpha-ocv .
docker build -f docker/Dockerfile --target ffmpeg  -t trt-alpha-ff  .
```

## 2. Run

```bash
docker run --rm --gpus all trt-alpha:latest list

docker run --rm --gpus all \
  -v "$PWD/engines:/app/engines:ro" -v "$PWD/save:/app/save" \
  trt-alpha:latest run --net yolov26 --source data/bus.jpg --save

# FFmpeg backend + NVDEC
docker run --rm --gpus all \
  -v "$PWD/engines:/app/engines:ro" -v "$PWD/save:/app/save" \
  trt-alpha:latest run --net yolov26 --source data/people.mp4 \
      --decode ffmpeg --hwaccel nvdec --save

docker run --rm -it --gpus all --entrypoint bash trt-alpha:latest
```

`engines/` and `onnx/` are not in the image, mount your own. `--gpus all` is required, otherwise `trtexec` prints
`CUDA driver version is insufficient for CUDA runtime version`. Add `--user "$(id -u):$(id -g)"` to keep
`save/` outputs owned by you.

NVDEC additionally needs the `video` driver capability, which the image already requests
(`NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`). If you override that variable, keep `video`, otherwise
`av_hwdevice_ctx_create` fails and `--hwaccel nvdec` reports the CUDA device as unavailable.

Verify the FFmpeg side inside a container:

```bash
docker run --rm --gpus all --entrypoint bash trt-alpha:latest -lc \
  'ldd /app/build/bin/Release/trt_alpha | grep -E "avcodec|avformat|not found"; \
   /app/build/bin/Release/trt_alpha --help | grep -- --decode'
```

Images, image directories and cameras still go through OpenCV even with `--decode ffmpeg`.

## 3. Clean up

```bash
docker image rm trt-alpha:latest
docker builder prune        # ~20 GB build cache, including the TensorRT download
```
