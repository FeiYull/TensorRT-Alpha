FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04
RUN sed -i 's#http://archive.ubuntu.com/#http://mirrors.tuna.tsinghua.edu.cn/#' /etc/apt/sources.list && \
    apt-get update

RUN apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    git \
    gdb \
    cmake \
    python3.8 \
    python3.8-dev \
    python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2 \
    && update-alternatives --config python3

#copy and unzip tensorrt8.4.2.4
RUN mkdir -p /home/feiyull/
COPY TensorRT-8.4.2.4.Linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz /home/feiyull/
RUN cd /home/feiyull/  && \
    tar -zxvf TensorRT-8.4.2.4.Linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz && \
    rm TensorRT-8.4.2.4.Linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz && \
    mkdir workspace

RUN \
    DEBIAN_FRONTEND=noninteractive apt-get install libgl1-mesa-glx -y \
    pkg-config \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libopencv-dev \
    && apt-get clean

# RUN pip3 install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# RUN pip install opencv-python-headless==4.8.0.74 && \
#     pip install opencv-python==4.8.0.74 \
#     pip install onnx==1.9.0 \
#     pip install torch==1.9.0 \
#     pip install torchvision==0.10.0 \
#     pip install onnx-simplifier==0.4.8

RUN cd /root/.cache/pip && \
    rm -r *