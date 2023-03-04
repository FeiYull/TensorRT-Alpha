# Python 调用 [TensorRT-Alpha](https://github.com/FeiYull/TensorRT-Alpha) dll 教程

视频教程：[【TensorRT-Alpha 】dll封装python 调用教程](https://www.bilibili.com/video/BV1kM411x7f2)

本教程以yolov8为例，并假设：

1. 你已经对照[win10下 yolov8 tensorrt模型加速部署【实战】](https://blog.csdn.net/m0_72734364/article/details/128865904) 完整配置好所有必须环境，并能正确运行编译后的exe 进行推理。

   该教程将会用到：

   1. visual stuido 2019 
   2. cuda 11.2, tensorRT 8.4.2.4, opencv4.5.5  开发环境
   3. cuda, tensorRT, opencv 属性表
   4. yolov8 trt 文件（本教程以官方的yolov8n.pt 封装的trt文件为例）

2. 对python 有一定基础，知道如何运行python 程序，并安装对应的依赖

## Dll 动态库基础配置

### 创建项目 - 动态链接库(DLL) 

![image-20230302203606848](/.github/examples/python_with_dll/image-20230302203606848.png)

注意勾选 “将解决方案和项目放在同一目录中”

![image-20230302203807549](/.github/examples/python_with_dll/image-20230302203807549.png)

### 引入属性表

>  属性表创建请参考：[win10下 yolov8 tensorrt模型加速部署【实战】](https://blog.csdn.net/m0_72734364/article/details/128865904) 2.3 创建属性表

在之前的教程中，我们新建项目，并创建了 `CUDA 11.2` `TensorRT8.4.2.4_x64` `OpenCV4.5.5_DebugX64` `OpenCV4.5.5_ReleaseX64` 几个属性表。

首先将项目改为 Debug x64 模式，然后引入上述属性表，引入完成后资源管理器应该如下图所示

![image-20230302205149660](/.github/examples/python_with_dll/image-20230302205149660.png)

### 项目文件引入

#### TensorRT-Alpha 项目下 utils 及 yolov8 下必要文件添加

1. 将[TensorRT-Alpha](https://github.com/FeiYull/TensorRT-Alpha) 项目下 utils 和 yolov8 **文件夹复制**到visual stuido 创建的项目根目录
   
   强调下，这是文件夹之间的复制，不是在visual studio中进行“添加-添加现有项”

2. 删除根目录 utils和yolov8 目录下无用文件

   utils文件夹必须保留如下文件：

   ```
   common_include.h
   kernel_function.cu
   kernel_function.h
   utils.cpp
   utils.h
   yolo.cpp
   yolo.h
   ```

   yolov8文件夹必须保留如下文件：

   ```
   decode_yolov8.cu
   decode_yolov8.h
   yolov8.cpp
   yolov8.h
   ```

3. visual studio 引入文件

   + 点击切换此图标，查看复制过来的源文件列表

     ![image-20230302211219640](/.github/examples/python_with_dll/image-20230302211219640.png)

     ![image-20230302211258968](/.github/examples/python_with_dll/image-20230302211258968.png)

   + 全选这些文件，右键属性，将 “包括在项目中” 设置为 True

     ![image-20230302211446110](/.github/examples/python_with_dll/image-20230302211446110.png)

     修改完成后，会发现文件上的红色icon消失


#### tensorRT logger.cpp 文件引用

1. 切换默认视图，右键点击**资源文件** ，选择**属性** - 选择 **添加现有项**

   ![image-20230302212805461](/.github/examples/python_with_dll/image-20230302212805461.png)

2. 添加 `TensorRT-8.4.2.4\samples\common\logger.cpp`文件

   ![image-20230302213219151](/.github/examples/python_with_dll/image-20230302213219151.png)

   ![image-20230302213246167](/.github/examples/python_with_dll/image-20230302213246167.png)



至此，项目文件引入完成，项目应该包括如下文件：

![image-20230302213433177](/.github/examples/python_with_dll/image-20230302213433177.png)



### 添加CUDA 依赖项目

可点击[此处](https://www.bilibili.com/video/BV1xT411f72f?t=70.9)参考视频教程

![image-20230302214127422](/.github/examples/python_with_dll/image-20230302214127422.png)

![image-20230302214103308](/.github/examples/python_with_dll/image-20230302214103308.png)

### 为.cu 及对应头文件设置NVCC编译（注意配置顺序）

> 一定是先添加CUDA 依赖项目 再进行此步操作

可点击[此处](https://www.bilibili.com/video/BV1xT411f72f?t=79.7)参考视频教程

![image-20230302221408389](/.github/examples/python_with_dll/image-20230302221408389.png)



## Dll 封装与python 调用

### 更改引入文件的预编译头为 “不使用预编译头”，解决图中头文件预编译引起的报错
更改**所有引入的cpp文件**的预编译头为 “不使用预编译头”

![image-20230302220950777](/.github/examples/python_with_dll/image-20230302220950777.png)

###  编写Dll Api

该示例`dll`将暴露如下`api`

1. `Init `根据传入的trt 路径及额外参数初始化yolov8实例
2. `Detect` 对传入的图片文件进行预测，并记录bbox信息

#### 编写pch.cpp

复制`examples/python_with_dll/c_files/pch.cpp` 覆盖现有项目下的`pch.cpp `

#### 编写pch.h

复制`examples/python_with_dll/c_files/pch.h` 覆盖现有项目下的`pch.h`

### 生成dll

因为目前是debug模式，所以生成文件为 `C:\Users\shancw\source\repos\yoloDemo\x64\Debug\yoloDemo.dll`

![image-20230302221617892](/.github/examples/python_with_dll/image-20230302221617892.png)

**注意：从debug 模式切换为release 模式，需要重复  “为.cu 及对应头文件设置NVCC编译” 这步操作**

## python 调用

进入项目的`examples/python_with_dll`目录

### 添加dll文件

```
- python_with_dll
------ yoloDemo.dll （我们编译出来的dll）
------ opencv_world455.dll (opencv安装目录\build\x64\vc15\bin)
------ opencv_world455d.dll (opencv安装目录\build\x64\vc15\bin)
------ cudart64_110.dll  (C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin)
------ nvinfer.dll  (tensorRT安装目录\lib\nvinfer.dll)
```

### 添加 trt 文件

假设此处为`yolov8n.trt`，

>  trt 文件从之前教程编译得到

```
- python_with_dll
------ yoloDemo.dll （我们编译出来的dll）
------ opencv_world455.dll  (opencv安装目录\build\x64\vc15\bin)
------ opencv_world455d.dll  (opencv安装目录\build\x64\vc15\bin)
------ cudart64_110.dll  (C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin)
------ nvinfer.dll  (tensorRT安装目录\lib\nvinfer.dll)
------ yolov8n.trt  (trt 文件)
```

注意：如果你的 trt 文件以及编译出来的dll名称不一致，那么需要进行修改
![images-20230304121452.png](/.github/examples/python_with_dll/images-20230304121452.png)

### 运行python

最终目录如下

```
- python_with_dll
------ yoloDemo.dll （我们编译出来的dll）
------ opencv_world455.dll  (opencv安装目录\build\x64\vc15\bin)
------ opencv_world455d.dll  (opencv安装目录\build\x64\vc15\bin)
------ cudart64_110.dll  (C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin)
------ nvinfer.dll  (tensorRT安装目录\lib\nvinfer.dll)
------ yolov8n.trt  (trt 文件)
------ python_trt.py
------ config
-------- screen_inf.py
```

1. 安装依赖包

   ```
   pip install opencv-python numpy pygame mss pywin32
   ```

2. 运行 (需要在python_trt.py 同级目录)

   ```
   python python_trt.py
   ```

   