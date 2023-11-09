#pragma once
// tensorrt
#include<logger.h>
#include<parserOnnxConfig.h>
#include<NvInfer.h>
// cuda
#include<cuda_runtime.h>
#include<stdio.h>
#include <thrust/sort.h>
#include<cuda_device_runtime_api.h>
#include<cuda_runtime.h>
#include<cuda_runtime_api.h>
#include<device_launch_parameters.h>
#include<device_atomic_functions.h>
// opencv
#include<opencv2/opencv.hpp>
// cpp std
#include<algorithm>
#include<cstdlib>
#include<fstream>
#include<iostream>
#include<sstream>
#include<vector>
#include<chrono>