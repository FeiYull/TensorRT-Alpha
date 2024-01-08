#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include<logger.h> // add file:   ../TensorRT-8.4.2.4/samples/common/logger.cpp
using namespace std;

int main() {
    // setting
    std::string onnx_file = "D:/ThirdParty/TensorRT-8.4.2.4/bin/yolov8n.onnx";
    std::string trt_file = "yolov8n.trt";
    int min_batchsize = 1;
    int opt_batchsize = 1;
    int max_batchsize = 2;
    nvinfer1::Dims4 min_shape(min_batchsize, 3, 640, 640);
    nvinfer1::Dims4 opt_shape(opt_batchsize, 3, 640, 640);
    nvinfer1::Dims4 max_shape(max_batchsize, 3, 640, 640);



    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger());
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger());
    if (!parser->parseFromFile(onnx_file.c_str(), 1)) {
        printf("Failed to parser demo.onnx\n");
        return false;
    }

    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
    config->setMaxWorkspaceSize(1 << 28);

    auto profile = builder->createOptimizationProfile();
    auto input_tensor = network->getInput(0);
    int input_channel = input_tensor->getDimensions().d[1];

    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, min_shape);
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, opt_shape);
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, max_shape);
    config->addOptimizationProfile(profile);

    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    if (engine == nullptr) {
        printf("Build engine failed.\n");
        return false;
    }
    nvinfer1::IHostMemory* model_data = engine->serialize();
    FILE* f = fopen(trt_file.c_str(), "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    model_data->destroy();
    parser->destroy();
    engine->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    printf("Done.\n");
    return true;
}
