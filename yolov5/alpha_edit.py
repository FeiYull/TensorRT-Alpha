import argparse
import onnx
import onnx.helper as helper
import torch
# import torchvision
import onnxsim  # pip install onnx-simplifier
import onnxruntime as ort
import numpy as np
import os


def infer_onnx(onnx_file, input_names, image_input_shape):
    ort_session = ort.InferenceSession(onnx_file)
    outputs = ort_session.run(
        None,
        {input_names[0]: np.ones(shape=image_input_shape).astype(np.float32)},
    )
    return outputs


def run(mode, net_name, model_path):
    #mode = "p5"
    mode = mode

    if mode == "p5":
        #net_name = "yolov5m"
        net_name = net_name
        image_input_shape = [1, 3, 640, 640]
    else: # mode == "p6":
        #net_name = "yolov5m6"
        net_name = net_name
        image_input_shape = [1, 3, 1280, 1280]



    path = model_path
    onnx_name = net_name + ".onnx"
    input_names = ["images"]
    output_names = ["output"]

    model = onnx.load_model(path + onnx_name)

    outputs = infer_onnx(path + onnx_name, input_names, image_input_shape)
    for output in outputs:
        print(output.shape)

    # delete some nodes
    if mode == "p5":
        item1 = model.graph.output[1]
        item2 = model.graph.output[2]
        item3 = model.graph.output[3]
        model.graph.output.remove(item1)
        model.graph.output.remove(item2)
        model.graph.output.remove(item3)
    else: # mode == "p6":
        item1 = model.graph.output[1]
        item2 = model.graph.output[2]
        item3 = model.graph.output[3]
        item4 = model.graph.output[4]
        model.graph.output.remove(item1)
        model.graph.output.remove(item2)
        model.graph.output.remove(item3)
        model.graph.output.remove(item4)

    onnx.save(model, path + "alpha_" + onnx_name)
    outputs = infer_onnx(path + "alpha_" + onnx_name, input_names, image_input_shape)
    for output in outputs:
        print(output.shape)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='p5', help='p5:640*640, p6:1280*1280')
    parser.add_argument('--net_name', type=str, default='yolov5s', help='yolov5n yolov5s yolov5m ... yolov5s6 ...')
    parser.add_argument('--model_path', type=str, default='', help='pth file path')
    opt = parser.parse_args()
    #print_args(vars(opt))
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

