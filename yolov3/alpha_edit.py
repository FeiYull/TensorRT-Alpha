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
        # {"data": np.ones(shape=image_input_shape).astype(np.float32)},
        {input_names[0]: np.ones(shape=image_input_shape).astype(np.float32)},
    )
    return outputs


net_name = "yolov3-tiny"
# net_name = "yolov3"
# net_name = "yolov3-spp"
path = "../data/yolov3/"

image_input_shape = [1, 3, 640, 640]
onnx_name = net_name + ".onnx"
input_names = ["images"]
output_names = ["output"]

model = onnx.load_model(path + onnx_name)

outputs = infer_onnx(path + onnx_name, input_names, image_input_shape)
for output in outputs:
    print(output.shape)

# delete some nodes
if net_name == "yolov3-tiny":
    item1 = model.graph.output[1]
    item2 = model.graph.output[2]
    model.graph.output.remove(item1)
    model.graph.output.remove(item2)
elif net_name == "yolov3" or net_name == "yolov3-spp":
    item1 = model.graph.output[1]
    item2 = model.graph.output[2]
    item3 = model.graph.output[3]
    model.graph.output.remove(item1)
    model.graph.output.remove(item2)
    model.graph.output.remove(item3)

onnx.save(model, path + "alpha_" + onnx_name)
outputs = infer_onnx(path + "alpha_" + onnx_name, input_names, image_input_shape)
for output in outputs:
    print(output.shape)
print("")