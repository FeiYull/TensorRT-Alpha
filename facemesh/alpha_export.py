import argparse
from facemesh import FaceMesh
import torch
import onnx
import onnx.checker
import onnx.utils
from onnx.tools import update_model_dims
import onnx.helper as helper

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='facemesh.pth', help='weights')
    opt = parser.parse_args()
    #*******************************************************************
    # 1. export onnx
    #*******************************************************************
    net = FaceMesh()
    net.load_weights(opt.weights)

    # opset_version=10
    torch.onnx.export(
        net, 
        torch.randn(1, 3, 192, 192), 
        "facemesh.onnx",
        input_names=("image", ), 
        output_names=("preds", "confs"), 
        opset_version=10,
        #do_constant_folding=False,
    )

    # opset_version=11, warning
    # opset_version=10, right

    #*******************************************************************
    # 2. change to dynamic model
    #*******************************************************************

    model = onnx.load("facemesh.onnx")

    # 获取输入dim
    in_b  = model.graph.input[0].type.tensor_type.shape.dim[0]
    in_c  = model.graph.input[0].type.tensor_type.shape.dim[1]
    in_h  = model.graph.input[0].type.tensor_type.shape.dim[2]
    in_w  = model.graph.input[0].type.tensor_type.shape.dim[3]

    # 修改输入
    in_b.dim_param= "batch_size"

    # 获取输出dim
    out1_b  = model.graph.output[0].type.tensor_type.shape.dim[0]
    out1_len  = model.graph.output[0].type.tensor_type.shape.dim[1]

    out2_b  = model.graph.output[1].type.tensor_type.shape.dim[0]
    out2_len  = model.graph.output[1].type.tensor_type.shape.dim[1]

    # 修改输出
    out1_b.dim_param= "batch_size"
    out2_b.dim_param= "batch_size"

    onnx.save(model, 'facemesh.onnx')
    print("ok")