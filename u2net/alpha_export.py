import argparse
import torch.nn
from model import U2NET
from model import U2NETP

import onnx
import numpy as np
import onnxsim  # pip install onnx-simplifier
import onnxruntime as ort
import numpy as np

class Alpha_U2Net(torch.nn.Module):
    def __init__(self, weight_file):
        super().__init__()
        self.model = U2NET(3, 1)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        
    def forward(self, x):
        y = self.model(x) 
        return y[0]

class Alpha_U2Netp(torch.nn.Module):
    def __init__(self, weight_file):
        super().__init__()
        self.model = U2NETP(3, 1)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        
    def forward(self, x):
        y = self.model(x) 
        return y[0]
"""
example:
python alpha_export.py --net=u2net --weights=saved_models/u2net/u2net.pth
python alpha_export.py --net=u2netp --weights=saved_models/u2netp/u2netp.pth
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='u2net', help='net type')
    parser.add_argument('--weights', type=str, default='saved_models/u2net/u2net.pth', help='net path')
    opt = parser.parse_args()

    net = ''
    image_input_shape = [1, 3, 320, 320]
    image_input = torch.autograd.Variable(torch.randn(image_input_shape))
    input_names = ["images"]
    output_names = ["output"]
    dynamic_axes = {"images": {0: "batch_size"}, "output": {0: "batch_size"}}

    net = opt.net
    if net=='u2net':  # for u2net.pt
        net_name = "u2net"
        onnx_name = net_name + ".onnx"
        model_path = opt.weights
        u2net = Alpha_U2Net(model_path)
        torch.onnx.export(u2net, image_input, "saved_models/onnx/" + onnx_name,
                        verbose=True,
                        input_names=input_names,
                        output_names=output_names,
                        opset_version=11,  # try  opset_version=9
                        training=False,
                        dynamic_axes=dynamic_axes)
    elif net=='u2netp':  # for u2netp.pt
        model_path = opt.weights
        u2netp = Alpha_U2Netp(model_path)
        torch.onnx.export(u2netp, image_input, "saved_models/onnx/u2netp.onnx",
                        verbose=True,
                        input_names=input_names,
                        output_names=output_names,
                        opset_version=11,
                        training=False,
                        dynamic_axes=dynamic_axes)
