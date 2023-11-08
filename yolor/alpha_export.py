import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from utils.google_utils import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from models.models import *
from utils.datasets import *
from utils.general import *

import argparse

import torch
from utils.google_utils import attempt_download


import onnx
import onnxruntime as ort
import numpy as np

"""
example:
python alpha_export.py --net=yolor_p6
python alpha_export.py --net=yolor_csp
python alpha_export.py --net=yolor_csp_star
python alpha_export.py --net=yolor_csp_x
python alpha_export.py --net=yolor_csp_x_star
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='yolor_p6', help='net type')
    opt = parser.parse_args()
    # init
    image_input_shape = ''
    img = ''
    model = ''

    net = opt.net 
    if net == "yolor_p6":
        # yolor_p6
        image_input_shape = (1, 3, 1280, 1280)
        img = torch.ones(image_input_shape)  # image size(1,3,320,192) iDetection
        model = Darknet("cfg/yolor_p6.cfg", 1280).cpu()
        opt.weights = 'yolor_p6.pt'
    elif net == "yolor_csp":
        # yolor_csp
        image_input_shape = (1, 3, 640, 640)
        img = torch.ones(image_input_shape)  # image size(1,3,320,192) iDetection
        model = Darknet("cfg/yolor_csp.cfg", 640).cpu()
        opt.weights = 'yolor_csp.pt'
    elif net == "yolor_csp_star":
        # yolor_csp_star
        image_input_shape = (1, 3, 640, 640)
        img = torch.ones(image_input_shape)  # image size(1,3,320,192) iDetection
        model = Darknet("cfg/yolor_csp.cfg", 640).cpu()
        opt.weights = 'yolor_csp_star.pt'
    elif net == "yolor_csp_x":
        # yolor_csp_x:
        image_input_shape = (1, 3, 640, 640)
        img = torch.ones(image_input_shape)  # image size(1,3,320,192) iDetection
        model = Darknet("cfg/yolor_csp_x.cfg", 640).cpu()
        opt.weights = 'yolor_csp_x.pt'
    elif net == "yolor_csp_x_star":
        # yolor_csp_x_star: 640*640
        image_input_shape = (1, 3, 640, 640)
        img = torch.ones(image_input_shape)  # image size(1,3,320,192) iDetection
        model = Darknet("cfg/yolor_csp_x.cfg", 640).cpu()
        opt.weights = 'yolor_csp_x_star.pt'

    model.load_state_dict(torch.load(opt.weights, map_location="cpu")['model'])

    model.eval()
    y = model(img)  # dry run
    print(y[0][0][0][0:10])

    # ONNX export
    # try
    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    f = opt.weights.replace('.pt', '.onnx')  # filename
    torch.onnx.export(model, img, f, verbose=False, opset_version=11, input_names=['images'], output_names=['output'],
                      dynamic_axes={
                'images': {
                    0: 'batch',
                    2: 'height',
                    3: 'width'},  # shape(1,3,640,640)
                'output': {
                    0: 'batch',
                    1: 'anchors'}  # shape(1,25200,85)
            })

    # Checks
    onnx_model = onnx.load(f)  # load onnx model

    input_names = ("images")
    ort_session = ort.InferenceSession(f)
    outputs = ort_session.run(
        None,
        {input_names: np.ones(shape=image_input_shape).astype(np.float32)},
    )
    print(outputs[0][0][0][0:10])
    onnx.checker.check_model(onnx_model)  # check onnx model
    print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
    print('ONNX export success, saved as %s' % f)

    if net == "yolor_p6":   
        item1 = onnx_model.graph.output[1]
        item2 = onnx_model.graph.output[2]
        item3 = onnx_model.graph.output[3]
        item4 = onnx_model.graph.output[4]
        onnx_model.graph.output.remove(item1)
        onnx_model.graph.output.remove(item2)
        onnx_model.graph.output.remove(item3)
        onnx_model.graph.output.remove(item4)
    else:
        item1 = onnx_model.graph.output[1]
        item2 = onnx_model.graph.output[2]
        item3 = onnx_model.graph.output[3]
        onnx_model.graph.output.remove(item1)
        onnx_model.graph.output.remove(item2)
        onnx_model.graph.output.remove(item3)

    # save
    onnx.save(onnx_model, f)
    # Finish
    print('\nExport complete. Visualize with https://github.com/lutzroeder/netron.')
