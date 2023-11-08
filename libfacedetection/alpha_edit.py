import argparse
from pyexpat import model
from turtle import width
import onnx
import onnx.checker
import onnx.utils
from onnx.tools import update_model_dims
import onnx.helper as helper
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, default='yunet_yunet_final_dynamic_simplify.onnx', help='onnx path')
    opt = parser.parse_args()

    model = onnx.load(opt.onnx)
    in_b  = model.graph.input[0].type.tensor_type.shape.dim[0]
    in_c  = model.graph.input[0].type.tensor_type.shape.dim[1]
    in_h  = model.graph.input[0].type.tensor_type.shape.dim[2]
    in_w  = model.graph.input[0].type.tensor_type.shape.dim[3]
    # loc
    out_loc_b               = model.graph.output[0].type.tensor_type.shape.dim[0]
    out_loc_num_candidates  = model.graph.output[0].type.tensor_type.shape.dim[1]
    out_loc_dim2            = model.graph.output[0].type.tensor_type.shape.dim[2] 
    # conf
    out_conf_b              = model.graph.output[1].type.tensor_type.shape.dim[0]
    out_conf_num_candidates = model.graph.output[1].type.tensor_type.shape.dim[1]
    out_conf_dim2           = model.graph.output[1].type.tensor_type.shape.dim[2] 
    # iou
    out_iou_b               = model.graph.output[2].type.tensor_type.shape.dim[0]
    out_iou_num_candidates  = model.graph.output[2].type.tensor_type.shape.dim[1]
    out_iou_dim2            = model.graph.output[2].type.tensor_type.shape.dim[2] 
    in_b.dim_param= "batch_size"
    in_h.dim_param= "height"
    in_w.dim_param= "width"
    out_loc_b.dim_param = "batch_size"
    out_conf_b.dim_param= "batch_size"
    out_iou_b.dim_param = "batch_size"
    out_loc_num_candidates.dim_param  = "num_condidates"
    out_conf_num_candidates.dim_param = "num_condidates"
    out_iou_num_candidates.dim_param  = "num_condidates"

    onnx.save(model, 'alpha_yunet_yunet_final_dynamic_simplify.onnx')
    print("ok")

