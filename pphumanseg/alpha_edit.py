import argparse
import onnx
import onnx.checker
import onnx.utils
from onnx.tools import update_model_dims
import onnx.helper as helper

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, default='../data/pphumanseg/human_segmentation_pphumanseg_2021oct.onnx', help='onnx path')
    opt = parser.parse_args()

    model = onnx.load(opt.onnx)

    in_b  = model.graph.input[0].type.tensor_type.shape.dim[0]
    in_c  = model.graph.input[0].type.tensor_type.shape.dim[1]
    in_h  = model.graph.input[0].type.tensor_type.shape.dim[2]
    in_w  = model.graph.input[0].type.tensor_type.shape.dim[3]

    out_loc_b        = model.graph.output[0].type.tensor_type.shape.dim[0]
    out_loc_num_candidates  = model.graph.output[0].type.tensor_type.shape.dim[1]
    out_loc_dim2       = model.graph.output[0].type.tensor_type.shape.dim[2] # 这个维度不修改

    in_b.dim_param= "batch_size"

    out_loc_b.dim_param = "batch_size"

    onnx.save(model, '../data/pphumanseg//human_segmentation_pphumanseg_2021oct_dynamic.onnx')
    print("ok")