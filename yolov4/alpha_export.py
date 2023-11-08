import sys
import torch
from tool.darknet2pytorch import Darknet

class AlphaYolov4(torch.nn.Module):
    def __init__(self, cfgfile, weightfile):
        super().__init__()
        self.model = Darknet(cfgfile)
        self.model.load_weights(weightfile)
        self.model.eval()
        self.model.print_network()
        
    def forward(self, x):
        y = self.model(x) 
        boxes = y[0]
        confs = y[1].unsqueeze(dim = 2)
        return torch.cat((boxes, confs), 3)

def transform_to_onnx(cfgfile, weightfile, batch_size=1, onnx_file_name=None):
    model = AlphaYolov4(cfgfile, weightfile)

    dynamic = False
    if batch_size <= 0:
        dynamic = True

    input_names = ["input"]
    output_names = ['output']

    if dynamic:
        x = torch.randn((1, 3, model.model.height, model.model.width), requires_grad=True)
        if not onnx_file_name:
            onnx_file_name = "yolov4_-1_3_{}_{}_dynamic.onnx".format(model.model.height, model.model.width)
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes)

        print('Onnx model exporting done')
        return onnx_file_name

    else:
        x = torch.randn((batch_size, 3, model.model.height, model.model.width), requires_grad=True)
        onnx_file_name = "yolov4_{}_3_{}_{}_static.onnx".format(batch_size, model.model.height, model.model.width)
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=None)

        print('Onnx model exporting done')
        return onnx_file_name


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('weightfile')
    parser.add_argument('--batch_size', type=int, help="Static Batchsize of the model. use batch_size<=0 for dynamic batch size")
    parser.add_argument('--onnx_file_path', help="Output onnx file path")
    args = parser.parse_args()
    transform_to_onnx(args.config, args.weightfile, args.batch_size, args.onnx_file_path)

