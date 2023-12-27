from super_gradients.training import models
from super_gradients.common.object_names import Models
import torch
import numpy as np

class AlphaYoloNas(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.get(Models.YOLO_NAS_S, pretrained_weights="coco")
        # self.model = models.get(Models.YOLO_NAS_M, pretrained_weights="coco")
        # self.model = models.get(Models.YOLO_NAS_L, pretrained_weights="coco")
        self.model.eval()

    def forward(self, x):
        y = self.model(x)
        return torch.cat((y[0], y[1]), 2)

input_size = (1, 3, 640, 640)
onnx_input = torch.Tensor(np.zeros(input_size))

net = AlphaYoloNas()
input_names = ["images"]
output_names = ["output"]
dynamic_axes = {input_names[0]: {0: "batch_size"}, 
                output_names[0]: {0: "batch_size"}}

torch.onnx.export(net, onnx_input, "yolonas_s.onnx",
            #verbose=True,
            input_names=input_names,
            output_names=output_names,
            opset_version=12,
            dynamic_axes=dynamic_axes)