import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from pytorch_lightning.core.lightning import LightningModule

import sys
sys.path.append('../')
from model.hrnet.hrnet import get_hrnet
from dataset.transform import PCBEVTransform


class PCBEVModel(LightningModule):
    def __init__(self):
        super().__init__()
        cfg = '../model/hrnet/conf_file/m1112_b2_c12.yaml'
        self.backbone = get_hrnet(cfg, upsample=True)
        self.transform = PCBEVTransform()

    def load_weights(self, model_path, strict=True):
        checkpoint = torch.load(model_path, map_location='cpu')
        self.load_state_dict(checkpoint['state_dict'], strict=strict)

    def forward(self, x):
        # x, _ = self.transform(x, None)
        features = self.backbone(x)
        return features

model_path = '/mnt/yrfs/yanrong/pvc-34488cf7-703b-4654-9fe8-762a747bbc58/wangtiantian/project/deep-lidar-model/logs2/deepgrid1202_2/epoch=06-Accu=0.89.ckpt'

model = PCBEVModel()
model.load_weights(model_path)

model = model.cpu()
model.eval()
print("==> Begin to transfer:")

x = torch.randn(1, 8, 170, 270, requires_grad=True)
torch_out = model(x)

# Export the model
torch.onnx.export(model,                     # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "lidar_ground_new5.onnx", # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
#                   do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['data'],   # the model's input names
                  output_names = ['output'], # the model's output names
#                   dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
#                                 'output' : {0 : 'batch_size'}}
                 )

print('==> Successful transfered to onnx ~')
