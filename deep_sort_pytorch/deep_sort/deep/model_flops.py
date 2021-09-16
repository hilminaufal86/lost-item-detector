import sys
import argparse
import torch
import torch.nn as nn
from thop import profile, clever_format
from model import Net

model = Net(reid=True)
model_path = "checkpoint/ckpt.t7"
state_dict = torch.load(model_path, map_location=torch.device('cpu'))['net_dict']
model.load_state_dict(state_dict)

input = torch.randn(1, 3, 640, 640)
macs, params = profile(model, inputs=(input,),)
macs, params = clever_format([macs, params], '%.3f')
print(macs)
print(params)