import sys
import os
import argparse
import torch
import torch.nn as nn
from thop import profile, clever_format
from model import Net
import psutil

process = psutil.Process(os.getpid())
model = Net(reid=True)
model_path = "checkpoint/ckpt.t7"
state_dict = torch.load(model_path, map_location=torch.device('cpu'))['net_dict']
model.load_state_dict(state_dict)
memory_used = 'Memory usage for model load (%.3fMB)' % (process.memory_info().rss / 1024 ** 2)
input = torch.randn(1, 3, 640, 640)
macs, params = profile(model, inputs=(input,),)
macs, params = clever_format([macs, params], '%.3f')
print(macs)
print(params)
print(memory_used)