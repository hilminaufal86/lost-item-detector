import os
import sys
import argparse
import torch
import torch.nn as nn
from thop import profile, clever_format
import psutil
import mish_cuda

def count_hardswish(m, x, y):
    x = x[0]
    nelements = x.numel()
    m.total_ops += torch.DoubleTensor([int(nelements)])

def count_mishcuda(m, x, y):
    x = x[0]
    nelements = x.numel() / 2
    m.total_ops += torch.DoubleTensor([int(nelements)])

yolov5_cust_ops = {
    nn.Hardswish: count_hardswish
}

syolov4_cust_ops = {
    mish_cuda.MishCuda: count_mishcuda
}


def run_eval(model_name='yolov5', weights='yolov5/weights/yolov5s_augmented.pt'):
    yolov5 = scaledyolov4 = yolov4csp = False
    if model_name=='yolov5':
        yolov5 = True
    elif model_name=='yolov4csp':
        yolov4csp=True
    elif model_name=='scaledyolov4':
        scaledyolov4=True
    else:
        print("please select one model only")
        return

    process = psutil.Process(os.getpid())
    if yolov5:
        sys.path.insert(0, './yolov5')
        from yolov5.models.experimental import attempt_load
        
    elif scaledyolov4:
        sys.path.insert(0, './scaledyolov4')
        from scaledyolov4.models.experimental import attempt_load
    
    elif yolov4csp:
        sys.path.insert(0, './yolov4csp')
        from yolov4csp.models.models import Darknet

    if yolov4csp:
        model = Darknet('yolov4csp/models/yolov4-csp.cfg', (640,640))
        model.load_state_dict(torch.load(weights, map_location='cpu')['model'])
    else:
        model = attempt_load(weights, map_location='cpu')
    memory_used = 'Memory usage for model load (%.3fMB)' % (process.memory_info().rss / 1024 ** 2)
    input = torch.randn(1, 3, 640, 640)
    macs, params = profile(model, inputs=(input,), custom_ops=syolov4_cust_ops)
    macs, params = clever_format([macs, params], '%.3f')
    print(macs)
    print(params)
    print(memory_used)
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='scaledyolov4')
    parser.add_argument('--weights', type=str, default='yolov5/weights/yolov5s_augmented.pt')

    opt = parser.parse_args()

    run_eval(opt.model_name, opt.weights)
