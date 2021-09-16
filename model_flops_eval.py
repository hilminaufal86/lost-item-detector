import sys
import argparse
import torch
import torch.nn as nn
from thop import profile, clever_format

def count_hardswish(m, x, y):
    x = x[0]
    nelements = x.numel()
    m.total_ops += torch.DoubleTensor([int(nelements)])

yolov5_cust_ops = {
    nn.Hardswish: count_hardswish
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
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:
        model = attempt_load(weights, map_location='cpu')
    
    input = torch.randn(1, 3, 640, 640)
    macs, params = profile(model, inputs=(input,), custom_ops=yolov5_cust_ops)
    macs, params = clever_format([macs, params], '%.3f')
    print(macs)
    print(params)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='yolov5')
    parser.add_argument('--weights', type=str, default='yolov5/weights/yolov5s_augmented.pt')

    opt = parser.parse_args()

    run_eval(opt.model_name, opt.weights)
