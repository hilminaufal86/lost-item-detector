# from onnx_opcounter import calculate_params, calculate_macs
# import onnx

# model = onnx.load_model('yolov5/weights/yolov5s_augmented.onnx')
# params = calculate_params(model)
# macs = calculate_macs(model)

# print('Number of params:', params)
# print('MACs:', macs)

import onnxruntime
# import sys
# sys.path.insert(0, './yolov5')
# from yolov5.models.experimental import attempt_load
import psutil
import os

process = psutil.Process(os.getpid())
# before = process.memory_info().rss / 1024 ** 2
# print('Memory usage before (%.3fMB)' % (before))
session = onnxruntime.InferenceSession('yolov5/weights/yolov5s_augmented.onnx', None)
# model = attempt_load('yolov5/weights/yolov5l_augmented.pt', map_location='cpu')
after = process.memory_info().rss / 1024 ** 2
print('Memory usage after (%.3fMB)' % (after))
# print('Memory usage to load model (%.3fMB)' % (after - before))