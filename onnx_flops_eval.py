from onnx_opcounter import calculate_params, calculate_macs
import onnx

model = onnx.load_model('yolov5/weights/yolov5s_augmented.onnx')
params = calculate_params(model)
macs = calculate_macs(model)

print('Number of params:', params)
print('MACs:', macs)