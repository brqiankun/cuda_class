import sys
import onnx
filename = "/home/rui.bai/bairui_file/cuda_learning/cuda_class/chapter5/bert-base-uncased/bert_model_sim.onnx"
model = onnx.load(filename)
try:
    onnx.checker.check_model(model)
except Exception:
    print("model incorrect")
else:
    print("model correct")