import os
import onnx
import onnxruntime as ort
import numpy as np

onnx_model_path = os.path.realpath("/home/br/program/bert_origin/bert_model_sim.onnx")
onnx_model = onnx.load("/home/br/program/bert_origin/bert_model_sim.onnx")
try:
    onnx.checker.check_model(onnx_model)
except Exception:
    print("------model incorrect------")
else:
    print("------model correct--------")

# input data
batch_size = 1
input_ids = np.ones((batch_size, 30))
token_type_ids = np.ones((batch_size, 30))
attention_mask = np.ones((batch_size, 30))

ort_sess = ort.InferenceSession("")