import tensorrt as trt
import onnx 
import numpy as np

model_path = ""

# default logger
logger = trt.Logger(trt.Logger.WARNING)

# user defined logger
class MyLogger(trt.ILogger):
    def __init__(self):
        trt.ILogger.__init__(self)
    
    def log(self, severity, msg):
        print("hello")

builder = trt.Builder(logger)

# create a network definition in python
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

# import a model using the onnx parser
parser = trt.OnnxParser(network, logger)

# read the model file
success = parser.parse_from_file(model_path)
for idx in range(parser.num_errors):
    print(parser.get_error(idx))

if not success:
    pass

# build an engine 
# build configuration
config = builder.create_builder_config()

config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1<<20)

serialized_engine = builder.build_serialized_network(network, config)

with open("sample.engine", "wb") as f:
    f.write(serialized_engine)


# deserializing a plan
runtime = trt.Runtime(logger)

engine = runtime.deserialize_cuda_engine(serialized_engine)

with open("sample.engine", "rb") as f:
    serialized_engine = f.read()


# performing inference
context = engine.create_execution_context()




