# from tensorrt 
import tensorrt as trt
import cuda
# import pycuda.autoinit  # 尝试官网的cuda-python
# import pycuda.driver as cuda
import time
import numpy as np
import torch

cuda.cuInit(0)

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_engine(model_file, max_ws=512*1024*1024, fp16=False):
    print("building engine")
    explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)  # 设置batch
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(explicit_batch)
    builder_config = builder.create_builder_config()
    builder_config.max_workspace_size = max_ws
    if fp16 == True:
        builder_config.set_flag(trt.BuilderFlag.FP16)                              # 设置精度

    with trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(model_file, 'rb') as model:
            parser = parser.parse(model.read())
            print("network.num_layers", network.num_layers)
            build_start_time = time.time()
            engine = builder.build_engine(network, builder_config)
            build_time_elapsed = (time.time() - build_start_time)
            TRT_LOGGER.log(TRT_LOGGER.INFO, "build engine in {:.3f} Sec".format(build_time_elapsed))
    
    print("-----engine build done!-----")
    serialized_engine = engine.serialize()  # Serialize the engine to a stream.
    with open('engine_br.trt', 'wb') as f:
        f.write(bytearray(serialized_engine))

    return engine


def main():
    start_time = time.time()
    engine = build_engine("/home/rui.bai/bairui_file/cuda_learning/cuda_class/chapter5/bert-base-uncased/bert_model_sim.onnx")
    with open("engine_br.trt", 'rb') as f:
        engine_bytes = f.read()

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_bytes)

    TRT_LOGGER.log(TRT_LOGGER.INFO, "engine load done!")

    # create execution context
    bert_context = engine.create_execution_context()

    batch_size = 1
    input_ids = np.ones((1, 30))
    token_type_ids = np.ones((1, 30))
    attention_mask = np.ones((1, 30))

    # specify buffers for outputs:
    bert_output = torch.zeros((1, 30, 30522)).cpu().detach().numpy()
    d_input_ids = cuda.cuda.cuMemAlloc(batch_size * input_ids.nbytes)
    d_token_type_ids = cuda.cuMemAlloc(batch_size * token_type_ids.nbytes)
    d_attention_mask = cuda.cuMemAlloc(batch_size * attention_mask.nbytes)

    d_output = cuda.cuMemAlloc(batch_size * bert_output.nbytes)

    # bindings array
    bindings = [int(d_input_ids), int(d_token_type_ids), int(d_attention_mask), int(d_output)]

    stream = cuda.cuStreamCreate(CU_STREAM_NON_BLOCKING)
    start = cuda.cuEventCreate()
    end = cuda.Event()
    
    # transfer input data from python buffer to device
    cuda.memcpy_htod_async(d_input_ids, input_ids, stream)
    cuda.memcpy_htod_async(d_token_type_ids, token_type_ids, stream)
    cuda.memcpy_htod_async(d_attention_mask, attention_mask, stream)

    # execute using the engine
    start.record(stream)
    bert_context.execute_async(batch_size, bindings, stream.handle, None)
    end.record(stream)
    end.synchronize()
    event_time = end.time_since(start)

    cuda.memcpy_dtoh_async(bert_output, d_output, stream)
    stream.synchronize()
    TRT_LOGGER.log(TRT_LOGGER.INFO, "memory transfer done")
    
    with open("/home/rui.bai/bairui_file/cuda_learning/cuda_class/chapter5/output_python.bin", "wb") as _f:
        _f.write(bert_output)
    TRT_LOGGER.log(TRT_LOGGER.INFO, "output write done")

    pred = torch.tensor(bert_output)
    pred_output_softmax = torch.nn.Softmax()(pred)
    print(pred_output_softmax.shape)
    _, predicted_idx = torch.max(pred_output_softmax, 1)
    print(predicted_idx)
    end_time = time.time()
    
    running_time = end_time - start_time
    print("python API program running time total: ", running_time, "seconds")
    print("python inference time by event: ", event_time, "ms")


if __name__ == "__main__":
    main()