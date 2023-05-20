### chapter5
1. 下载bert模型
https://huggingface.co/bert-base-uncased/tree/main

2. 测试bert模型
test_bert_model.py

3. bert模型转成onnx
先通过pytorch接口将模型从文件读入
之后通过torch.onnx.export()导出成onnx
bertmodel2onnx.py

4. 使用onnxsim简化onnx
```
onnxsim bert-base-uncased/model.onnx bert-base-uncased/model-sim.onnx --input-shape input_ids:1,12 token_type_ids:1,12 attention_mask:1,12 --dynamic-input-shape
```


5. 调用onnx parser的 python/cpp API来进行engine构建，序列化，反序列化以及infer
- 创建一个builder; nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
- create a network definition 使用TRT_API搭建网络结构(INetworkDefinition) 
  - 可以手动使用api搭建，见第6章
  - 也可使用TRT自带的parser解析建好的ONNX模型  
    - nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    - parser->parseFromFile(modelFile.data(), static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));
- build an engine
  - create a build configuration  nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
  - 设置config
  - 根据 __network definition__ 与 __config__ 将network序列化成 __engine__
  nvinfer1::IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);
- save the engine to disk
- deserializing a plan
    ```
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), engine_file_size, nullptr);
      
    ```
- inference
  - 创建execution context   nvinfer1::IExecutionContext* context = engine->createExecutionContext();
  - 数据准备，分配输入输出内存空间
  - 将数据付给context
    ```
    context->setTensorAddress(input_ids.data(), input_ids_d);
    context->setTensorAddress(token_type_ids.data(), token_type_ids_d);
    context->setTensorAddress(attention_mask.data(), attention_mask_d);
    context->setTensorAddress(logits.data(), logits_d);
    ```
  - start inference  context->enqueueV3(0);
  
安装cuda-python后api较为复杂，先用cpp API跑通

使用下载的 _libtorch-shared-with-deps-1.11.0+cu113.zip_ 可以编译跑通
