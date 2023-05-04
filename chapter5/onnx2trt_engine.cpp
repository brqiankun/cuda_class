#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <cuda_runtime.h>
#include <torch/torch.h>

#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <algorithm>
#include <vector>
#include <chrono>

#include "dbg.h"

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;   
    }
};

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();
    Logger logger;
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);

    uint32_t flag = 1U << static_cast<uint32_t>
                       (nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    // create a network definition
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flag);
    std::cout << "hello" << std::endl;
    // create an onnx parser to populate the network
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    
    // read the model file and process any errors
    std::string modelFile {"/home/rui.bai/bairui_file/cuda_learning/cuda_class/chapter5/bert-base-uncased/bert_model_sim.onnx"};
    std::cout << modelFile << std::endl;
    parser->parseFromFile(modelFile.data(), static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));
    for (int32_t i = 0; i < parser->getNbErrors(); i++) {
        std::cout << parser->getError(i)->desc() << std::endl;
    }
    dbg("onnxfile parser succeed");

    // build an engine, create a build configuration first
    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    // some properties to control how tensorRT optimizes the network
    // maximum workspace size 当多个engine在一个device上生成时需要进行限制
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 22);
    config->setMaxWorkspaceSize(1U << 29);   // 设置最大工作空间为512MB
    // build the angine
    nvinfer1::IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);
    dbg("serialize model done");
    // serialized engine contains the necessary copies of the weights, the parser, network definition, builder configuration and builder are no longer necessary
    delete parser;
    delete config;
    delete network;
    delete builder;

    // // save the engine to disk
    std::string trt_engine_path {"/home/rui.bai/bairui_file/cuda_learning/cuda_class/chapter5/engine_cpp_br.trt"};
    std::fstream serializedModel_file;
    serializedModel_file.open(trt_engine_path, std::ios::out | std::ios::binary);
    assert(serializedModel_file.is_open() == true);
    serializedModel_file.write((char*)serializedModel->data(), serializedModel->size());
    serializedModel_file.close();
    dbg("file write done");
    delete serializedModel;

    // deserializing a plan
    serializedModel_file.open(trt_engine_path, std::ios::in | std::ios::binary);
    assert(serializedModel_file.is_open() == true);
    serializedModel_file.seekg(0, std::ios::end);
    size_t engine_file_size = serializedModel_file.tellg();
    std::cout << "file size: " << engine_file_size << std::endl;
    // char* engine_data = new char[engine_file_size];
    std::vector<char> engine_data(engine_file_size);
    assert(engine_data.size() > 0);
    serializedModel_file.seekg(0, std::ios::beg);  // 在seekg到文件结尾后要将position indicator放置到文件开头
    serializedModel_file.read(engine_data.data(), engine_file_size);
    // Runtime interface
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), engine_file_size, nullptr);
    dbg("deserializeCudaEngine done");

    // performing inference
    // ExecutionContext
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    // an engine can have multiple execution contexts
    // must pass TensorRT buffers for input and output
    // setTensorAddress()
    std::string input_ids {"input_ids"};
    std::string token_type_ids {"token_type_ids"};
    std::string attention_mask {"attention_mask"};
    std::string logits {"logits"};
    // input_ids = np.random.randn(1, 30)
    // token_type_ids = np.ones((1, 30))
    // attention_mask = np.ones((1, 30))

    const int cnt = 30;
    const int data_size = cnt * sizeof(float);
    torch::Tensor input_ids_h = torch::ones(30, torch::dtype(torch::kFloat32).device(torch::kCPU));
    torch::Tensor token_type_ids_h = torch::ones(30, torch::dtype(torch::kFloat32).device(torch::kCPU));
    torch::Tensor attention_mask_h = torch::ones(30, torch::dtype(torch::kFloat32).device(torch::kCPU));
    torch::Tensor logits_h = torch::zeros(30 * 30522, torch::dtype(torch::kFloat32).device(torch::kCPU));
    // float* token_type_ids_h = new float[cnt];
    // float* attention_mask_h = new float[cnt];
    // float* logits_h = new float[30 * 30522];
    float *input_ids_d, *token_type_ids_d, *attention_mask_d, *logits_d;
    
    cudaMalloc(&input_ids_d, data_size);
    cudaMalloc(&token_type_ids_d, data_size);
    cudaMalloc(&attention_mask_d, data_size);
    cudaMalloc(&logits_d, 30 * 30522 * sizeof(float));
    // cudaStream_t stream;
    // cudaStreamCreate(&stream);
    cudaMemcpy(input_ids_d, input_ids_h.data_ptr(), data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(token_type_ids_d, token_type_ids_h.data_ptr(), data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(attention_mask_d, attention_mask_h.data_ptr(), data_size, cudaMemcpyHostToDevice);


    context->setTensorAddress(input_ids.data(), input_ids_d);
    context->setTensorAddress(token_type_ids.data(), token_type_ids_d);
    context->setTensorAddress(attention_mask.data(), attention_mask_d);
    context->setTensorAddress(logits.data(), logits_d);

    //start inference using a cuda stream
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    context->enqueueV3(0);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(logits_h.contiguous().data_ptr(), logits_d, 30 * 30522 * sizeof(float), cudaMemcpyDeviceToHost);
    std::fstream output_file;
    output_file.open("/home/rui.bai/bairui_file/cuda_learning/cuda_class/chapter5/output_cpp.bin", std::ios::out | std::ios::binary);
    output_file.write((char*)logits_h.data_ptr(), 30 * 30522 * sizeof(float));
    // delete input_ids_h;
    // delete token_type_ids_h;
    // delete attention_mask_h;
    // delete logits_h;
    cudaFree(input_ids_d);
    cudaFree(token_type_ids_d);
    cudaFree(attention_mask_d);
    cudaFree(logits_d);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto running_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "cpp API program running time: " << running_time.count() << " milliseconds" << std::endl;
    std::cout << "cuda Event elapsedTime inference:" << elapsedTime << " ms" << std::endl;
    

    return 0;
}