/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
 
 #include <cuda_runtime.h>
 #include "LayerNormPlugin.h"

using namespace nvinfer1;

PluginFieldCollection LayerNormPluginCreator::fc_{};
std::vector<PluginField> LayerNormPluginCreator::attr_;

__global__ void layerNormKernel(const float *pInput, float *pOutput)
{
    const int tx = threadIdx.x, index = blockIdx.x * 256 + threadIdx.x;

    __shared__ float temp[128];

    float value0 = pInput[index];
    float value1 = pInput[index + 128];

    temp[tx] = value0 + value1;
    __syncthreads();

    // 所有256个元素求和
    for (int stride = 64; stride >= 1; stride /= 2)
    {
        if (tx < stride)
        {
            temp[tx] += temp[tx + stride];
        }
        __syncthreads();
    }
    // 256个元素求平均
    float mean = temp[0] / 256;
    __syncthreads();

    // 求出value0 和value1的方差
    temp[tx] = (value0 - mean) * (value0 - mean) + (value1 - mean) * (value1 - mean);
    __syncthreads();

    // 求出256个元素的方差之和
    for (int stride = 64; stride >= 1; stride /= 2)
    {
        if (tx < stride)
        {
            temp[tx] += temp[tx + stride];
        }
        __syncthreads();
    }
    // 求出均方差
    float var = temp[0] / 256;

    // 求出 layernormal后的每个元素   rsqrf(x) return 1 / (x)^(1/2)
    pOutput[index]       = (value0 - mean) * rsqrtf(var + 6e-6);
    pOutput[index + 128] = (value1 - mean) * rsqrtf(var + 6e-6);
}

int32_t LayerNormPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, 
                                 const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
    printf("nBlock: %d\n", nBlock);
    layerNormKernel <<<nBlock, 128, 0, stream>>>((const float *)inputs[0], (float *)outputs[0]);
    return 0;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);

