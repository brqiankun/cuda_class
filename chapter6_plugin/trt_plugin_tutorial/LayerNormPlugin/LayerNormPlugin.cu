#include "LayerNormPlugin.h"

#define SWITCH 0

using namespace nvinfer1;

PluginFieldCollection LayerNormPluginCreator::fc_{};
std::vector<PluginField> LayerNormPluginCreator::attr_;

template <int VPT>
struct BytesToType;

template <>
struct BytesToType<2>
{
    using type = uint16_t;
};
template <>
struct BytesToType<4>
{
    using type = uint32_t;
};
template <>
struct BytesToType<8>
{
    using type = uint64_t;
};
template <>
struct BytesToType<16>
{
    using type = float4;
};

template <int Bytes>
__device__ inline void copy(const void* local, void* data)
{
    using T = typename BytesToType<Bytes>::type;

    const T* in = static_cast<const T*>(local);
    T* out = static_cast<T*>(data);
    *out = *in;
}

struct mySum
{
    __host__ __device__ __forceinline__ float2 operator()(const float2 &a, const float2 &b) const
    {
        return make_float2(a.x + b.x, a.y + b.y);
    }
};

#if SWITCH == 0
template <typename T, int TPB, int VPT>
__global__ void layerNormKernel(const T* input, const T* gamma, const T* beta, T* output)
{
    const int idx = blockIdx.x * 256 + threadIdx.x * VPT;
    T localX[VPT], localGamma[VPT], localBeta[VPT];

    copy<sizeof(T) * VPT>(&input[idx], localX);
    float2 localFloat2 = {0.f,0.f};

    const float rld = float(1)/ float(256);
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        const float tmp = rld * (float)localX[it];
        localFloat2.x += tmp;
        localFloat2.y += tmp * (float)localX[it];
    }

    copy<sizeof(T) * VPT>(&beta[threadIdx.x * VPT], localBeta);
    copy<sizeof(T) * VPT>(&gamma[threadIdx.x * VPT], localGamma);

    using BlockReduce = cub::BlockReduce<float2, TPB>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float mu;     // mean
    __shared__ float rsigma; // 1 / std.dev.

    //const float2 sumKV = BlockReduce(temp_storage).Reduce(localFloat2, cub::Sum());
    const float2 sumKV = BlockReduce(temp_storage).Reduce(localFloat2, mySum());

    if (threadIdx.x == 0)
    {
        mu = sumKV.x;
        rsigma = rsqrt(sumKV.y - mu * mu + 1e-6);
    }
    __syncthreads();
#pragma unroll
    for (int it = 0; it < VPT; it++)
    {
        localX[it] = (float)localGamma[it] * ((float)localX[it] - mu) * rsigma + (float)localBeta[it];
    }

    copy<sizeof(T) * VPT>(localX, &output[idx]);
}
#endif

#if SWITCH == 1
__global__ void layerNormKernel(const float *pInput, float *pOutput) {
    const int tx = threadIdx.x;
    const int index = blockIdx.x * 256 + threadIdx.x;

    __shared__ float temp[128];

    float value0 = pInput[index];
    float value1 = pInput[index + 128];

    temp[tx] = value0 + value1;
    __syncthreads();

    // 所有256个元素求和
    for (int stride = 64; stride >= 1; stride /= 2) {
        if (tx < stride) {
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
    for (int stride = 64; stride >= 1; stride /= 2) {
        if (tx < stride) {
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
#endif

#if SWITCH == 0
// Explicit instantiation definition 
template __global__ void layerNormKernel<float, 64, 4>(const float*, const float*, const float*, float*);
// template __global__ void layerNormKernel<half, 32, 8>(const half*, const half*, const half*, half*);
#endif

int LayerNormPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, 
                             void* const* outputs, void* workspace, cudaStream_t stream) noexcept {
#if SWITCH == 0
    const int gridSize = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
    printf("in layerNormKernel by br");
    printf("\n------------------   gridSize: %d   --------------------------\n",gridSize);
    if (inputDesc[0].type == DataType::kFLOAT) {
        constexpr int VPT = 16 / sizeof(float);
        constexpr int TPB = 256 / VPT;
        printf("\n\nVPT: %d   TPB: %d\n\n", VPT, TPB);
        //                                                64
        (layerNormKernel<float, TPB, VPT>)   <<<gridSize, TPB, 0, stream>>>  ((const float*)inputs[0], (const float*)inputs[1], (const float*)inputs[2], (float*)outputs[0]);
    }
#elif SWITCH == 1
    const int nBlock = inputDesc[0].dims.d[0] * inputDesc[0].dims.d[1];
    printf("nBlock: %d\n", nBlock);
    layerNormKernel<<<nBlock, 128, 0, stream>>>((const float *)inputs[0], (float *)outputs[0]);
#endif
    return 0;
}

REGISTER_TENSORRT_PLUGIN(LayerNormPluginCreator);

