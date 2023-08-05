#include <cuda_runtime.h>
#include <curand_kernel.h>


// dev_ctx 调用GPU/CPU框架的上下文，用来给输入输出分配显存/内存
// 泛化上下文数据类型和输入输出数据类型
// seed_val 设置随机数种子
// 输出结果直接在传入的指针指向的内存中
// 公式
// r = rand()
// dropout(x) = x * (1 / (1 - p)) * mask
// 需要实现两部分， 1. 根据seed_val和p生成mask； 2. 根据公式计算输出dropout(x)
template <typename T, typename Context>
__global__ void DropoutKernel(const Context& dev_ctx, const Tensor& x, const Scalar& p, 
                              int seed_val, Tensor* y, Tensor* mask) {

}

// 实现随机数生成mask
// functor of compute mask = 0 or 1
template <typename T1, typename T2 = T1, typename OutT = T1>
struct MaskFunctor {
    const float retain_prob_;
    float factor;
    MaskFunctor(const float retain_prob) : retain_prob_(retain_prob) {
        factor = 1.0f / retain_prob;   // 1 / (1 - p)
    }
    inline void operator()(OutT* mask, const T2* rand, int num) const {
        static constexpr int kCount = uniform_distribution<T2>::kCount;
#pragma unroll
        for (int i = 0; i < kCount; i++) {
            if (rand[i] < retain_prob_) {
                mask[i] = static_cast<T1>(1);
            } else {
                mask[i] = static_cast<T1>(0);
            }
        }
    }
};

// 假设有6 x 256个元素，则分配6个线程块，每个线程块中256个线程，每个线程处理一个元素mask的生成
// 生成mask分为， 1. 使用curand生成随机数rand；2. 通过maskfunctor比较rand和概率p来决定mask是0或1
template <typename T>
struct uniform_distribution {
    inline T operator()(curandStatePhilox4_32_10_t* state) const {
        return static_cast<T>(curand_uniform(state));
    }
};

// CUDA 并行程序，计算mask
template <typename MaskType>
__global__ void DropoutMaskKernel(Tensor* mask) {
    curandStatePhilox4_32_10_t state;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float rands[1];
    MaskType dst_mask[1];
    using Rand = uniform_distribution<float>;
    using mask_functor = MaskFunctor<MaskType, float>(1.0f - dropout_prob);
    auto random_tuple = Rand()(&state);
    rands[0] = static_cast<float>((&random_tuple.x)[0]);
    // compute mask
    mask_functor(&dst_mask[0], &rands[0], 1);
    mask[i] = static_cast<MaskType>(dst_mask[i]);
}

// functor of compute dropout result, when mask = 1, res = input * mask, otherwise 0
template<typename T, typename MaskType>
struct DstFunctor {
    float factor;
    inline DstFunctor(const float retain_prob) : retain_prob_(retain_prob) {
        factor = 1.0f / retain_prob_;
    }
};


