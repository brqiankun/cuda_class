## TRT 常用链接

https://www.bilibili.com/video/BV15Y4y1W73E?spm_id_from=333.337.search-card.all.click

https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/cookbook

(TensorRT 文档)[https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html]
(TensorRT C++ API)[https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/]
(TensorRT Python API)[https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/]
(TensorRT 下载)[https://developer.nvidia.com/nvidia-tensorrt-download]
(TensorRT 版本支
持列表)[https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html]

https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html


## chapter6 TensorRT Plugin
支持TRT 不支持的算子
合并算子: 对于复杂网络: 包括中间数据的一些处理 
将访存密集型的算子进行合并
pluginType
pluginVersion
pluginName
pluginVersion

#### pluginCreater
用于序列化和反序列化

根据输入维度是否是静态的
Dynamic Shape 
Static Shape

IPluginV2IOExt/IPluginV2DynamicExt: 插件类, 用于写插件的具体实现
IPluginCreator: 插件工厂类

### static shape plugin API 
1. 构造函数: 前三种情况是一类
    1. network def   
    2. clone  
    3. creator
    4. deserialize阶段
删除默认构造函数

2. 输出相关函数
获得layer的输出个数

3. 序列化和反序列化相关函数
getPluginType()

4. 初始化 配置 销毁函数
initialize()
terminate()

判断输入输出，数据类型等是否支持
configurePlugin()
supportsFormatCombination()

5. 运行相关函数
getWorkspaceSize() eg. A x B + b = C;  A x B 的中间结果大小满足workspace的大小 
最好不要在plugin enqueue中使用cudaMalloc()申请显存
workspace可以进行显存复用, 防止显存溢出
enqueue()

IPluginCreator()  

EmbLayerNormPlugin 
Embedding + Layernorm合并

权值所占显存是不能复用的


createPlugin()  用户外部调用，对plugin构造进行封装

### dynamic shape plugin api
static implicit batch    dynamic explicit batch

getOutputDimensions() 存在差异 

PluginCreator注册
加载NvInferRuntimeCommon.h头文件时，会得到getPluginRegistry, 这个类中包含所有已经注册的IPluginCreator, 使用时通过getPluginCreator函数得到对应的IPluginCreator
1. 调用API注册  getPluginRegistry()->registerCreator()
2. REGISTER_TENSORRT_PLUGIN注册

TRT是闭源软件, API相对复杂
1. API/parser构建网络, 模型转换后误差很大
2. 增加plugin实现算子合并后, 结果对不上
3. 使用FP16/INT8优化后, 算法精度降低很多(FP16大多可以)

debug方法
1. parser转换网络, dump API接口, 检查网络结构
2. plugin的单元测试
3. 打印每一层的输出. 将可疑层的输出设置为network output; 增加debug plugin


### 算子相关
#### 线性层（Linear layer）
在深度学习中，线性层（Linear layer），也称为全连接层（Fully Connected layer），是一种常见的神经网络层类型。线性层将输入数据与权重矩阵相乘，并加上偏置项得到输出。其公式可以表示为：y=xW+b, 其中x 表示输入张量，W 表示权重矩阵，b 表示偏置项，y 表示输出张量。
线性层通常用于实现基本的降维、分类、回归等任务，例如图像分类、语言模型、机器翻译等。在深度神经网络中，多个线性层可以组合起来形成更复杂的神经网络结构。由于线性层只是对输入进行线性变换，因此它无法处理非线性问题。因此，在许多情况下，线性层需要与非线性激活函数一起使用，例如ReLU（Rectified Linear Unit）等函数，以增加模型的表达能力和拟合能力。

主要完成builder.py 使用 __TRT API__ 手动构建网络, 添加自定义算子，而不是使用 __ONNX parser__ 导入


1. Linear  add_fully_connected

2. linear_norm plugin


### TRT 使用流程
1. 导出网络定义和相关权重
2. 解析网络定义和相关权重
3. 根据硬件(显卡)算子构建出最优执行计划
4. 将执行计划序列化存储
5. 反序列化执行计划
6. 进行推理
从步骤3看出，TRT是和硬件绑定的，在部署时，硬件(显卡)和软件(驱动driver， cudatoolkit， cudnn)发生变化，则需要从3重新开始。
换句话说，不同系统环境下生成Engine 包含硬件有关优化，不能跨硬件平台使用，注意环境统一(硬件环境+ CUDA/cuDNN/TensorRT 环境)，并且不同版本TensorRT 生成的 engine不能相互兼容，同平
台同环境多次生成的engine可能不同。


推理框架的性能优化目标:
1. 尽可能把非GEMM kernel进行融合
2. GEMM kernel(Tensor Core)占比较高，例如在90%以上

使用cuda-graph来减少launch bound

优化流程:
1. run Framework -> ONNX -> TRT to get baseline
2. Profile and find perf bottleneck
3. Optimize with **ONNX-graphsurgen** and **TRT plugin**
可以使用多stream，int8量化
4. repeat step2 and step3 until satisifed
Conv + BN + Relu 融合
融合后可以减少memory bounded的算子对内存的读写次数

**nsight system** 优化模型层次的性能分析
1. 寻找CPU和GPU的空闲时间
2. 多卡，多stream，memory transfers

**nsight compute** 优化cuda kernel
SOL, Memory Chart
https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html

**NVTX** 人为插入一下标签来标注优化
TRT 自带NVTX，根据layer name匹配kernel和layer
因为TRT的优化，原始模型结构和实际执行的模型结构不相同
在build engine过程中会自动加入reformat layer，这些reformat layer在模型中并不存在，被表示为unnamed

Myelin 自动实现算子融合 

**trtexec**

### 性能优化技巧
1. batching 增大batch size，增大吞吐量
2. stream 使用多个执行stream， 避免使用default stream, 增加implicit sync
3. cuda graph, 对于多个小kernel执行，可能存在launch bounded情况，可以通过cuda graph来加速。 不支持dynamic shape，需要capture多个CUDA Graph. 设置nsys --cuda-graph-trace=node 查看kernel

#### MatMul Layer
如果使用FP16， 而且K不是8的倍数，TRT可能会引入reformat kernel来实现tensor core的使用，可以手动padding到8的倍数来规避。如果input是4维，可以通过尝试reshape转换为3维或2维。

使用Tensor Core才能充分发挥GPU的性能。计算密集型算子，MatrixMultiply, FullyConnected, Convolution, and Deconvolution等计算密集型的算子会使用Tensor Core.
Tenosr Core需要满足**data alignment**的要求。
可以使用nsys --gpu-metrics-device all 来查看Tensor Core使用情况。

#### Sparsity
结构化稀疏，减少参数，使用Sparse Tensor Core进行加速。
训练阶段，使用ASP(Automatic SParsity)修剪参数，然后微调模型来维持精度
推理阶段，是设置build_config的SPARSE_WEIGHTS flag来挑选sparse kernel

#### Optimizing Plugins


trt中自带了LayerNorm plugin，导致自定义layerNorm plugin不会被选中生效，修改自定义plugin的名字后生效


