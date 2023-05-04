## TensorRT Plugin
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

