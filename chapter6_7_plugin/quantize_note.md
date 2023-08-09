### FP32
FP32 [31, 30 ~ 23, 22 ~ 0] 31位为符号位sign, 30 ~ 23 为指数位， 22~0位数值位

FP16 [15, 14~10, 9 ~ 0]
TRT FP16量化直接在导出engine时设置即可
config->setFlag(BuilderFlag::kFP16);
builder->platformHasFastFP16();
build->platformHasFastInt8();
### INT8
将浮点模型转换为基于
build->platformHasFastInt8();
基本只针对 __矩阵乘法__ 和 __卷积(也是矩阵乘)__；

FP32输入-> 量化 -> INT8矩阵乘 -> 反量化 -> FP32

INT8量化可以加速的原因：
指令加速：DP4A常规的乘加器每次完成
硬件加速：Volta架构加入了TensorCore 
#### FP16 和 INT8加速的本质：
单位时钟周期内，FP16和INT8类型的运算次数大于FP32类型的运算次数

INT8较为成熟 
神经网络具有鲁棒性，量化造成的精度损失可以看成是一种噪声
神经网络权值大部分是正态分布的，值域较小且对称

#### INT8 量化算法
1. 动态对称量化算法
实时统计数值的|max|
server
|max| 是模最大值
scale = |max| * 2 / 256
real_value = scale * quantized_value
存在位宽浪费
 

2. 动态非对称量化算法
arm
scale = |max - min| / 256
real_value = scale * (quantized_value - zero_point)
不存在位宽浪费，精度有保证  
计算较复杂，量化耗时长



3. 静态对称量化算法
TRT
推理时使用预先的缩放阈值，截断部分阈值外的数据
input的数据也具有正态分布，值域小且对称的性质
饱和机制, 使用一批calibration dataset，预先用FP32 infer统计出量化|max|

INT8 收益 = Tfloat - Tint8 - Tquant(特征输入量化运算时间) - Tdequant(输出反量化时间) 

输入较大时，可能存在负收益(量化反量化占用时间太长)

TensorRT在并行

1. 先从Sample入手, 尝试替换掉已有模型，再自定义搭建网络
2. 
3. 推荐使用FP16/INT8计算模式   FP16操作简单，精度影响小，速度提升明显，INT8可能导致精度下降
4. 使用pytorch集成TRT的部分，看看Pytorch相关量化文档
5. engine不能通用
