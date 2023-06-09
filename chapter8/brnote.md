### ONNX

ONNX定义了一组与平台环境无关的算子集合是一个。

pth --> onnx --> TRT

####ONNXRuntime
直接使用github 上编译好的onnxruntime库，针对不同平台，arm/x86 CPU推理

GPU上线: TRT
ARM上线
CPU上线: ONNXRuntime

ONNX
1. Node op中权值保存在输入tensor中
2. tensor


#### onnx-graphsurgeon NV推出的ONNX模型的编辑器
python API
1. 修改计算图
2. 修改子图
3. 优化计算图

算子Lower:用一个或多个常规算子来模拟模型里不支持的复杂的算子 <br>
算子Upper: 多个算子进行融合，提高训练推理速度; plugin算子融合，提高推理速度 <br>

Myelin 深度学习算子的CUDA代码生成

自动合并生成的算子可能会加重fp16/int8模式的精度损失

myelin自动生成的算子可能会导致nan值的出现(数值溢出)

#### polygraphy 调试工具
包括python API 和命令行工具
使用多种后端运行推理计算，ONNXRuntime， TRT
比较不同的后端的逐层计算结果(有些层不支持逐层对比， myelin层融合)
查看模型网络的逐层信息
分析权值，避免量化分布的问题

小模型，模型详细信息，分析权值 使用polygrapth

#### trtexec
 1. 转换ONNX模型
 2. 查看逐层信息
 3. 模型性能测试 端到端的性能测试结果





