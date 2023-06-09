## 转换工具
不支持的结构需要使用ONNX-graphsurgen 修改


1. Torch-TensorRT
不支持的子图使用libtorch实现

2. ONNX-Parser

3. TVM(Tensor Virtual Machine)

## API
demo/BERT
开发成本太高，前期手动搭建网络，写plugin成本较大, 还需要编写测试代码
但是可以对算子进行迁移。


## Plugin 手写CUDA代码

__API+plugin构建的方式比较适合工业界__

