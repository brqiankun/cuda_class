1. 这是一个element-wise kernel， 即表示每个输入元素做相同的操作
2. 输入为一个tensor x和标量值(概率)p
3. 输出为一个tensor y和一个tensor mask(掩码)


curand库实现随机数