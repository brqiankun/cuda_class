# 防止训练时过拟合，随机使一些输出张量为0
# During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution. 
# Each channel will be zeroed out independently on every forward call.
import numpy as np

import torch

class Dropout():
    def __init__(self, dropout_ratio = 0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flag = True):
        self.mask = np.random.rand(*x.shape) > self.dropout_ratio   # the function call with the *-operator to unpack the arguments out of a list or tuple
        print("x.shape: {}".format(x.shape))
        print("self.mask.shape: {}".format(self.mask.shape))
        return x * self.mask
    
    def backward(self, dout):
        return dout * self.mask
    

def dropout_test_torch():
    m = torch.nn.Dropout(p=0.5)  # p (float) – probability of an element to be zeroed
    input = torch.randn(3, 3)
    output = m(input)
    print(input)
    print(output)

def my_dropout_test():
    mydropout = Dropout(dropout_ratio=0.5)
    input = torch.randn(3, 3)
    output = mydropout.forward(input)
    print(input)
    print(output)




if __name__ == "__main__":
    dropout_test_torch()
    print("--" * 20)
    my_dropout_test()