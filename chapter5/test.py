n1 = 1
n2 = 2
n3 = n1 | n2
print(n3)

import torch
a = torch.randn(3, 4)
print(a)
b = torch.max(a, 0) # 
print(b)
c = torch.max(a, 1)
print(c)