import torch

N, C, H, W = 2, 3, 5, 5
input = torch.randn(N, C, H, W)
layer_norm = torch.nn.LayerNorm([C, H, W])
output = layer_norm(input)
print("--" * 20)
print(input)
print("--" * 20)
print(output)
