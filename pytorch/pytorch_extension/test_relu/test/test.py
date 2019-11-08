import torch
import test_relu

a = torch.zeros([10])
b = test_relu.forward(a)
print(b)
