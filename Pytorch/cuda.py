import torch

x=torch.rand(5,3)

print(torch.cuda.is_available())
print(x)