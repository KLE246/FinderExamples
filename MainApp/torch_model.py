import torch
from dl_model import x_train, y_train
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot

# if there is available GPU then use that for computation, otherwise use cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# currently using cpu, Intel graphics card does not support CUDA

# transforming Numpy arrays into PyTorch Tensors then send to device for storage
# in shape of 1 column
x_train_tensor = torch.from_numpy(x_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)

print(type(x_train), type(x_train_tensor), x_train_tensor.type())

#FIRST
# Initialize a and b randomly
# to apply gradient descent on parameters, use requires_grad = True
a = torch.randn(1, requires_grad = True, dtype = torch.float)
b = torch.randn(1, requires_grad = True, dtype = torch.float)
print(a, b)