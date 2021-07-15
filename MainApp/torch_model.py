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

lr = 1e-1
n_epochs = 1000

torch.manual_seed(42)
a = torch.randn(1, requires_grad = True, dtype = torch.float, device=device)
b = torch.randn(1, requires_grad = True, dtype = torch.float, device=device)

# y would be the target values
# the model is made to make a regression line that best fits the model
# x values and corresponding y values are used for training
# the difference across the list of training values gives a loss value (MSE)
# a and b are continually updated to minimize this loss value, giving a complete final regression line that is best fit

for epoch in range(n_epochs):
    yhat = a + b * x_train_tensor #use train to put x values into function for yhat
    error = y_train_tensor - yhat # y - yhat for the difference between training and actual (try to understand what is happening here)
    loss = (error ** 2).mean() #MSE value