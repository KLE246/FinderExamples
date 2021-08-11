import torch
x = torch.arange(12)
x.shape
x.numel()
X = x.reshape(3,4)

torch.zeros((2,3,4))
torch.ones((2,3,4))
torch.randn(3, 4)
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

x = torch.tensor([1.0,2,4,8])
y = torch.tensor([2,2,2,2])
x + y, x - y, x * y, x / y, x**y

torch.exp(x)
X = torch.arange(12, dtype=torch.float32). reshape((3,4))
Y = torch.tensor([[2.0,1,4,3],[1,2,3,4], [4,3,2,1]])
torch.cat((X, Y), dim=0)
torch.cat((X, Y), dim=1)


X == Y

X.sum()
# reshape((z-dimension, rows, columns)
a = torch.arange(3).reshape((3,1))
b = torch.arange(0, 27).reshape((3, 3 ,3))
b = torch.arange(0, 9).reshape(3,3)
a
b
a + b

X[-1]
X[1:3]

X[1,2] = 9
X

X[0:2, :] = 12

Z = torch.zeros_like(Y)
Z
print('id(Z):', id(Z))
# saving in space to Z again, no new memory allocated
Z[:] = X+Z
Z
print('id(Z):', id(Z))

A = X.numpy()
A
B = torch.tensor(A)
B
type(A), type(B)

a = torch.tensor([3.5])
a, a.item(), float(a), int(a)

import os
os.makedirs(os.path.join('..', 'data'), exist_ok = True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA, Pave, 127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
    f.write('NA,Pave,NA\n')
    f.write('1,Pave,130000\n')
    f.write('2,NA,57000\n')



import pandas as pd

data = pd.read_csv(data_file)
#missing = max(data.isna().sum()) -1
#missing
#data = data.dropna(axis = 1, thresh=len(data)- missing)
print(data)
 # convert all values to numerical to be able to convert to tensor format
inputs, outputs = data.iloc[:, 0:2], data.iloc[:,2]
inputs = inputs.fillna(inputs.mean())

inputs = pd.get_dummies(inputs, dummy_na = True) # dummy_na adds extra na column

X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
X, y

A = torch.arange(20).reshape(5, 4)
len(A)
A.T

#symmetric matrix
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
B.T == B

X = torch.arange(24).reshape(2,3,4)
X

A = torch.arange(20, dtype = torch.float32).reshape(5,4)
B = A.clone()
A
A + B
A * B

a = 2
X = torch.arange(24).reshape(2,3,4)
len(X)
X.sum(axis = 2)
a +X
(a*X).shape

# x is one column, 4 rows
x = torch.arange(4, dtype = torch.float32)
x
x.sum()

A.shape
A.sum()
## *labes below only applies for 2 dimensions, priority goes ... depth, rows, columns
# last (highest axis number) is the last dimension, always columns
# axis = 0 collapses rows into one summary row
A_sum_axis0 = A.sum(axis = 0)
A_sum_axis0.shape
# axis = 1 collapses columns into one column, but shape of tensor still looks like a row
A_sum_axis1 = A.sum(axis = 1)
A_sum_axis1.shape

#mean
A.mean()
A.sum()/A.numel()

A.mean(axis = 0)
A.sum(axis = 0)/A.shape[0]

A.sum(axis = 1)/A.shape[1]

sum_A = A.sum(axis = 1, keepdims = True)
sum_A
sum_A.shape

A / sum_A

A/torch.arange(1, 5)
A.cumsum(axis= 0)
A.sum(axis = 0)

y = torch.ones(4, dtype=torch.float32)
x
y
torch.dot(x,y)

A
x
torch.mv(A, x)

A * x
torch.sum(A * x, axis = 1)


# (rows, columns)
B = torch.arange(12.).reshape(4,3)
A
B
A.shape[1] == B.shape[0]
torch.mm(A,B)

u = torch.tensor([3.0, -4.0])
torch.norm(u)

torch.abs(u).sum()

torch.norm(torch.ones((4,9)))

x = torch.arange(4.0)
x.requires_grad_(True)
x.grad

y = 2 * torch.dot(x,x)
y
y.backward()
x.grad
x.grad == 4*x

x.grad.zero_()
x
y = x.sum()
y.backward() # dy/dx all elements
x.grad

x.grad.zero_()
y = x*x
#y.backward(torch.ones(len(x)))
#test add to repo
y.sum().backward()
x.grad

#Change


