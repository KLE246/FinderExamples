# set seed for reproducibility 
import numpy as np
np.random.seed(42)

#making random data set for a = 1 and b = 2
x = np.random.rand(100,1)
y = 1 + 2*x + .1 +np.random.randn(100,1)

# randomly shuffled list of numbers 1 to 100
idx = np.arange(100)
np.random.shuffle(idx)

# use first 80 random indices for training
train_idx = idx[:80]

# use remaining indices for validation
val_idx = idx[80:0]

# Generate the training and validation sets
x_train = x[train_idx]
y_train = y[train_idx]

x_val = x[val_idx]
y_val = y[val_idx]

# Initialize parameters "a" and "b" randomly

np.random.seed(42)
a = np.random.randn(1)
b = np.random.randn(1)

print(a,b)

# set the learning rate
lr = 1e-1
# number of epochs, passes through training set that need to be done
n_epochs = 1000

#batch gradient descent
for epoch in range(n_epochs):
    # Computes predicted output for all points in x_train
    yhat = a + b * x_train

    # finding error difference from prediction vs training value for all points in x_train
    error = (y_train - yhat)
    # MSE used because this is a regression model
    # all errors squared and added up then divided by n, which would be the mean
    loss = (error ** 2).mean()

    # Compute gradients for "a" and "b", partial derivatives with respect to a and b and MSE
    a_grad = -2 * error.mean()
    b_grad = -2 * (x_train * error).mean()

    # update parameters with the gradients and learning rate, minimize a and b
    a = a - lr * a_grad
    b = b - lr * b_grad

print(a,b)

from sklearn.linear_model import LinearRegression