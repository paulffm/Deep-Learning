import numpy as np
import torch
import torchvision
import torch.nn as nn

torch.set_printoptions(precision=10)

np.random.seed(12345)


def initialize(input_dim, hidden1_dim, hidden2_dim, output_dim, batch_size):
    W1 = np.random.randn(hidden1_dim, input_dim) * 0.01
    b1 = np.zeros((hidden1_dim,))
    W2 = np.random.randn(hidden2_dim, hidden1_dim) * 0.01
    b2 = np.zeros((hidden2_dim,))
    W3 = np.random.randn(output_dim, hidden2_dim) * 0.01
    b3 = np.zeros((output_dim,))

    parameters = [W1, b1, W2, b2, W3, b3]
    x = np.random.rand(input_dim, batch_size)
    y = np.random.randn(output_dim, batch_size)

    return parameters, x, y

def forward(parameters, x):

    W1 = parameters[0]
    b1 = parameters[1]
    W2 = parameters[2]
    b2 = parameters[3]
    W3 = parameters[4]
    b3 = parameters[5]

    z1 = torch.matmul(W1, x) + torch.reshape(b1, (-1, 1))
    h1 = torch.sigmoid(z1)
    print('Hidden Layer 1: \n', h1)
    z2 = torch.matmul(W2, h1) + torch.reshape(b2, (-1, 1))
    h2 = torch.sigmoid(z2)
    print('Hidden Layer 2: \n', h2)
    o3 = torch.matmul(W3, h2) + torch.reshape(b3, (-1, 1))
    print('Output \n', o3)

    return h1, h2, o3


def loss(y, ypred):

    l = 0.5 * torch.sum(torch.square(ypred - y), 1)

    return torch.mean(l)

def main():
    input_dim = 3
    hidden_dim = 4
    output_dim = 2
    batch_size = 5
    parameters, X, Y = initialize(input_dim, hidden_dim, hidden_dim, output_dim, batch_size)

    # convert to tensor
    x = torch.tensor(X)
    y = torch.tensor(Y)

    # gradients can be computed
    W1 = torch.tensor(parameters[0], requires_grad=True)
    b1 = torch.tensor(parameters[1], requires_grad=True)

    W2 = torch.tensor(parameters[2], requires_grad=True)
    b2 = torch.tensor(parameters[3], requires_grad=True)

    W3 = torch.tensor(parameters[4], requires_grad=True)
    b3 = torch.tensor(parameters[5], requires_grad=True)

    tensorparam = [W1, b1, W2, b2, W3, b3]


    h1, h2, o3 = forward(tensorparam, x)

    loss_val = loss(y, o3)

    # backpropagation:
    loss_val.backward()
    print('loss val:', loss_val)
    print('dW1:', W1.grad)
    print('db1:', b1.grad)
    print('dW2:', W2.grad)
    print('db2:', b2.grad)
    print('dW3:', W3.grad)
    print('db3:', b3.grad)
    b = 3
    print(b)


if __name__ == '__main__':
    main()