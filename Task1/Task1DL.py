import numpy as np
from matplotlib import pyplot as plt

# Neural Network with sigmoid as activation function and squared loss



def initialize(input_dim, hidden1_dim, hidden2_dim, output_dim, batch_size):

    # rows = neurons, columns = inputs
    # W1 connecting input units and units for the first hidden layer
    # W1 (1,1) = weight for input x1 for neuron h_1(1)
    # W1 (1,2) = weight for input x2 for neuron h_1(1)
    # W1 (4,3) = weight for input x3 for neuron h_4(1)

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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):

    return np.exp(-x) / ((1+np.exp(-x))**2)

def forward(parameters, X):

    W1, b1, W2, b2, W3, b3 = parameters
    batch_size = X.shape[1]
    hidden1_dim = W1.shape[0]
    hidden2_dim = W2.shape[0]
    output_dim = W3.shape[0]

    hid_1 = np.zeros((hidden1_dim, batch_size))
    hid_2 = np.zeros((hidden2_dim, batch_size))
    outputs = np.zeros((output_dim, batch_size))

    affn_1 = np.dot(W1, X)+b1[:, np.newaxis]
    hid_1 = sigmoid(affn_1)

    affn_2 = np.dot(W2, hid_1) + b2[:, np.newaxis]
    hid_2 = sigmoid(affn_2)

    #no activation function
    outputs = np.dot(W3, hid_2) + b3[:, np.newaxis]

    activations = [X, hid_1, affn_1, hid_2, affn_2, outputs]

    return activations

def squared_loss(predictions, targets):
    """ Computes mean squared error

    predictions: (output_dim, batch_size)
    targets: (output_dim, batch_size)

    """

    loss = np.zeros(targets.shape[1])
    # sum over columns I -> loss size = 1x5, then average over these 5 batches
    loss = np.sum(0.5 * (predictions - targets)**2, axis=0)
    return np.mean(loss)

def deriv_squared_loss(predictions, targets):

    batch_size = targets.shape[1]
    dloss = np.zeros(targets.shape)
    dloss = (predictions - targets) / batch_size

    return dloss

def backward(activations, targets, parameters):

    X, hid_1, z1, hid_2, z2, predictions = activations

    input_dim = X.shape[0]
    hidden1_dim = hid_1.shape[0]
    hidden2_dim = hid_2.shape[0]
    output_dim = predictions.shape[0]

    W1, b1, W2, b2, W3, b3 = parameters


    dW1 = np.zeros((hidden1_dim, input_dim))
    db1 = np.zeros((hidden1_dim,))
    dW2 = np.zeros((hidden2_dim, hidden1_dim))
    db2 = np.zeros((hidden2_dim,))
    dW3 = np.zeros((output_dim, hidden2_dim))
    db3 = np.zeros((output_dim,))

    delta = deriv_squared_loss(predictions, targets)

    dW3 = np.dot(delta, hid_2.T)
    #sum over batches
    db3 = np.sum(delta, axis=1)
    delta = np.dot(W3.T, delta) * deriv_sigmoid(z2)

    dW2 = np.dot(delta, hid_1.T)
    db2 = np.sum(delta, axis=1)
    delta = np.dot(W2.T, delta) * deriv_sigmoid(z1)
    dW1 = np.dot(delta, X.T)
    db1 = np.sum(delta, axis=1)

    grads = [dW1, db1, dW2, db2, dW3, db3]
    return grads

def convert_to_1d_vector(parameters):
    W1, b1, W2, b2, W3, b3 = parameters
    params = np.concatenate([W1.ravel(), b1.ravel(),
                             W2.ravel(), b2.ravel(),
                             W3.ravel(), b3.ravel()], axis=0)

    return params

def convert_to_list(params, input_dim, hidden1_dim, hidden2_dim, output_dim):
    base_idx = 0

    W1 = np.reshape(params[base_idx: base_idx + input_dim * hidden1_dim],
                    (hidden1_dim, input_dim))
    base_idx += input_dim * hidden1_dim

    b1 = params[base_idx: base_idx + hidden1_dim]
    base_idx += hidden1_dim

    W2 = np.reshape(params[base_idx: base_idx + hidden1_dim * hidden2_dim],
                    (hidden2_dim, hidden1_dim))
    base_idx += hidden1_dim * hidden2_dim

    b2 = params[base_idx: base_idx + hidden2_dim]
    base_idx += hidden2_dim

    W3 = np.reshape(params[base_idx: base_idx + hidden2_dim * output_dim],
                    (output_dim, hidden2_dim))
    base_idx += hidden2_dim * output_dim

    b3 = params[base_idx: base_idx + output_dim]

    parameters = [W1, b1, W2, b2, W3, b3]

    return parameters


def eval_numerical_gradient(parameters, gradients, X, Y, loss, eps=1e-7):
    W1, b1, W2, b2, W3, b3 = parameters
    network_structure = [X.shape[0], W1.shape[0], W2.shape[0], W3.shape[0]]

    # convert a list of parameters to a single vector
    params = convert_to_1d_vector(parameters)
    print(params.shape)
    # to compare
    grads = convert_to_1d_vector(gradients)


    n_params = len(params)
    num_grads = np.zeros((n_params,))
    diff = 0.

    ##############################
    #   Computing the numerical gradients
    #
    for i in range(n_params):
        pass
        # Compute cost of theta + epsilon
        params_eps_plus = np.copy(params)
        params_eps_plus[i] += eps
        parameters_plus = convert_to_list(params_eps_plus, *network_structure)

        activations = forward(parameters_plus, X)
        P = activations[-1]
        losses_plus = loss(P, Y)

        # Compute cost of theta - epsilon
        params_eps_minus = np.copy(params)
        params_eps_minus[i] -= eps
        parameters_minus = convert_to_list(params_eps_minus, *network_structure)

        activations = forward(parameters_minus, X)
        P = activations[-1]
        losses_minus = loss(P, Y)

        # Compute numerical gradients
        num_grads[i] = (losses_plus - losses_minus) / (2 * eps)

    #normalize the grad
    diff = np.linalg.norm(grads - num_grads) / (np.linalg.norm(grads) + np.linalg.norm(num_grads))
    return diff, num_grads

def main():
    input_dim = 3
    hidden_dim = 4
    output_dim = 2
    batch_size = 5

    # for _ in range(10):
    parameters, X, Y = initialize(input_dim, hidden_dim, hidden_dim, output_dim, batch_size)

    activations = forward(parameters, X)

    # outputs
    P = activations[-1]

    loss = squared_loss(P, Y)
    print('Loss: {}'.format(loss))

    grads = backward(activations, Y, parameters)

    eps = 1e-5
    diff, num_grads = eval_numerical_gradient(parameters, grads, X, Y, squared_loss, eps=eps)

    print('\nGradient checking: ')
    if diff < eps:
        print('\tPassed:', diff)
    else:
        print('\tFailed:', diff)



if __name__ == '__main__':
    main()