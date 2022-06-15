from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import numpy as np
import matplotlib.pyplot as plt
import pdb
np.random.seed(12345)

def initialize(input_dim, hidden1_dim, output_dim, batch_size):
    W1 = np.random.randn(hidden1_dim, input_dim) * 0.01
    b1 = np.zeros((hidden1_dim, ))
    W2 = np.random.randn(output_dim, hidden1_dim) * 0.01
    b2 = np.zeros((output_dim, ))
    parameters = [W1, b1, W2, b2]
    return parameters

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def dsigmoid(x):
    # input has to be sigmoid
    # in backpropagation you insert hidden_i
    return x * (1 - x)

def loss(prediction, target):
    y = np.sum(0.5 * (prediction - target) ** 2)
    return (1 / target.shape[1]) * np.mean(y)

def dloss(prediction, target):
    return (1 / target.shape[1]) * (prediction - target)

# helper functions
def convert_to_1d_vector(parameters):
    W1, b1, W2, b2 = parameters
    params = np.concatenate([W1.ravel(), b1.ravel(),
                             W2.ravel(), b2.ravel()], axis=0)

    return params
def convert_to_list(params, input_dim, hidden1_dim, output_dim):
    base_idx = 0

    W1 = np.reshape(params[base_idx: base_idx + input_dim * hidden1_dim],
                    (hidden1_dim, input_dim))
    base_idx += input_dim * hidden1_dim

    b1 = params[base_idx: base_idx + hidden1_dim]
    base_idx += hidden1_dim

    W2 = np.reshape(params[base_idx: base_idx + hidden1_dim * output_dim],
                    (output_dim, hidden1_dim))
    base_idx += hidden1_dim * output_dim

    b2 = params[base_idx: base_idx + output_dim]

    parameters = [W1, b1, W2, b2]

    return parameters

def visualize(X,y,name):
    """Visualize the MNIST dataset. Show 5 neighboured, randomly chosen digits."""
    N = X.shape[0]
    end = np.round(np.random.rand() * N).astype('int')
    start = end - 5
    plt.figure(figsize=(20, 4))
    for index, (image, label) in enumerate(zip(X[start:end], y[start:end])):
        plt.subplot(1, 5, index + 1)
        plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
        plt.title(name + ': ' + str(label), fontsize = 20)
    plt.show()

def vectorized_results(y):
    """Returns a one-hot-coded vector [0,0,1,0,0,0,0,0,0] which represents a digit (here 3).
    This is used to compare to the output of the neural network."""
    v = np.zeros(10, dtype=int)
    v[y] = 1.0
    return v

def devectorize_results(y):
    """Returns the predicted digit from a one-hot-coded vector. [0,0,1,0,0,0,0,0,0] will give 3."""
    return np.argmax(y)


def update_mini_batch(net, mini_batch, learning_rate):
    """Most of the training is done here.
    Weights + biases are updated for mini-batch"""

    N = mini_batch.shape[1]
    y = mini_batch[:, N - net.output_dim:N].T
    x = mini_batch[:, 0:N - net.output_dim].T

    activations = net.forward(x)
    loss_value = loss(activations, y)
    dParameters = net.backward(targets=y)
    net.parameters = net.parameters - learning_rate * dParameters

    return net.parameters, loss_value

# Train neural network with stochastic gradient descent.
def train(net, train_data, label_data, epochs=1000, batch_size=5, learning_rate=0.01):
    N = len(train_data)
    data = np.concatenate([train_data, label_data], axis=1)

    losses = list()
    for i in range(epochs):
        # stochastic mini batch
        np.random.shuffle(data)
        # divide data set into batch_size/N parts
        mini_batches = [data[j:j+batch_size] for j in range(0, N, batch_size)]
        for mini_batch in mini_batches:
            _, loss_value = update_mini_batch(net, mini_batch, learning_rate)
            losses.append(loss_value)

        #print ('Epoch {0} complete'.format(i))

    return net.parameters, losses


class NeuralNet(object):
    def __init__(self, batch_size=5, input_dim=3, hidden_dim=4, output_dim=2):
        self.batch_size = batch_size

        # size of layers
        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim
        self.output_dim = output_dim

        self.placeholder_in = np.ones((self.input_dim, self.batch_size))
        self.placeholder_latent1 = np.ones((self.hidden_dim_1, self.batch_size))
        self.placeholder_out = np.ones((self.output_dim, self.batch_size))

        self.parameters = initialize(input_dim, hidden_dim, output_dim, batch_size)

        self.W1, self.b1, self.W2, self.b2 = self.parameters
        self.parameters = convert_to_1d_vector(self.parameters)

    def forward(self, x):

        #   Computing forward pass
        # input activations
        self.placeholder_in = np.zeros((self.input_dim, x.shape[1]))

        # hidden layer activations
        self.placeholder_latent1 = np.zeros((self.hidden_dim_1, x.shape[1]))

        # output activation
        self.placeholder_out = np.zeros((self.output_dim, x.shape[1]))

        self.placeholder_in = x
        self.placeholder_latent1 = sigmoid(np.dot(self.W1, self.placeholder_in) + self.b1[:, np.newaxis])
        self.placeholder_out = self.W2 @ self.placeholder_latent1 + self.b2[:, np.newaxis]

        return self.placeholder_out

    def backward(self, targets):
        [self.W1, self.b1, self.W2, self.b2] = convert_to_list(self.parameters, self.input_dim, self.hidden_dim_1, self.output_dim)

        dw1 = np.zeros((self.hidden_dim_1, self.input_dim))
        db1 = np.zeros((self.hidden_dim_1,))

        dw2 = np.zeros((self.output_dim, self.hidden_dim_1))
        db2 = np.zeros((self.output_dim,))

        delta = dloss(self.placeholder_out, targets)
        dw2 = delta @ self.placeholder_latent1
        db2 = np.sum(delta, axis=1)

        delta = self.W1 @ delta @ dsigmoid(self.placeholder_latent1)
        dw1 = delta @ self.placeholder_in.T
        db1 = np.sum(delta, axis=1)

        dParameters = convert_to_1d_vector([dw1, db1, dw2, db2])

        return dParameters

    def predict(self, x, parameters):
        """Predict a test data set on the trained parameters."""
        self.W1, self.b1, self.W2, self.b2 = convert_to_list(parameters, self.input_dim, self.hidden_dim_1,
                                                             self.output_dim)
        return self.forward(x)

def read_data():
    # get data set: MNIST digits
    mnist = fetch_openml('mnist_784')
    X = np.array(mnist.data.astype('float64')) #shape: 1000 examples of 28x28px images
    y = np.array(mnist.target, dtype='int')
    random_state = check_random_state(0)
    # split into training and test data set. Reduce size to 1000/300 samples instead of 60000/10000
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1000, test_size=300, random_state=random_state)
    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)
    return [X_train, X_test, y_train, y_test]

def gradient_check(parameters, gradients, X, Y, loss, forward_pass, eps=1e-7):
    W1, b1, W2, b2 = parameters
    network_structure = [X.shape[0], W1.shape[0], W2.shape[0]]

    # convert a list of parameters to a single vector
    params = convert_to_1d_vector(parameters)
    grads = gradients

    n_params = len(params)
    losses_plus = np.zeros((n_params,))
    losses_minus = np.zeros((n_params,))
    num_grads = np.zeros((n_params,))

    for i in range(n_params):
        params_eps_plus = np.copy(params)
        params_eps_plus[i] += eps

        parameters_plus = convert_to_list(params_eps_plus, *network_structure)

        activations = forward_pass(parameters_plus, X)
        losses_plus = loss(activations, Y)

        params_eps_minus = np.copy(params)
        params_eps_minus[i] -= eps

        parameters_minus = convert_to_list(params_eps_minus, *network_structure)

        activations = forward_pass(parameters_minus, X)
        losses_minus = loss(activations, Y)

        num_grads[i] = (losses_plus - losses_minus) / (2*eps)

    diff = np.linalg.norm(grads - num_grads) / (np.linalg.norm(grads) + np.linalg.norm(num_grads))

    return diff



def main():
    batch_size = 10
    epochs = 10
    learning_rate = 0.01
    X_train, X_test, y_train, y_test = read_data()
    print(X_train.shape)
    # if batch size = all samples
    # batch_size = X_train.shape[0]

    print('That is how the data set looks like:')
    visualize(X_train, y_train, 'Training')

    # bring labels into one-hot-coded vector form
    expected = np.array([vectorized_results(y) for y in y_train])

    hidden_dim = 15
    output_dim = 10

    net = NeuralNet(batch_size=batch_size, input_dim=784, hidden_dim=hidden_dim, output_dim=output_dim)

    # Training
    parameters, losses = train(net, X_train, expected, epochs, batch_size, learning_rate=learning_rate)

    print('The loss decreasing during training:')
    plt.figure(figsize=(12, 6))
    plt.plot(losses)
    plt.show()

    # Prediction
    predicted = net.predict(X_test.T, parameters).T

    expected = y_test
    predicted = [devectorize_results(y) for y in predicted]

    print('Here are some predicted examples:')
    visualize(X_test, predicted, 'Test')

    # Results
    print('Classification report :')
    print(classification_report(expected, predicted))
    print('Confusion matrix:')
    print(confusion_matrix(expected, predicted))


    '''input_dim = 3
    hidden_dim = 4
    output_dim = 2
    batch_size = 5

    net = NeuralNet(batch_size=batch_size, input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    X = np.random.rand(input_dim, batch_size)
    Y = np.random.randn(output_dim, batch_size)
    parameters = [net.W1, net.b1, net.W2, net.b2]

    activations = net.forward(X)
    loss_value = loss(activations, Y)
    print('Loss: {}'.format(loss_value))

    grads = net.backward(targets=Y)


def forward_pass(params, x):
    net.W1, net.b1, net.W2, net.b2 = params
    return net.forward(x)


diff = gradient_check(parameters, grads, X, Y, loss, forward_pass=forward_pass)

print('Gradient checking: ')
if diff < 1e-7:
    print('\tPassed')
else:
    print('\tFailed')'''

if __name__ == '__main__':
    main()