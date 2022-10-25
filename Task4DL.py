import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

'https://pytorch.org/docs/stable/generated/torch.nn.Module.html'


# define NN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # number of hidden nodes in each layer
        hidden_1 = 512
        hidden_2 = 512

        # first fully connected layer
        self.fc1 = nn.Linear(28 * 28, hidden_1)
        # activation function
        self.relu = nn.ReLU()
        # second fully connected layer
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        # third fully connected layer
        self.fc3 = nn.Linear(hidden_2, 10)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)

    #####################

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        # fc1(x) = W1 x + b (docu: x A^T + b)
        # fc1 = nn.Linear(20, 30)
        # x.shape >> (128, 20)
        # fc1(x).shape >> (128, 30) --> x W + b

        x = self.fc1(x)
        # if defined: self.relu, ass nn.ReLU:
        x = self.relu(x)
        # x = F.ReLU(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x



def main():


    # number of subprocesses to use for data loading
    num_workers = 0
    # how many samples per batch to load
    batch_size = 64
    # percentage of training set to use as validation
    valid_size = 0.2

    # convert data to torch.FloatTensor
    # normalization just on data
    transform = transforms.ToTensor()
    print(transform)

    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_index, valid_index = indices[split:], indices[:split]


    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(valid_index)

    # prepare data loaders
    # pulls instances of data from the Dataset
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              num_workers=num_workers)

    # matplotlib inline
    # obtain one batch of training images
    # Extract a batch of batch size images: 64
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    images = images.numpy()
    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, int(20 / 2), idx + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(images[idx]), cmap='gray')
        # print out the correct label for each image
        # .item() gets the value contained in a Tensor
        ax.set_title(str(labels[idx].item()))
    plt.show()

    model = Net()
    print(model)

    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    # Generally: measures performance of a classification model
    # - sum over c=1:M y_o, c log(p_o, c)
    # M=number of classes, y=binary indicator(0, 1) if class label c is correct for observation o
    # p=predicted prob. observation o is of class c

    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # for GPU usage, cuda sends model to current device
    model.to('cpu')  # cpu instead of cuda if no gpu is available

    # number of epochs to train the model
    n_epochs = 50
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf  # set initial "min" to infinity
    for epoch in range(n_epochs):
        # monitor losses
        train_loss = 0
        valid_loss = 0

        # sets model in training mode
        '''effectively layers like dropout, batchnorm etc. which behave different on the train and test procedures 
        know what is going on and hence can behave accordingly.'''
        model.train()
        # also possible: for index, data in enumerate(train_loader) -> every instance is data+label
        for data, label in train_loader:
            data = data.to('cpu')  # cpu instead of cuda if no gpu is available
            label = label.to('cpu')  # cpu instead of cuda if no gpu is available
            # clear the gradients of all optimized variables
            # otherwise we keep computed gradients and adding it up
            optimizer.zero_grad()
            # forward pass: make prediction
            outputs = model(data)
            # calculate loss
            loss = criterion(outputs, label)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            # gradients from back kinda stored in the parameters and optimizer optimize the parameters with that?
            optimizer.step()
            # update running training loss: item() extracts loss value as python float
            train_loss += loss.item() * data.size(0)
            # print(data.size(0)) -> batch size

        # Sets the module in evaluation mode.
        model.eval()  # prep model for evaluation
        for data, label in valid_loader:
            data = data.to('cpu')  # cpu instead of cuda if no gpu is available
            label = label.to('cpu')  # cpu instead of cuda if no gpu is available
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, label)
            # update running validation loss
            valid_loss = loss.item() * data.size(0)

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch + 1,
            train_loss,
            valid_loss
        ))

        #####################
        #
        #   Your code here
        #   TODO: decide if you want to apply early stopping

        # save model if validation loss has decreased (early stopping)
        '''if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), 'model.pt')
            valid_loss_min = valid_loss'''

    # initialize lists to monitor test loss and accuracy
    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    model.eval()  # prep model for evaluation
    for data, target in test_loader:
        data = data.to('cpu')
        target = target.to('cpu')
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = criterion(output, target)
        # update test loss
        test_loss += loss.item() * data.size(0)
        # convert output probabilities to predicted class
        # argmax? -> class of highest prob
        _, pred = torch.max(output, 1)
        # compare predictions to true label
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        # calculate test accuracy for each object class
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    # calculate and print avg test loss
    test_loss = test_loss / len(test_loader.sampler)
    print('Test Loss: {:.6f}\n'.format(test_loss))
    for i in range(10):
        if class_total[i] > 0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))
    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))


if __name__ == '__main__':
    main()

