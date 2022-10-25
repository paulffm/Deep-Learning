from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from scipy import signal as sg
import numpy as np
import matplotlib.pyplot as plt
import pdb
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms

class ConvNN():
    def __init__(self, n_channels, out_channels, kernel_size, dilation=1, padding=0, stride=1):
        super(ConvNN, self).__init__()

        self.kernel_size = (kernel_size, kernel_size)
        self.kernel_size_number = kernel_size * kernel_size
        self.out_channels = out_channels
        self.dilation = (dilation, dilation)
        self.padding = (padding, padding)
        self.stride = (stride, stride)
        self.n_channels = n_channels

        # parameters of conv layer
        self.weights = nn.Parameter(torch.ones(self.out_channels, self.n_channels, self.kernal_size_number))
        self.bias = nn.Parameter(torch.zeros((self.out_channels)))

        print("weights", self.weights.requires_grad)
        print("bias", self.bias.requires_grad)

        # init parameters
        self.reset_parameters()


def main():



if __name__ == '__main__':
        main()


