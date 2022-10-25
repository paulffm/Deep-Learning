import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch; torch.manual_seed(0)
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import numpy as np

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
image_size = 784 # 28*28 pixels, since we use linear layers
h_dim = 400
z_dim = 2
num_epochs = 10
batch_size = 128
learning_rate = 1e-3

# MNIST dataset
dataset = torchvision.datasets.MNIST(root='../../data',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=True)


# Show samples from MNIST
def visualize(X, name):
    """Visualize the MNIST dataset. Show 5 neighboured, randomly chosen digits."""
    N = X.shape[0]
    end = np.round(np.random.rand() * N).astype('int')
    start = end - 5
    plt.figure(figsize=(20, 4))
    for index, image in enumerate(X[start:end]):
        plt.subplot(1, 5, index + 1)
        plt.imshow(np.reshape(image, (28, 28)), cmap=plt.cm.gray)
    plt.show()


visualize(np.array(dataset.data.detach().numpy().astype('float64')), "MNIST")


# Only works if z_dim = 2
def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
    w = 28  # MNIST dim
    img = np.zeros((n * w, n * w))  # init 1 large image
    for i, y in enumerate(np.linspace(*r1, n)):  # fill in each 12*12 cells with a generated MNIST image
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = autoencoder.decoder(z)
            x_hat = x_hat.reshape((-1, 1, 28, 28))
            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
            img[(n - 1 - i) * w:(n - 1 - i + 1) * w, j * w:(j + 1) * w] = x_hat
    plt.imshow(img, extent=[*r0, *r1])

# "normal" encoder (encodes x into latent vector z)
class Encoder(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)  # encodes to hidden dim
        self.fc2 = nn.Linear(h_dim, z_dim)  # generates the latent z directly

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h)

    def forward(self, x):
        z = self.encode(x)  # generate latent z directly
        return z


# VAE encoder (encodes x into mean and variance from which we sample z)
class VAE_Encoder(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE_Encoder, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)  # encodes to hidden dim
        self.fc2 = nn.Linear(h_dim, z_dim)  # generates the mean
        self.fc3 = nn.Linear(h_dim, z_dim)  # generates the var

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)  # generate mean and var
        z = self.reparameterize(mu, log_var)  # sample latent z from gaussian
        return z, mu, log_var


class Decoder(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(Decoder, self).__init__()
        self.fc4 = nn.Linear(z_dim, h_dim)  # takes latent sample z
        self.fc5 = nn.Linear(h_dim, image_size)  # decodes to image

    def decode(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))

    def forward(self, z):
        x_reconst = self.decode(z)  # reconstruct from z
        return x_reconst


class Autoencoder(nn.Module):
    def __init__(self, variational=False, image_size=784, h_dim=400, z_dim=20):
        super(Autoencoder, self).__init__()
        self.variational = variational
        if not self.variational:
            self.encoder = Encoder(image_size, h_dim, z_dim)
        else:
            self.encoder = VAE_Encoder(image_size, h_dim, z_dim)
        self.decoder = Decoder(image_size, h_dim, z_dim)

    def forward(self, x):
        if not self.variational:
            z = self.encoder(x)
            return self.decoder(z)
        else:
            z, mu, log_var = self.encoder(x)
            return self.decoder(z), mu, log_var


# intialize an AE and a VAE
models = [Autoencoder(False, image_size, h_dim, z_dim).to(device),
          Autoencoder(True, image_size, h_dim, z_dim).to(device)]

# Start training
for model in models:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for i, (x, _) in enumerate(data_loader):  # don't use labels since we use the image as "label"
            # Forward pass
            x = x.to(device).view(-1, image_size)

            if model.variational:
                x_reconst, mu, log_var = model(x)
                # KL div for univariate guassian, ensures our latent space is continuous (inter-cluster distance)
                # # For KL divergence, see Appendix B in VAE paper or http://yunjey47.tistory.com/43
                kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                # Reconstruction loss ensures that we find clusters(gaussians) that represent the training data
                reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
                # Compute loss
                loss = reconst_loss + kl_div
            else:
                x_reconst = model(x)
                # Reconstruction loss ensures that we find clusters(of gaussians) that represent the training data
                reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
                # Compute loss
                loss = reconst_loss

            # Backprop and optimize
            optimizer.zero_grad()
            loss.backward()  # calc grads
            optimizer.step()  # update

            if (i + 1) % 10 == 0:
                if model.variational:
                    print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                          .format(epoch + 1, num_epochs, i + 1, len(data_loader), reconst_loss.item(), kl_div.item()))
                else:
                    print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}"
                          .format(epoch + 1, num_epochs, i + 1, len(data_loader), reconst_loss.item()))

# AE samples
if z_dim == 2:
    plot_reconstructed(models[0], r0=(-5, 10), r1=(-5, 10))

# VAE samples
if z_dim == 2:
    plot_reconstructed(models[1], r0=(-5, 10), r1=(-5, 10))