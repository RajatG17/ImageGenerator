import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms

from generators.generator_cnn import Generator
from discriminators.discriminator_cnn import Discriminator
from dataloaders.loader import load_cifar100, CIFAR100Dataset

z_dim = 100

generator = Generator(z_dim, img_shape=(3, 32, 32))
discriminator = Discriminator(img_shape=(3, 32, 32))

lr = 2e-5
b1 = 0.5
b2 = 0.999

# optimizer for both generator and discriminator
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Data Loading and Preprocessing

train_images, train_labels, test_images, test_labels = load_cifar100()




