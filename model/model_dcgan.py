import torch
import os
import matplotlib.pyplot as plt
import PIL.Image
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from generators.generator_dcgan import Generator
from discriminators.discriminator_dcgan import Discriminator
from dataloaders.loader import load_cifar100, CIFAR100Dataset

# visualize data
def show_training_images(dataloader, num_images=16):
    """Display a few images from the training dataset."""
    images, labels = next(iter(dataloader))  # Get a batch of images
    images = images[:num_images]  # Take only the first `num_images`

    # Normalize back to [0,1] range if necessary
    images = (images * 0.5) + 0.5  # Assuming images are normalized to [-1,1]

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        img = images[i].cpu().numpy()
        ax.imshow(img)
        ax.axis("off")

    plt.show()

# Training

def train_dcgan(generator, discriminator, train_loader, num_epochs, z_dim, device):
    
    for epoch in range(num_epochs):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0

        for batch_idx, (real_images, _) in enumerate(train_loader):
            real_images = real_images.permute(0, 3, 1, 2).to(device)
            # print("Real Image shape", real_images.shape)
            batch_size = real_images.size(0)

            # train discriminator
            noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
            # print("Noise shape", noise.shape)
            fake_images = generator(noise)
            # print("Fake Image shape", fake_images.shape)    
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_images), real_labels)
            fake_loss = adversarial_loss(discriminator(fake_images.detach()), fake_labels)
            d_loss = (real_loss + fake_loss)
            epoch_d_loss += d_loss.item()
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
            fake_images = generator(noise)
            fake_labels = torch.ones(batch_size, 1, device=device)
            loss_G = adversarial_loss(discriminator(fake_images), fake_labels)
            epoch_g_loss += loss_G.item()
            loss_G.backward()
            optimizer_G.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {(epoch_d_loss/len(train_loader)):.4f}, G Loss: {(epoch_g_loss/len(train_loader)):.4f}") #
        
        if (epoch)%10 == 0:
            save_generated_images(generator, epoch, z_dim, device)
            
    
    print('Training Finished')

def save_generated_images(generator, epoch, z_dim, device, save_dir="generated_images"):
    os.makedirs(save_dir, exist_ok=True)

    generator.eval()
    with torch.no_grad():
        z = torch.randn(16, z_dim, 1, 1, device=device)
        fake_images = generator(z).cpu().numpy()

    fake_images = (fake_images * 0.5 + 0.5)  # Rescale to [0, 1]
    fake_images = fake_images.transpose(0, 2, 3, 1)  # Convert to (N, 32, 32, 3)

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(fake_images[i])
        ax.axis("off")

    save_path = os.path.join(save_dir, f"epoch_{epoch+1}.png")
    plt.savefig(save_path)
    plt.close()
    generator.train()
                

if __name__ == "__main__":
    # hyperparemeters
    z_dim = 100
    img_channels = 3
    feature_dim = 64
    batch_size = 64
    lr_d = 2e-5
    lr_g = 5e-4
    b1 = 0.5
    b2 = 0.999
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


    # generator and discriminator
    generator = Generator(z_dim, img_channels, feature_dim).to(device)
    discriminator = Discriminator(img_channels=img_channels, feature_dim=feature_dim).to(device)

    # optimizer for generator and discriminator
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas = (b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(b1, b2))

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Data Loading and Preprocessing
    train_images, train_labels, test_images, test_labels = load_cifar100()
    
    train_dataset = CIFAR100Dataset(train_images, train_labels, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Visualize some training images
    # show_training_images(train_loader)

    train_dcgan(generator, discriminator, train_loader, num_epochs, z_dim, device)

    image_path = "generated_images/epoch_100.png"
    img = PIL.Image.open(image_path)
    plt.imshow(img)
    plt.axis("off")
    plt.show()