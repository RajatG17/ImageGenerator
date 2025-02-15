import torch
import os
import matplotlib.pyplot as plt
import PIL.Image
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from generators.generator_cnn import Generator
from discriminators.discriminator_cnn import Discriminator
from dataloaders.loader import load_cifar100, CIFAR100Dataset



def generate_and_save_images(generator, epoch, z_dim, save_dir="generated_images"):
    os.makedirs(save_dir, exist_ok=True)

    generator.eval()  # Set generator to evaluation mode
    with torch.no_grad():
        z = torch.randn(16, z_dim).to(device)  # Generate 16 random noise samples
        fake_images = generator(z).detach().cpu().numpy()

    # Reshape and denormalize images
    fake_images = (fake_images * 127.5 + 127.5).astype(np.float32)  # Scale back to [0, 255]
    fake_images = fake_images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # Reshape to (N, 32, 32, 3)

    # Plot images
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(fake_images[i])
        ax.axis("off")

    # Save image
    save_path = os.path.join(save_dir, f"epoch_{epoch+1}.png")
    plt.savefig(save_path)
    plt.close()
    generator.train()  # Set generator back to training mode


z_dim = 100

# Hyperparameters
z_dim = 100
img_shape = 3*32*32
batch_size = 64
lr = 5e-4
b1 = 0.5
b2 = 0.999
num_epochs = 2000
device = torch.device("cuda" if torch.cuda.is_available() else 'mps')
print(device)

generator = Generator(z_dim, img_shape).to(device)
discriminator = Discriminator(img_shape).to(device)


# optimizer for both generator and discriminator
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Data Loading and Preprocessing

# NOT USED HERE
# transform = transforms.Compose([
#     # transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     ])

train_images, train_labels, test_images, test_labels = load_cifar100()

train_dataset = CIFAR100Dataset(train_images, train_labels, transform=None)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# Visualize training data
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

show_training_images(train_loader)


# labels
real_label = 1
fake_label = 0

for epoch in range(num_epochs):
    for real_imgs, _ in train_loader:
        real_imgs = real_imgs.view(-1, img_shape).to(device) # Flattening images

        # Train Discriminator
        optimizer_D.zero_grad()
        real_preds = discriminator(real_imgs)
        real_loss = adversarial_loss(real_preds, torch.ones_like(real_preds)*real_label)

        z = torch.randn(real_imgs.size(0), z_dim).to(device)
        fake_imgs = generator(z)
        fake_preds = discriminator(fake_imgs.detach()) # detach to avoid backpropagation through generator
        fake_loss = adversarial_loss(fake_preds, torch.zeros_like(fake_preds))


        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        # Train Generator  
        optimizer_G.zero_grad()
        fake_preds = discriminator(fake_imgs)
        g_loss = adversarial_loss(fake_preds, torch.ones_like(fake_preds)*real_label)

        g_loss.backward()
        optimizer_G.step()

        if (epoch+1)%100 == 0:
            generate_and_save_images(generator, epoch, z_dim)

    print(f"Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item()}, g_loss: {g_loss.item()}")

print("Training finished")


image_path = "generated_images/epoch_500.png"
img = PIL.Image.open(image_path)
plt.imshow(img)
plt.axis("off")
plt.show()
