import torch
import torch.nn as nn

class Generator(nn.Module):
    def  __init__(self, z_dim=100, img_channels=3, feature_dim=64):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, feature_dim*4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_dim*4),
            nn.ReLU(True),

            # nn.ConvTranspose2d(feature_dim*8, feature_dim*4, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(feature_dim*4),
            # nn.ReLU(True),

            nn.ConvTranspose2d(feature_dim*4, feature_dim*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_dim*2, feature_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_dim, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        output = self.model(x)
        # print(f"Generator Output Shape: {output.shape}") 
        return output