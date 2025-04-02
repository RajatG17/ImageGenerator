import torch 
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feature_dim=64):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, feature_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_dim ,feature_dim*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_dim*2, feature_dim*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim*4),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv2d(feature_dim*4, feature_dim*8, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(feature_dim*8),
            # nn.LeakyReLU(0.2, inplace=True),
          
            nn.Conv2d(feature_dim*4, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        ) 
    
    def forward(self, x):
        output = self.model(x).contiguous()
        output = output.view(output.shape[0], -1)  # Flatten the output
        # print(f"Discriminator Output Shape: {output.shape}")

        return output
