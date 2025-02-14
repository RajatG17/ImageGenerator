import numpy as np
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim, img_shape):
        super(Generator, self).__init__()
        
        self.img_shape = img_shape

        ## Experiment later, first develop a working baselie
        # self.model = nn.Sequential(
        #     nn.Linear(z_dim, 128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(128, 256),
        #     nn.BatchNorm1d(256, 0.8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(256, 514),
        #     nn.BatchNorm1d(512, 0.8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(512, np.prod(img_shape)),
        #     nn.Tanh()
        # )

        ## simple baseline model

        self.model = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024,img_shape),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)

        return img
