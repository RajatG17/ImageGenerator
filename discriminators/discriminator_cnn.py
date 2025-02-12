import numpy as np
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        ## experiment later, first develop a working baseline
        # self.model = nn.Sequential(
        #     nn.Linear(np.prod(img_shape), 512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(512, 256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(256, 1),
        #     nn.Sigmoid(),
        # )

        ## simple baseline model
        self.model = nn.Sequential(
            np.Flatten(),
            nn.Linear(np.prod(img_shape), 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    
    def forward(self, x):
        return self.fc(x)
