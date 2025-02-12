import os
import numpy as np
import torch

from unpickle import unpickle

class CIFAR100Dataset:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label
    



def load_cifar100():
    data_dir = './dataset/cifar-100-python/'

    # Full paths to the train and test files
    train_file = os.path.join(data_dir, 'train')
    test_file = os.path.join(data_dir, 'test')

    train_data = unpickle(train_file)
    test_data = unpickle(test_file)

    # print("Keys in train data:", train_data.keys())
    # print("Keys in test data:", test_data.keys())

    # preprocess the data
    train_images, train_labels = preprocess_data(train_data)
    test_images, test_labels = preprocess_data(test_data)

    return train_images, train_labels, test_images, test_labels

def preprocess_data(data_dict):
    images = data_dict[b'data']
    labels = data_dict[b'fine_labels']

    # Normalize the data
    data = data / 255.0
    data = data.reshape(-1, 3, 32, 32)
    data = data.transpose(0, 2, 3, 1)

    return images, labels

