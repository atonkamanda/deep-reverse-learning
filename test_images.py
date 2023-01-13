import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import pathlib 
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.feature_extraction import get_graph_node_names,create_feature_extractor
import matplotlib.pyplot as plt


# Hyperparams   
batch_size = 60
# set device variable to cuda if cuda is available if not set to cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Loading MNIST
train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True,
                               )

test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              transform=transforms.ToTensor())


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           pin_memory=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          pin_memory=True)


# Display the first image of the training set
plt.imshow(train_dataset[0][0].numpy().squeeze(), cmap='viridis')
plt.show()

noise_injection = torch.randn(size=(1,28,28)) 
plt.imshow(noise_injection[0], cmap='viridis')
plt.show()