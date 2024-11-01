import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import PIL
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torchsummary import summary
import matplotlib.pyplot as plt
import logging

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the CNN architecture
class LoRaCNN(nn.Module):
    def __init__(self, M):
        super(LoRaCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=M//4, kernel_size=4, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=M//4, out_channels=M//2, kernel_size=4, stride=1, padding=2)
        
        # Pooling layer
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.batchNorm1=nn.BatchNorm2d(M//4)
        self.batchNorm2=nn.BatchNorm2d(M//2)
        self.batchNorm3=nn.BatchNorm1d(M*4)
        self.batchNorm4=nn.BatchNorm1d(M*2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(M//2 * (M//4) * (M//4), 4 * M) #nr parameters=(M//2 * (M//4) * (M//4)* 4 * M +4*M,      +4*M:biases?
        self.fc2 = nn.Linear(4 * M, 2 * M)
        self.fc3 = nn.Linear(2 * M, M)
        
    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.batchNorm1(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))  
        x = self.batchNorm2(x)
        x = self.pool(x)
        
        # Flatten the output from conv layers
        x = x.view(-1, self.num_flat_features(x))
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.batchNorm3(x)
        x = F.relu(self.fc2(x))
        x = self.batchNorm4(x)
        x = self.fc3(x)
        
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # All dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

M=128
model=LoRaCNN(M)
model=model.to(device)
summary(model, input_size=(1, M, M))
