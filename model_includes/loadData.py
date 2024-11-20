import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))
from model_includes.imageDataSet import CustomImageDataset
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
import torch


def loadData(data_dir, training:bool, batch_size, SNR, rate_param, img_size:list, M=2**7):

    transform = transforms.Compose([
        transforms.Resize((img_size[0], img_size[1])),  # Resize images to a fixed size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize with mean and std
    ])


    dataset =CustomImageDataset(img_dir=data_dir, specific_label=SNR, rate_param=rate_param, transform=transform)
    if training==True:
        #for training
        train_size=int(len(dataset)*0.8)
        test_size=len(dataset)-train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
    else:
        #for testing
        train_loader = DataLoader(dataset, shuffle=True)
        return train_loader