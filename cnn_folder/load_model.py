import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from cnn_folder.cnn_v1 import LoRaCNN # import CNN architecture from cnn_v1.py

def load_model(model_path, M):
    # Load the model
    model = LoRaCNN(M)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, specific_label=None, transform=None):
        self.img_dir = img_dir
        self.img_list = os.listdir(img_dir)
        self.transform = transform
        self.specific_label = specific_label

        if specific_label is not None:
            self.img_list = [img for img in self.img_list if float(img.split('_')[1]) == specific_label]

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to a fixed size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize with mean and std
])

if __name__ == "__main__":
    ...