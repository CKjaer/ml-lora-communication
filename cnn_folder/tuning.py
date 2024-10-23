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
import matplotlib.pyplot as plt
import logging
import numpy as np
import random
import wandb
import yaml

"""
The purpose of this script is to conduct a sweep of hyperparameters for the CNN model.

It only considers a single snr condition, as the assumption is that the best hyperparameters for one snr condition will be the best for all snr conditions.

The configuration of hyperparameters can be found in sweep_config.yaml.
"""

############# model architecture #############
class LoRaCNN(nn.Module):
    def __init__(self, M):
        super(LoRaCNN, self).__init__()
        
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=M//4, kernel_size=4, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=M//4, out_channels=M//2, kernel_size=4, stride=1, padding=2)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.batchNorm1 = nn.BatchNorm2d(M//4)
        self.batchNorm2 = nn.BatchNorm2d(M//2)
        
        # Fully connected layers with batch normalization
        self.fc1 = nn.Linear(M//2 * (M//4) * (M//4), 4 * M)
        self.fc2 = nn.Linear(4 * M, 2 * M)
        self.fc3 = nn.Linear(2 * M, 128)  # Output size is 128 for 128 labels
        self.batchNorm3 = nn.BatchNorm1d(4 * M)
        self.batchNorm4 = nn.BatchNorm1d(2 * M)
        
    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = self.pool(x)
        
        # Flatten the output from conv layers
        x = x.view(-1, self.num_flat_features(x))
        
        # Fully connected layers
        x = F.relu(self.batchNorm3(self.fc1(x)))
        x = F.relu(self.batchNorm4(self.fc2(x)))
        x = self.fc3(x)
        
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # All dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


############# data processing #############
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, logger, samples_per_label, specific_label=None, transform=None, ):
        self.img_dir = img_dir
        self.img_list = os.listdir(img_dir)
        self.transform = transform
        self.specific_label = specific_label
        self.samples_per_label = samples_per_label
        self.logger = logger

        # Filter images based on specific label (if provided)
        if specific_label is not None:
            self.img_list = [img for img in self.img_list if float(img.split('_')[1]) == specific_label]

        logger.info(f"Total images after filtering by specific label {specific_label}: {len(self.img_list)}")

        # Group images by label
        label_image_dict = {}
        for img in self.img_list:
            label = int(img.split('_')[3])  # Assuming label is after 'class_'
            if label not in label_image_dict:
                label_image_dict[label] = []
            label_image_dict[label].append(img)

        # Log the number of images for each label
        # for label, images in label_image_dict.items():
            #logger.info(f"Label: {label}, Number of images before sampling: {len(images)}")

        # Randomly sample images for each label
        self.img_list = []
        for label, images in label_image_dict.items():
            sampled_images = random.sample(images, min(self.samples_per_label, len(images)))
            self.img_list.extend(sampled_images)
            #logger.info(f"Label: {label}, Number of images after sampling: {len(sampled_images)}")

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name)

        try:
            # Load image and extract label from filename
            image = Image.open(img_path).convert("L")  # Convert to grayscale if necessary
            label = int(img_name.split('_')[3])  # Assuming label is after 'class_'

            # Apply transformations
            if self.transform:
                image = self.transform(image)

            label = torch.tensor(label, dtype=torch.long)

            return image, label

        except (PIL.UnidentifiedImageError, IndexError, FileNotFoundError) as e:
            self.logger.error(f"Error loading image {img_name}: {e}")
            return None, None

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to a fixed size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize with mean and std
])
  
  
############# training #############
def train(model, train_loader, num_epochs, optimizer, criterion, logger, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:
                logger.info(f'Epoch [{epoch+1}], Step [{i+1}], Loss: {running_loss / 100:.4f}')
                
                wandb.log({'Epoch': epoch+1, 'Step': i+1, 'Loss': np.round(running_loss / 100, 4)})
                running_loss = 0.0

############# evaluation #############
def evaluate_and_calculate_ser(model, test_loader, criterion, logger, device):
    model.eval()  # Set model to evaluation mode
    correct_predictions = 0
    total_predictions = 0
    total_loss = 0.0
    incorrect_predictions = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            incorrect_predictions += (predicted != labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = 100 * correct_predictions / total_predictions
    ser = incorrect_predictions / total_predictions
    average_loss = total_loss / len(test_loader)

    logger.info(f'Validation/Test Loss: {average_loss:.4f}')
    logger.info(f'Validation/Test Accuracy: {accuracy:.2f}%')
    logger.info(f'Symbol Error Rate (SER): {ser:.6f}')
    
    wandb.log({'Validation/Test Loss': np.round(average_loss, 4)})
    wandb.log({'Validation/Test Accuracy': np.round(accuracy, 2)})
    
    return accuracy

def main():
    # load sweep config
    wandb.init()
    # Directory paths
    img_dir = "./output/full_set_20241003-214909/plots/"
    output_folder = './cnn_output/'
    
    # Create the directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Configure logger
    logfilename = f"{__name__}.log"
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',datefmt='%Y-%m-%d %H:%M:%S', filename=os.path.join(output_folder, logfilename), encoding='utf-8', level=logging.INFO)
    logger.info("Starting the program")
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # TODO change this to whatever condition we want to tune on
    snr_value = -10
    
    # define hyperparameters
    num_epochs = wandb.config.num_epochs
    M = wandb.config.num_symbols
    learning_rate = wandb.config.lr
    batch_size = wandb.config.batch_size
    optimizer = wandb.config.optimizer
    
    # Load the dataset
    dataset = CustomImageDataset(img_dir=img_dir, logger=logger, samples_per_label=250,specific_label=float(snr_value), transform=transform)
    
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = LoRaCNN(M).to(device) 
    if optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        
    
    criterion = nn.CrossEntropyLoss()
    
    
    # Train the model
    train(model, train_loader, num_epochs, optimizer, criterion, logger, device)
    
    # Evaluate the model
    accuracy = evaluate_and_calculate_ser(model, test_loader, criterion, logger, device)
    logger.info(f"Final accuracy: {accuracy}")
    
    wandb.log({'Final accuracy': accuracy})

if __name__ == "__main__":
    #load sweep_config.yaml
    with open("./cnn_folder/sweep_config.yaml") as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(sweep=params, project="CNN")
    
    wandb.agent(sweep_id, function=main, count=10)