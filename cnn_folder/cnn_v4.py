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

# Configure logger
logfilename = "cnn.log"
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',datefmt='%Y-%m-%d %H:%M:%S', filename=logfilename, encoding='utf-8', level=logging.INFO)
logger.info("Starting the program")


# Define the CNN architecture
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
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten the output from conv layers
        x = x.view(-1, self.num_flat_features(x))
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.batchNorm3(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.batchNorm4(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # All dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Example model
M = 128
model = LoRaCNN(M).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, specific_label=None, transform=None, samples_per_label=250):
        self.img_dir = img_dir
        self.img_list = os.listdir(img_dir)
        self.transform = transform
        self.specific_label = specific_label
        self.samples_per_label = samples_per_label

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
        for label, images in label_image_dict.items():
            logger.info(f"Label: {label}, Number of images before sampling: {len(images)}")

        # Randomly sample images for each label
        self.img_list = []
        for label, images in label_image_dict.items():
            sampled_images = random.sample(images, min(self.samples_per_label, len(images)))
            self.img_list.extend(sampled_images)
            logger.info(f"Label: {label}, Number of images after sampling: {len(sampled_images)}")

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
            logger.error(f"Error loading image {img_name}: {e}")
            return None, None

 

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to a fixed size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize with mean and std
])


# Training function
def train(model, train_loader, num_epochs, optimizer, criterion):
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
                running_loss = 0.0


# Evaluation function
def evaluate_and_calculate_ser(model, test_loader, criterion):
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
    return ser


# Directory paths
img_dir = "./output/full_set_20241003-214909/plots/"
output_folder = './cnn_output/'

# Create the directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List of specific values for which SER will be calculated
specific_values = [i for i in range(-16, -2, 2)]

# Placeholder to store symbol error rates
symbol_error_rates = []

# Loop over each specific value
for value in specific_values:
    logger.info(f"Calculating SER for specific value: {value}")

    dataset = CustomImageDataset(img_dir=img_dir, specific_label=float(value), transform=transform, samples_per_label=250)
    logger.info(f"Number of images in dataset: {len(dataset)}")
    # Dataset size check
    if len(dataset) == 0:
        logger.warning(f"No images found for specific value: {value}. Skipping.")
        continue

    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = LoRaCNN(M).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train(model, train_loader, 3, optimizer, criterion)

    # Save model and optimizer
    torch.save(model.state_dict(), os.path.join(output_folder, f'model_{value}_snr.pth'))
    torch.save(optimizer.state_dict(), os.path.join(output_folder, f'optimizer_{value}_snr.pth'))

    # Evaluate model and calculate SER
    ser = evaluate_and_calculate_ser(model, test_loader, criterion)
    symbol_error_rates.append(ser)

logger.info("All SER values have been calculated.")

# Plotting SNR vs SER
plt.figure(figsize=(10, 6))
plt.plot(specific_values, symbol_error_rates, marker='o', linestyle='-', color='b')
plt.xlabel('SNR')
plt.ylabel('Symbol Error Rate (SER)')
plt.yscale('log')
plt.ylim(1e-5, 1e0)
plt.title('SNR vs Symbol Error Rate')
plt.grid(True)

# Save and show plot
final_plot_filename = os.path.join(output_folder, 'snr_vs_ser_final_plot.png')
plt.savefig(final_plot_filename)
plt.show()
plt.close()

print(f"Final plot has been saved to {final_plot_filename}.")
