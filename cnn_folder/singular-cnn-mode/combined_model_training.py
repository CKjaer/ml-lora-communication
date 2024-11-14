import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import logging

# Logger configuration
logfilename = "combined_training.log"
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=logfilename,
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Check gpu availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class LoRaCNN(nn.Module):
    def __init__(self, M):
        super(LoRaCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=M//4, kernel_size=4, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=M//4, out_channels=M//2, kernel_size=4, stride=1, padding=2)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.batchNorm1 = nn.BatchNorm2d(M//4)
        self.batchNorm2 = nn.BatchNorm2d(M//2)
        self.fc1 = nn.Linear(M//2 * (M//4) * (M//4), 4 * M)
        self.fc2 = nn.Linear(4 * M, 2 * M)
        self.fc3 = nn.Linear(2 * M, 128)
        self.batchNorm3 = nn.BatchNorm1d(4 * M)
        self.batchNorm4 = nn.BatchNorm1d(2 * M)
        
    def forward(self, x):
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.batchNorm3(self.fc1(x)))
        x = F.relu(self.batchNorm4(self.fc2(x)))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class BinaryImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = np.loadtxt(file_path, delimiter=',')
        image = data.reshape(128, 128)  
        label = 0  
        if self.transform:
            image = self.transform(Image.fromarray(image))
        return image, label

# Transformation of the images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load dataset
data_dir = '/LoRa_Symbol_Detection/ml-lora-communication/output/batch_scaling_training_set_250_samples_20241025-132034'
dataset = BinaryImageDataset(data_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Load combined model
model_path = '/LoRa_Symbol_Detection/ml-lora-communication/cnn_output/combined_model_initialization.pth'
model = LoRaCNN(128).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

# Define optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()

# Training
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, _) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = torch.zeros(inputs.size(0), dtype=torch.long).to(device)  

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0

# Saved trained and combined model
trained_model_path = '/LoRa_Symbol_Detection/ml-lora-communication/cnn_output/trained_combined_model_trained.pth'
torch.save(model.state_dict(), trained_model_path)
logger.info(f"Modelo combinado entrenado guardado en {trained_model_path}")
