import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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

# GPU verification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Defining model architecture
class LoRaCNN(nn.Module):
    def __init__(self):
        super(LoRaCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=2)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.batchNorm2 = nn.BatchNorm2d(64)

        # FC layer dynamic input size
        self.feature_size = self._get_conv_output((1, 2, 128))  
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.batchNorm3 = nn.BatchNorm1d(512)
        self.batchNorm4 = nn.BatchNorm1d(256)
        
    def _get_conv_output(self, shape):
        """Calcula el tamaño de salida de las capas convolucionales."""
        with torch.no_grad():
            input = torch.rand(1, *shape)
            output = self._forward_conv(input)
            return int(np.prod(output.size()))

    def _forward_conv(self, x):
        """Aplicar las capas convolucionales y de pooling."""
        x = self.pool(F.relu(self.batchNorm1(self.conv1(x))))
        x = self.pool(F.relu(self.batchNorm2(self.conv2(x))))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Aplanar
        x = F.relu(self.batchNorm3(self.fc1(x)))
        x = F.relu(self.batchNorm4(self.fc2(x)))
        x = self.fc3(x)
        return x


class BinaryImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = np.loadtxt(file_path, delimiter=';')
        data = data.reshape(1, 2, 128)  
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(0, dtype=torch.long)  
        return data, label


current_folder = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_folder, 'csv')
transform = transforms.Compose([
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalizar los datos
])
dataset = BinaryImageDataset(data_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Combined model inicialization
model = LoRaCNN().to(device)

# Loss function and optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# training
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {running_loss/10:.4f}")
            running_loss = 0.0

# Save trained model
trained_combined_model_path = os.path.join(current_folder, 'trained_combined_model.pth')
torch.save(model.state_dict(), trained_combined_model_path)
logger.info(f"Modelo combinado entrenado guardado en {trained_combined_model_path}")

#SAve the model in a txt file
with open('trained_model_combined.txt', 'w') as f:
    f.write(f"{model}\n")
print("Modelo guardado en model_combined.txt")

# Save state_dict of model in txt file  
# write model_combined into a txt file:
av_combined_model = [model.state_dict()]
#write av_combined_model into a txt file:
with open('trained_model_combined.txt', 'w') as f:
    for model in av_combined_model:
        f.write(f"{model}\n")




