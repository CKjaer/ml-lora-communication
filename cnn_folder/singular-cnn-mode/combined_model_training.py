
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import logging

# Configurar Logger
logfilename = "combined_training.log"
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filename=logfilename,
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Verificar GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Definir la arquitectura del modelo combinado
class LoRaCNN(nn.Module):
    def __init__(self, M):
        super(LoRaCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=M//4, kernel_size=4, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=M//4, out_channels=M//2, kernel_size=4, stride=1, padding=2)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.batchNorm1 = nn.BatchNorm2d(M//4)
        self.batchNorm2 = nn.BatchNorm2d(M//2)

        # Ajustar `fc1` para que coincida con la salida de las capas convolucionales
        self.fc1 = nn.Linear(2048, 4 * M)
        self.fc2 = nn.Linear(4 * M, 2 * M)
        self.fc3 = nn.Linear(2 * M, 128)
        self.batchNorm3 = nn.BatchNorm1d(4 * M)
        self.batchNorm4 = nn.BatchNorm1d(2 * M)
        
    def forward(self, x):
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = self.pool(x)
        
        print(f"Tamaño después de la última capa conv/pool: {x.shape}")  # Confirmar las dimensiones
        
        # Aplanar la salida para que coincida con `fc1`
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.batchNorm3(self.fc1(x)))
        x = F.relu(self.batchNorm4(self.fc2(x)))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # Excluir la dimensión del batch
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# Dataset para cargar los datos binarios
class BinaryImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = np.loadtxt(file_path, delimiter=';')
        print(f"File format of {file_path}: {data.shape}")
        data = data.reshape(1, 2, 128)  # Ajustar dimensiones según los datos
        data = torch.tensor(data, dtype=torch.float32)
        label = 0  # Etiqueta dummy, no necesaria en este contexto
        return data, label

# Cargar el Dataset
current_folder = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_folder, 'csv')
transform = transforms.Compose([
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalizar los datos
])
dataset = BinaryImageDataset(data_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Cargar el modelo combinado inicial
model_path = os.path.join(current_folder, 'combined_model_initialization.pth')
model = LoRaCNN(128).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

# Definir optimizador y función de pérdida
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Entrenamiento
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, _) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = torch.zeros(inputs.size(0), dtype=torch.long).to(device)  # Etiquetas dummy

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}], Loss: {running_loss/10:.4f}")
            running_loss = 0.0

# Guardar el modelo entrenado
trained_combined_model_path = os.path.join(current_folder, 'trained_combined_model_initialization.pth')
torch.save(model.state_dict(), trained_combined_model_path)
logger.info(f"Modelo combinado entrenado guardado en {trained_combined_model_path}")
