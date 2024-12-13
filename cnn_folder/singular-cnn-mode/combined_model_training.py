import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import numpy as np
from scipy import stats

import logging

# Logger configuration
logfilename = "combined_testing.log"
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
        self.feature_size = self._get_conv_output((1, 128, 128))  
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.batchNorm3 = nn.BatchNorm1d(512)
        self.batchNorm4 = nn.BatchNorm1d(256)
        
    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.rand(1, *shape)
            output = self._forward_conv(input)
            return int(np.prod(output.size()))

    def _forward_conv(self, x):
        x = self.pool(F.relu(self.batchNorm1(self.conv1(x))))
        x = self.pool(F.relu(self.batchNorm2(self.conv2(x))))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  
        x = F.relu(self.batchNorm3(self.fc1(x)))
        x = F.relu(self.batchNorm4(self.fc2(x)))
        x = self.fc3(x)
        return x


class BinaryImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        image = Image.open(file_path).convert('L')  
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)  # Convert to tensor

        filename = os.path.basename(file_path)
        if "symbol_" in filename:
            label = int(filename.split('symbol_')[1].split('_')[0])  
        else:
            raise ValueError(f"The file {filename} has no 'symbol_' in naming.")
        
        label = torch.tensor(label, dtype=torch.long)
        return image, label, filename


current_folder = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(current_folder, '/ceph/project/LoRa_Symbol_Detection/ml-lora-communication/output/autoscaling_training_set_20241121-134229/plots'))
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to 128x128
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize data
])
dataset = BinaryImageDataset(data_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=64)

# Combined model initialization
model = LoRaCNN().to(device)
pth = os.path.abspath(os.path.join(current_folder, '/ceph/project/LoRa_Symbol_Detection/ml-lora-communication/singular-cnn-mode/combined_model_initialization.pth'))
model.load_state_dict(torch.load(pth, weights_only=True))
# Loss function and optimizer
optimizer = optim.Adam(model.parameters(), lr=0.015)
criterion = nn.CrossEntropyLoss()
"""
# Training
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels, _) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
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
logger.info(f"Trained model saved at {trained_combined_model_path}")
"""
model = LoRaCNN().to(device)
pth = os.path.abspath(os.path.join(current_folder, '/ceph/project/LoRa_Symbol_Detection/ml-lora-communication/singular-cnn-mode/trained_combined_model.pth'))
model.load_state_dict(torch.load(pth, weights_only=True))



# Evaluate model with SER vs SNR calculation
# Evaluate model with SER vs SNR calculation
def evaluate_model_by_rate_and_snr(model, data_loader):
    model.eval()
    ser_per_rate_snr = {}
    rates = ["0.0", "0.25", "0.5", "0.7", "1.0"]  # Define rates
    results_dir = "ser_vs_snr_results_by_rate"
    os.makedirs(results_dir, exist_ok=True)  # Create directory to store results
    logger.info(f"Starting evaluation of the model by rate and SNR...")

    with torch.no_grad():
        for inputs, labels, filenames in data_loader:
            logger.info("Processing a batch of data...")
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # Parse filenames to extract SNR and Rate
            for filename, label, pred in zip(filenames, labels, predicted):
                snr = float(filename.split("snr_")[1].split("_")[0])  # Extract SNR
                rate = filename.split("rate_")[1].split("_")[0]        # Extract Rate
                
                if rate not in ser_per_rate_snr:
                    ser_per_rate_snr[rate] = {}
                    logger.info(f"Initialized storage for rate {rate}.")
                if snr not in ser_per_rate_snr[rate]:
                    ser_per_rate_snr[rate][snr] = {"total": 0, "incorrect": 0}
                    logger.info(f"Initialized storage for SNR {snr} under rate {rate}.")

                ser_per_rate_snr[rate][snr]["total"] += 1
                if label != pred:
                    ser_per_rate_snr[rate][snr]["incorrect"] += 1
                    logger.debug(f"Incorrect prediction for SNR {snr}, rate {rate}: True label={label}, Predicted={pred}.")

    # Save results to separate .txt files for each rate
    for rate in rates:
        if rate in ser_per_rate_snr:
            snrs = sorted(ser_per_rate_snr[rate].keys())
            ser = [ser_per_rate_snr[rate][snr]["incorrect"] / ser_per_rate_snr[rate][snr]["total"] for snr in snrs]

            results_file = os.path.join(results_dir, f"ser_vs_snr_rate_{rate}.txt")
            with open(results_file, "w") as f:
                f.write("SNR (dB)\tSER\n")
                for snr_value, ser_value in zip(snrs, ser):
                    f.write(f"{snr_value:.2f}\t{ser_value:.6f}\n")
            logger.info(f"Saved SER vs SNR results for rate {rate} to {results_file}.")

    logger.info("Evaluation completed successfully.")
    return ser_per_rate_snr




test_data_dir = os.path.abspath(os.path.join(current_folder, '/ceph/project/LoRa_Symbol_Detection/ml-lora-communication/output/autoscaling_test_set_20241121-152225/plots'))
test_dataset = BinaryImageDataset(test_data_dir, transform=transform)
test_data_loader = DataLoader(test_dataset, batch_size=512, shuffle=True, num_workers=80)


# Evaluate model using the test dataset
ser_per_rate_snr=ser_per_rate_snr=evaluate_model_by_rate_and_snr(model, test_data_loader)


