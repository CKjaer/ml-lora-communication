import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torchvision.transforms as transforms
import torch.nn.functional as F
from cnn_v3 import LoRaCNN, CustomImageDataset


# Define the CNN architecture
class LoRaCNNDebug(LoRaCNN):
    def __init__(self, img_size):
        super(LoRaCNNDebug, self).__init__(img_size)
        self.print_plots = False
    
    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        self.print_cnn_activations(x,"conv 1")
        x = F.relu(self.batchNorm1(x))
        x = self.pool(x)
        self.print_cnn_activations(x, "pool 1")
        x = self.conv2(x)
        self.print_cnn_activations(x, "conv 2")
        x = F.relu(self.batchNorm2(x))
        x = self.pool(x)
        self.print_cnn_activations(x, "pool 2")
        # Flatten the output from conv layers
        x = x.view(-1, self.num_flat_features(x))
        
        # Fully connected layers
        x = F.relu(self.batchNorm3(self.fc1(x)))
        self.print_fully_connected(x, "fc 1")
        x = F.relu(self.batchNorm4(self.fc2(x)))
        self.print_fully_connected(x, "fc 2")
        x = self.fc3(x)
        return x

    def print_cnn_activations(self, x, title):
        if (self.print_plots):
            n = np.sqrt(x.shape[1])
            n = np.ceil(n).astype(int)
            plt.subplots(n, n)
            for i in range(x.shape[1]):
                plt.subplot(n, n, i+1)
                plt.imshow(x[0, i, :, :].detach().numpy(), cmap='gray')
            plt.suptitle(title)
            plt.show()

    def print_fully_connected(self, x, title):
        l = np.arange(x.shape[1])
        if (self.print_plots):
            plt.plot(l, x[0].detach().numpy(), 'o-')
            plt.title(title)
            plt.show()

# Evaluation function
def evaluate_and_calculate_ser(model, test_loader, criterion):
    #model.eval()  # Set model to evaluation mode
    correct_predictions = 0
    total_predictions = 0
    total_loss = 0.0
    incorrect_predictions = 0
    i = 0

    with torch.no_grad():
        for data in test_loader:    
            print(f"Batch {i}")
            i += 1
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if True:
                print(f"inputs shape: {inputs.shape}")
                print(f"outputs shape: {outputs.shape}")
                tsm = torch.softmax(outputs.data,dim=1)
                print(f"softmax shape: {tsm.shape}")
                print(f"Label: {labels}, Predicted: {torch.argmax(tsm)}")
                #plt.subplots(2,1,subplot_kw=dict(box_aspect=1))
                #plt.subplot(2,1,1)
                plt.imshow(inputs[0][0].detach().numpy())
                #plt.subplot(2,1,2)
                #for i in range(len(tsm[0])):
                    #print(f"[{i}: {tsm[0][i]:.3f}]",end = ", ")
                plt.plot(tsm[0]*128,c='w')
                plt.title(f"Label: {labels}, Predicted: {torch.argmax(tsm)}")
                plt.show()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            incorrect_predictions += (predicted != labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = 100 * correct_predictions / total_predictions
    ser = incorrect_predictions / total_predictions
    average_loss = total_loss / len(test_loader)

    print(f"Correct predictions: {correct_predictions}")
    print(f"Incorrect predictions: {incorrect_predictions}")
    print(f"Total predictions: {total_predictions}")
    print(f'Validation/Test Loss: {average_loss:.4f}')
    print(f'Validation/Test Accuracy: {accuracy:.2f}%')
    print(f'Symbol Error Rate (SER): {ser:.6f}')
    return ser

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to a fixed size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize with mean and std
])

dir = os.path.dirname(__file__)
model = "model_snr_-10_rate0.pth"
model_path = os.path.join(dir, model)
optimizer = "optimizer_snr_-10_rate_0.pth"
optimizer_path = os.path.join(dir, optimizer)
device=torch.device('cpu')

model = LoRaCNNDebug(128)
model.print_plots = True

optimizer = optim.Adam(model.parameters(), lr=0.001)
model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))
optimizer.load_state_dict(torch.load(optimizer_path, map_location=device,weights_only=True))

# Directory paths
img_dir = "C:\\Users\\rdybs\\Desktop\\gitnstuff\\ml-lora-communication\\ser_includes\\output\\20241031-100117\\plots"
# List of snr and rate parameters for which SER will be calculated
specific_values = [i for i in range(-10, -2, 2)] # TODO change this to -16, -2, 2
rates = [0,0.25,0.5,0.7,1]
specific_values = [-6]
rates = [0]

# Placeholder to store symbol error rates
symbol_error_rates = {} # dictionary to store SER for each rate
for rate in rates:
    symbol_error_rates[rate] = []

for i, param in enumerate(model.named_parameters()):
    print(f"Parameter {i}: {param[0]}")
    print(f"Parameter {i} shape: {param[1].shape}")
    if len(param[1].shape) == 4:
        if(param[1].shape[1] == 1):
            plt.subplots(8, 4)
            for j in range(8):
                for k in range(4):
                    plt.subplot(8, 4, j + k +1)
                    plt.imshow(param[1][j][k].detach().numpy(), cmap='gray')
                    plt.title(f"Kernel {j}")
                    plt.colorbar()
            plt.show()
        if(param[1].shape[1] == 32):
            plt.subplots(8, 8)
            for j in range(8):
                for k in range(8):
                    plt.subplot(8, 8, j + k + 1)
                    plt.imshow(param[1][j][k].detach().numpy(), cmap='gray')
                    plt.title(f"Kernel {j}")
            plt.colorbar()
            plt.show()
        if len(param[1].shape) == 2:
            plt.plot(param[1].detach().numpy())
            plt.title(f"Parameter {i}")
            plt.show()

exit()


model.eval()
# Loop over each specific value
for value in specific_values:
    for rate in rates:
        print(f"Calculating SER for specific value: {value}, rate {rate}")
        dataset = CustomImageDataset(img_dir=img_dir, specific_label=float(value), rate_param=float(rate), transform=transform, samples_per_label=200)
        print(f"Number of images in dataset: {len(dataset)}")
        # Dataset size check
        if len(dataset) == 0:
            print(f"No images found for specific value: {value}. Skipping.")
            continue

        criterion = nn.CrossEntropyLoss()

        #train, testset = random_split(dataset, [int(0.8 * len(dataset)), int(0.2 * len(dataset))])
        testset = dataset
        test_loader = DataLoader(testset, batch_size=32, shuffle=True)
        print(f"Number of batches in test loader: {len(test_loader)}")
        # Evaluate model and calculate SER
        ser = evaluate_and_calculate_ser(model, test_loader, criterion)
        symbol_error_rates[rate].append((ser, value)) # store SER and SNR value in corresponding rate
        
