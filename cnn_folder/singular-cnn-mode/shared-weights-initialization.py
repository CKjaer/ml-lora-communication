"""
Here the different CNN models trained for each SNR values are merged into a combined
CNN model.
First, filter alignment is done before doing a shared weight initialization and fine -tunning it (retraining the model
with full SNR range dataset).
"""
"""

import torch
import os

current_folder = os.path.dirname(os.path.abspath(__file__))
models_folder = os.path.join(current_folder, 'models')

model_files = [os.path.join(models_folder, f) for f in os.listdir(models_folder) if f.endswith('.pth')]

for file in model_files:
    print(f"Inspecting model: {file}")
    state_dict = torch.load(file, map_location='cpu')
    if isinstance(state_dict, dict):
        print(f"Keys in state_dict of {file}: {state_dict.keys()}")
    else:
        print(f"{file} is not a dictionary, might be a direct model.")


"""

from ast import mod
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_folder = os.path.dirname(os.path.abspath(__file__))
models_folder = os.path.abspath(os.path.join(current_folder, '../../cnn_output/batch_scaled/models'))

if not os.path.exists(models_folder):
    raise FileNotFoundError(f"Can't find the folder models: {models_folder}")

print(f"Carpeta 'models' encontrada en: {models_folder}")

# Only the snr models (not optimizer), remove the and part if we want to 
model_files = [os.path.join(models_folder, f) for f in os.listdir(models_folder) if f.endswith('.pth')  and 'optimizer' not in f]
models = []


for file in model_files:
    print(f"Loading model: {file}")
    state_dict = torch.load(file, map_location=device)
    models.append(state_dict)
print("Models loaded")

# write models into a txt file:
#with open('models.txt', 'w') as f:
#    for model in models:
#        f.write(f"{model}\n")
#print("Models saved in models.txt")

# We create a model with the same architecture as the ones we trained
model_combined = LoRaCNN(128).to(device)

# We go thorugh the layers of the combined model
for name, param in model_combined.named_parameters():
    # We filter the name of the layers to take only conv and fc
    if any(layer in name for layer in ['conv', 'fc']):
        # For each layer we take the weights of the models
        weights_list = [model[name] for model in models]
        # Stack the weights into a tensor and compute average along models dimension,
        # (Basically wight list is a list of tensors, each tensor is the weights of a layer in a model, an by averaging we compute the mean of each position)
        averaged_weights = torch.mean(torch.stack(weights_list), dim=0)
        # We assign the averaged weights to the combined model
        param.data = averaged_weights

# write averaged_weights
#with open('averaged_weights.txt', 'w') as f:
#    for weights in averaged_weights:
#        f.write(f"{weights}\n")

combined_model_path = os.path.join(current_folder, 'combined_model_initialization.pth')
torch.save(model_combined.state_dict(), combined_model_path)
print(f"Combined weights model saved at {combined_model_path}")



# write model_combined into a txt file:
# av_combined_model = [model_combined.state_dict()]
# with open('av_combined_model.txt', 'w') as f:
#    for model in av_combined_model:
#        f.write(f"{model}\n")

#Load and write the state_dict of the optimizers
# optimizer_files = [os.path.join(models_folder, f) for f in os.listdir(models_folder) if f.endswith('.pth') and 'optimizer' in f]
# optimizers = []

# for file in optimizer_files:
#     print(f"Loading optimizer: {file}")
#     state_dict = torch.load(file, map_location=device)
#     optimizers.append(state_dict)
# print("Optimizers loaded")

# #Write optimizers into a txt file:
# with open('optimizers.txt', 'w') as f:
#     for optimizer in optimizers:
#         f.write(f"{optimizer}\n")
# print("Optimizers saved in optimizers.txt")

