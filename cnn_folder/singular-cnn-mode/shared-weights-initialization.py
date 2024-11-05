"""
Here the different CNN models trained for each SNR values are merged into a combined
CNN model.
First, filter alignment is done before doing a shared weight initialization and fine -tunning it (retraining the model
with full SNR range dataset).
"""

import tensorflow as tf
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim


current_folder = os.path.dirname(os.path.abspath(__file__))  
cnn_folder_path = os.path.join(current_folder, '..')  
sys.path.append(cnn_folder_path)  


try:
    from cnn_train import LoRaCNN
    print("Successfully imported LoRaCNN from cnn_train.py")
except ImportError as e:
    print(f"Error importing LoRaCNN: {e}")
sys.path.append(cnn_folder_path)

from cnn_train import LoRaCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# Load trained models (don't know if file)
models_folder = './cnn_output/final_run/models'
model_files = [os.path.join(models_folder, f) for f in os.listdir(models_folder) if f.endswith('.pth')]
models = []
for file in model_files:
    model = LoRaCNN(128).to(device)  
    model.load_state_dict(torch.load(file))
    models.append(model)
# Cretation of a combined model using the architecture of the first model
model_combined = LoRaCNN(128).to(device)

# Iteration on layers of the combine model
for name, param in model_combined.named_parameters():
    # I dont know if the best is to use the naming for conv and dense layers
    if 'conv' in name or 'fc' in name: #according to the name of layers
        # obtain weights biases of ech layer
        weights_list = [model.state_dict()[name] for model in models]
        # calcualtion of the average weight and combine them
        averaged_weights = torch.mean(torch.stack(weights_list), dim=0)
        param.data = averaged_weights

# We save model
combined_model_path = os.path.join(models_folder, 'combined_model.pth')
torch.save(model_combined.state_dict(), combined_model_path)
print("Combined model saved")
