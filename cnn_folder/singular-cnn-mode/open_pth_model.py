import torch
import os

current_folder = os.path.dirname(os.path.abspath(__file__))

models_folder = os.path.join(current_folder, 'models')
# Cargar el archivo y mostrar sus claves
checkpoint = torch.load(models_folder, map_location='cpu')

if isinstance(checkpoint, dict):
    print("Keys in checkpoint:", checkpoint.keys())
else:
    print("Checkpoint does not contain a dictionary.")
