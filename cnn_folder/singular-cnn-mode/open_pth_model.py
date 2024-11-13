import torch
import os

current_folder = os.path.dirname(os.path.abspath(__file__))

models_combined = os.path.join(current_folder, 'opent_pth_model.py')
# Cargar el archivo y mostrar sus claves
checkpoint = torch.load(current_folder)

if isinstance(checkpoint, dict):
    print("Keys in checkpoint:", checkpoint.keys())
else:
    print("Checkpoint does not contain a dictionary.")
