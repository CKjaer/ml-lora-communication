import os
import logging
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))
from model_includes.ML_models.LoRaCNN import LoRaCNN
from model_includes.ML_models.IQ_cnn import IQ_cnn
import torch

def find_model(model:str):
    ML_model_files=os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),"ML_models"))
    ML_models=[modeler.replace(".py", "",-1) for modeler in ML_model_files]
    
    for i in ML_models:
        if i==model:
            return i

if __name__=="__main__":
    model="LoRaCNN"
    print(find_model(model))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=getattr(sys.modules[__name__], model)
    M=2**7
    
    model(M).to(device)
    print(model)
    





