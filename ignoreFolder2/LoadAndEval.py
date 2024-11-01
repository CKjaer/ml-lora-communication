from evalModel import evaluate_and_calculate_ser
from loadData import loadData
from model import LoRaCNN
import torch
import torch.nn as nn
import os

def loadAndevalModel(img_dir, model_name, batch_size, SNR, rate_param, M=128, seed=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # img_dir="output/20241030-093008/plots/"
    output_folder="output/"
    img_dir=os.path.join(output_folder, img_dir)
    modelFolder=os.path.join(output_folder, "modelFolder")
    savedModel=os.path.join(modelFolder,model_name)

    _, test_loader=loadData(img_dir, batch_size, SNR, rate_param, M, seed)

    criterion=nn.CrossEntropyLoss()
    M=2**7
    model=LoRaCNN(M).to(device)
    model.load_state_dict(torch.load(savedModel, weights_only=True))

    evaluate_and_calculate_ser(model, test_loader, criterion)


loadAndevalModel("20241030-093008/plots/", "model_snr_-14_rate0.25.pth",4,-14,0.25)