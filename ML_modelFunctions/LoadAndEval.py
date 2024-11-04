from evalModel import evaluate_and_calculate_ser
from loadData import loadData
from ML_models import *
# from ML_models.LoRaCNN import LoRaCNN
# from ML_models.IQ_cnn import IQ_cnn
from find_model import find_model
import torch
import torch.nn as nn
import os
import logging
import sys


def loadAndevalModel(logger:logging.Logger, img_dir, output_dir, trained_model, batch_size, snr_list:list, rates:list, base_model, M=128, seed=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # img_dir="output/20241030-093008/plots/"
    img_dir=os.path.join(output_dir, img_dir)
    modelFolder=os.path.join(output_dir, "modelFolder")
    savedModel=os.path.join(modelFolder,trained_model)

    criterion=nn.CrossEntropyLoss()
    M=2**7
    SERs=[[None]*len(rates) for _ in range(len(snr_list))]

    model=find_model(base_model)
    if model!=None:
        try:
            model_class=getattr(sys.modules[__name__], model)
        except Exception as e:
            print("error loading model")
            return
    else:
        print("no such class is found")
        return
    for snr in range(len(snr_list)):
        for rate in range(len(rates)):
            try:
                model=model_class(M).to(device)
                model.load_state_dict(torch.load(savedModel, weights_only=True))
            except Exception as e:
                print(f"error loading model: {e}")
                return
            _, test_loader=loadData(img_dir, batch_size, snr_list[snr], rates[rate], M, seed)
            ser=evaluate_and_calculate_ser(model, test_loader, criterion)
            SERs[snr][rate]=ser
            logger.info(f"Evalulated model for SNR: {snr_list[snr]} and rate:{rates[rate]}. SER is {ser}")
    return SERs


if __name__=="__main__":
    logfilename="./LoadAndEval.log"
    logger=logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', 
                        filename=logfilename, 
                        encoding='utf-8', 
                        level=logging.INFO)
    logger.info("Starting model train and evaluation")
    loadAndevalModel(logger=logger, img_dir="20241030-093008/plots/",output_dir="output/", trained_model="model_snr_-14_rate0.25.pth",batch_size=4,snr_list=[-14],rates=[0.25], base_model="LoRaCNN")