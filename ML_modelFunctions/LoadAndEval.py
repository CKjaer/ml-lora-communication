import torch
import torch.nn as nn
import os
import logging
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))
from ML_modelFunctions.evalModel import evaluate_and_calculate_ser
from ML_modelFunctions.loadData import loadData
from ML_modelFunctions.ML_models import *
from ML_modelFunctions.find_model import find_model


def loadAndevalModel(logger:logging.Logger, img_dir, output_dir, trained_model_folder, batch_size, snr_list:list, rates:list, base_model, M=128, seed=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # img_dir="output/20241030-093008/plots/"
    img_dir=os.path.join(output_dir, img_dir)
    modelFolder=os.path.join(output_dir, "models")
    # savedModel=os.path.join(modelFolder,trained_model)
    trained_models=os.listdir(trained_model_folder)
    # savedModel=trained_model
    criterion=nn.CrossEntropyLoss()
    M=2**7
    SERs=[[[None]*len(snr_list) for _ in range(len(rates))] for _ in range(len(trained_models))]
    for trained_model in range(len(trained_models)):
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
                    model.load_state_dict(torch.load(os.path.join(trained_model_folder,trained_models[trained_model]), weights_only=True))
                except Exception as e:
                    print(f"error loading model: {e}")
                    return
                _, test_loader=loadData(img_dir, batch_size, snr_list[snr], rates[rate], M, seed)
                ser=evaluate_and_calculate_ser(model, test_loader, criterion)
                SERs[trained_model][snr][rate]=ser
                logger.info(f"Evalulated {trained_models[trained_model]} for SNR: {snr_list[snr]} and rate:{rates[rate]}. SER is {ser}")
    return SERs, trained_models


if __name__=="__main__":
    logfilename="./LoadAndEval.log"
    logger=logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', 
                        filename=logfilename, 
                        encoding='utf-8', 
                        level=logging.INFO)
    logger.info("Starting model train and evaluation")
    SERs, trained_models = loadAndevalModel(logger=logger, img_dir="20241030-093008/plots/",output_dir="output/", trained_model_folder="C:/Users/lukas/Desktop/AAU/EIT7/Project/git/ml-lora-communication/ML_output/Test/models",batch_size=4,snr_list=[-14, -12],rates=[0.25, 0], base_model="LoRaCNN", seed=0)
    print(SERs, trained_models)
