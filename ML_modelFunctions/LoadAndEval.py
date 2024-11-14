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


def loadAndevalModel(logger:logging.Logger, train_dir, test_dir, img_size:list, output_dir, trained_model_folder, batch_size, snr_list:list, rates:list, base_model, M=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if train_dir!=None:
        train_dir=os.path.join(train_dir)
    test_dir=os.path.join(test_dir)
    trained_models=os.listdir(trained_model_folder)
    criterion=nn.CrossEntropyLoss()
    SERs=[[[None]*len(snr_list) for _ in range(len(rates))] for _ in range(len(trained_models))]
    for trained_model in range(len(trained_models)):
        model=find_model(base_model)
        if model!=None:
            try:
                model_class=getattr(sys.modules[__name__], model)
            except Exception as e:
                logger.error("error loading model")
                return
        else:
            logger.error("no such model is found")
            return
        for snr in range(len(snr_list)):
            for rate in range(len(rates)):
                try:
                    model=model_class(M).to(device)
                    model.load_state_dict(torch.load(os.path.join(trained_model_folder,trained_models[trained_model]), weights_only=True))
                except Exception as e:
                    logger.error(f"error loading model: {e}")
                    return
                _, test_loader=loadData(test_dir=test_dir, train_dir=None, batch_size=batch_size, SNR=snr_list[snr], rate_param=rates[rate], M=M, img_size=img_size)
                ser=evaluate_and_calculate_ser(model, test_loader, criterion)
                SERs[trained_model][rates][snr]=ser
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
    SERs, trained_models = loadAndevalModel(logger=logger, 
                                            train_dir=None, 
                                            test_dir="C:/Users/lukas/Desktop/AAU/EIT7/output/20241030-093008/plots",
                                            output_dir="output/", 
                                            trained_model_folder="C:/Users/lukas/Desktop/AAU/EIT7/Project/git/ml-lora-communication/ML_output/Test/models",
                                            batch_size=4,
                                            snr_list=[-14, -12],
                                            rates=[0.25, 0], 
                                            base_model="LoRaCNN", 
                                            img_size=[128,128])
    print(SERs, trained_models)
