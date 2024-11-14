import torch
import torch.nn as nn
import os
import logging
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))
from model_includes.evalModel import evaluate_and_calculate_ser
from ser_includes.load_files import load_data
from model_includes.ML_models import *
from model_includes.find_model import find_model
from torch.utils.data import random_split, DataLoader
import pandas as pd


# Load the IQ data from .csv files
# logger:logging.Logger, train_dir, test_dir, batch_size, SNR, rate_param, M=2**7
def load_csv(logger:logging.Logger):

    # Loads the IQ data as a pandas dataframe
    file_dir = os.path.join("output")
    dataset = load_data(file_dir, logger, header="iq")

    # Slice the data into training and testing sets
    train_size=int(len(dataset)*0.8)
    test_size=len(dataset)-train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # transform = transforms.Compose([
    #     transforms.Resize((img_size[0], img_size[1])),  # Resize images to a fixed size
    #     transforms.ToTensor(),  # Convert to tensor
    #     transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize with mean and std
    # ])




def evaluate_model(logger:logging.Logger, train_dir, test_dir, img_size:list, output_dir, trained_model_folder, batch_size, snr_list:list, rates:list, base_model, M=128):
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
                _, test_loader=load_csv(logger=logger)
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

    load_csv(logger=logger)
    SERs, trained_models = evaluate_model(logger=logger, 
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
