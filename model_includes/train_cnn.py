import os
import torch
import torch.optim as optim
import torch.nn as nn
import logging
import sys
import wandb
import numpy as np
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))
from model_includes.load_data import load_data
from model_includes.ML_models import *
from model_includes.train_model import train

def find_model(model: str):
    """
    Searches for a machine learning model file in the "ML_models" directory and returns the model name if found.
    Args:
        model (str): The name of the model to search for.
    Returns:
        str: The name of the model if found, otherwise None.
    """

    ML_model_files = os.listdir(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ML_models")
    )
    ML_models = [modeler.replace(".py", "", -1) for modeler in ML_model_files]

    for i in ML_models:
        if i == model:
            return i

def train_cnn(logger:logging.Logger, train_dir, img_size, output_folder, snr_list:list, rates:list, batch_size: int, base_model:str, M=128, optimizer_choice="SGD", num_epochs=3, learning_rate=0.01, patience=5, min_delta=0.05, sweep=False, resume=False):
    criterion=nn.CrossEntropyLoss()
    #ensure the output directories exist
    saveModelFolder=os.path.join(output_folder, "models")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(saveModelFolder):
        os.makedirs(saveModelFolder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #initiate symbol error rate nested list
    SERs=[[None]*len(rates) for _ in range(len(snr_list))]
    

    #initiate progress files
    if resume and os.path.exists(os.path.join(output_folder, "progress.txt")):
        progress_txt=open(os.path.join(output_folder, "progress.txt"), "a")
        tested_models=open(os.path.join(output_folder, "progress.txt"), "r")
        tested_models=tested_models.readlines()
        data_txt=open(os.path.join(output_folder, "data.txt"), "a")
    else:
        progress_txt=open(os.path.join(output_folder, "progress.txt"), "w")
        data_txt=open(os.path.join(output_folder, "data.txt"), "w")

    #find the model architecture in the ML_models folder
    str_model=find_model(base_model)
    if str_model!=None:
        try:
            #get the model architecture for the given model
            model_class=getattr(sys.modules[__name__], str_model)
        except Exception as e:
            logger.error(f"No such model is found: {e}")
            return
    else:
        logger.error("no such class is found")
        print("error finding model")
        return
    # loop through all SNR and interferer rate combinations
    for snr in range(len(snr_list)):
        for rate in range(len(rates)):
            if resume: #if model has already been trained, continue
                if f"{snr_list[snr]}_{rates[rate]}\n" in tested_models:
                    continue
            try:
                #initiate new model
                model=model_class(M).to(device)
            except Exception as e:
                logger.error(f"error loading model: {e}")
                return
            if optimizer_choice == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            elif optimizer_choice == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            else:
                logger.error("No such optimizer is implemented")
                break


            train_loader, val_loader=load_data(data_dir=train_dir,
                                               training=True,
                                               batch_size=batch_size, 
                                               SNR=snr_list[snr], 
                                               rate_param=rates[rate], 
                                               #M=M, # I guess this parameter has been removed? 
                                               img_size=img_size)

            ser=train(model, train_loader, num_epochs, optimizer, criterion, val_loader, logger=logger, patience=patience, min_delta=min_delta, sweep=sweep)
            torch.save(model.state_dict(), os.path.join(saveModelFolder, f"{str_model}_snr_{snr_list[snr]}_rate_{rates[rate]}.pth"))
            data_txt.write(f"{snr_list[snr]}_{rates[rate]}_SER_{ser}\n")
            progress_txt.write(f"{snr_list[snr]}_{rates[rate]}\n") # update progress file
            logger.info(f"Trained and evaluated model for SNR: {snr_list[snr]} and rate:{rates[rate]}. SER is {ser}")
            
            if sweep:
                wandb.log({"final_accuracy": (1 - ser) * 100})
            
            SERs[snr][rate]=ser 
    return SERs



if __name__=="__main__":
    logfilename="./train_cnn.log"
    logger=logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', 
                        filename=logfilename, 
                        encoding='utf-8', 
                        level=logging.INFO)
    logger.info("Starting model train and evaluation")
    print("Running model train and evaluation")
    batch_size=4
    optimizer_choice="SGD"
    learning_rate=0.02
    num_epochs=3
    M=2**7

    snr_list=[i for i in range(-16, -12,2)]
    rates=[0,0.25]
    img_dir="C:/Users/lukas/Desktop/AAU/EIT7/output/20241030-093008/plots"
    output_folder="output/"
    train_cnn(logger=logger, 
                      test_dir=img_dir,
                      train_dir=img_dir,
                      img_size=[128,128],
                      output_folder=output_folder, 
                      snr_list=snr_list, 
                      rates=rates, 
                      batch_size=batch_size, 
                      base_model="LoRaCNN")
    logger.info("finnished model train and evaluation")