import os
import torch
import torch.optim as optim
import torch.nn as nn
import logging
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))
from model_includes.loadData import loadData
from model_includes.ML_models import *
from model_includes.trainModel import train
from model_includes.evalModel import evaluate_and_calculate_ser
from model_includes.find_model import find_model

def ModelTrainAndEval(logger:logging.Logger, train_dir, test_dir, img_size, output_folder, snr_list:list, rates:list, batch_size: int, base_model:str, M=128, optimizer_choice="SGD", num_epochs=3, learning_rate=0.01):
    criterion=nn.CrossEntropyLoss()

    saveModelFolder=os.path.join(output_folder, "models")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(saveModelFolder):
        os.makedirs(saveModelFolder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SERs=[[[None]*len(snr_list) for _ in range(len(rates))]]

    str_model=find_model(base_model)
    if str_model!=None:
        try:
            model_class=getattr(sys.modules[__name__], str_model)
        except Exception as e:
            logger.error(f"no such model is found: {e}")
            return
    else:
        logger.error("no such class is found")
        print("error finding model")
        return

    for snr in range(len(snr_list)):
        for rate in range(len(rates)):
            try:
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


            train_loader, test_loader=loadData(train_dir=train_dir, 
                                               test_dir=test_dir, 
                                               batch_size=batch_size, 
                                               SNR=snr_list[snr], 
                                               rate_param=rates[rate], 
                                               M=M, 
                                               img_size=img_size)

            ser=train(model, train_loader, num_epochs, optimizer, criterion, test_loader=test_loader, logger=logger)
            torch.save(model.state_dict(), os.path.join(saveModelFolder, f"{str_model}_snr_{snr_list[snr]}_rate{rates[rate]}.pth"))
            logger.info(f"Trained and evalulated model for SNR: {snr_list[snr]} and rate:{rates[rate]}. SER is {ser}")
            SERs[0][rate][snr]=ser 
    return SERs, [base_model]



if __name__=="__main__":
    logfilename="./ModelTrainAndEval.log"
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
    ModelTrainAndEval(logger=logger, 
                      test_dir=img_dir,
                      train_dir=img_dir,
                      img_size=[128,128],
                      output_folder=output_folder, 
                      snr_list=snr_list, 
                      rates=rates, 
                      batch_size=batch_size, 
                      base_model="LoRaCNN")
    logger.info("finnished model train and evaluation")