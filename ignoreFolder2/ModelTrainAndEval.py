from loadData import loadData
from model import LoRaCNN
from trainModel import train
from evalModel import evaluate_and_calculate_ser
import os
import torch
import torch.optim as optim
import torch.nn as nn
import logging

def ModelTrainAndEval(logger:logging.Logger, img_dir, output_folder, snr_list, rates, batch_size: int, M=128, optimizer="SGD", num_epochs=3, learning_rate=0.01, model="LoRaCNN"):
    criterion=nn.CrossEntropyLoss()

    modelFolder=os.path.join(output_folder, "modelFolder")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(modelFolder):
        os.makedirs(modelFolder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SERs=[[None]*len(rates) for _ in range(len(snr_list))]
    for snr in range(len(snr_list)):
        for rate in range(len(rates)):
            model=LoRaCNN(M).to(device)
            if optimizer_choice == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            elif optimizer_choice == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            else:
                print("No such optimizer is implemented")
                logger.error("No such optimizer is implemented")
                break


            train_loader, test_loader=loadData(img_dir, batch_size, snr_list[snr], rates[rate], M)
            train(model, train_loader, num_epochs, optimizer, criterion)
            torch.save(model.state_dict(), os.path.join(modelFolder, f"model_snr_{snr_list[snr]}_rate{rates[rate]}.pth"))
            ser=evaluate_and_calculate_ser(model, test_loader, criterion)
            SERs[snr][rate]=ser
            logger.info(f"Trained and evalulated model for SNR: {snr_list[snr]} and rate:{rates[rate]}. SER is {ser}")
    return SERs



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
    # snr_list = [i for i in range(-16, -2, 2)] # TODO change this to -16, -2, 2
    # rates = [0,0.25,0.5,0.7,1]
    snr_list=[i for i in range(-16, -12,2)]
    rates=[0,0.25]
    img_dir="output/20241030-093008/plots/"
    output_folder="output/"
    ModelTrainAndEval(logger, img_dir, output_folder, snr_list, rates, batch_size)
    logger.info("finnished model train and evaluation")