import torch
import torch.nn as nn
import os
import logging
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))
from model_includes.load_data import load_data
from model_includes.ML_models import *

def calculate_ser(device, model, test_loader):
    model.eval()  # Set model to evaluation mode
    total_predictions = 0
    incorrect_predictions = 0
    
    with torch.no_grad(): # Disable gradient calculation
        # Iterate over the  test set
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1) # Get the predicted labels

            # Calculate prediction statistics
            incorrect_predictions += (predicted != labels).sum().item()
            total_predictions += labels.size(0)

    ser = incorrect_predictions / total_predictions
    
    return ser

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


def test_model(logger:logging.Logger, test_dir, img_size:list, trained_model, snr_list:list, rates:list, base_model, M=128, device=device):
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dir=os.path.join(test_dir)
    criterion=nn.CrossEntropyLoss()
    SERs=[[None]*len(rates) for _ in range(len(snr_list))]

    model=find_model(base_model)
    if model!=None:
        try:
            model_class=getattr(sys.modules[__name__], model)
        except Exception as e:
            logger.error(f"error loading model: {e}")
            return
    else:
        logger.error("no such model is found")
        return
    for snr in range(len(snr_list)):
        for rate in range(len(rates)):
            try:
                model=model_class(M).to(device)
                model.load_state_dict(torch.load(os.path.join(trained_model), weights_only=True))
            except Exception as e:
                logger.error(f"error loading model: {e}")
                return
            test_loader=load_data(data_dir=test_dir, training=False, batch_size=None, SNR=snr_list[snr], rate_param=rates[rate], M=M, img_size=img_size)
            ser, _= calculate_ser(device, model, test_loader)
            SERs[snr][rate]=ser
            logger.info(f"Evalulated {trained_model} for SNR: {snr_list[snr]} and rate:{rates[rate]}. SER is {ser}")

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
    SERs, trained_models = test_model(logger=logger,
                                            test_dir="C:/Users/lukas/Desktop/AAU/EIT7/output/20241030-093008/plots",
                                            trained_model="C:/Users/lukas/Desktop/AAU/EIT7/Project/git/ml-lora-communication/ML_output/Test/models",
                                            snr_list=[-14, -12],
                                            rates=[0.25, 0], 
                                            base_model="LoRaCNN", 
                                            img_size=[128,128])
    
    print(SERs, trained_models)
