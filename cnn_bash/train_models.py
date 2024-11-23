"""
This script trains CNN models using parameters specified in a configuration file (train_cnn_config.json).
The trained models and related data are saved in an output directory named with a unique test ID.
Usage:
    1. Edit the train_cnn_config.json file with the desired training parameters.
    2. Run the train_models.sh script which calls this file.
    3. The trained models and related data will be saved in the ~/cnn_output directory with a unique test ID.
Configuration:
    The script reads a configuration file (~/cnn_bash/train_cnn_config.json) with the following parameters:
    - test_id: A unique identifier for the test run. If empty, a timestamp will be used.
    - spreading_factor: Spreading factor of LoRa modulation.
    - train_dir: Directory containing the training data.
    - img_size: Size of the input images.
    - batch_size: Batch size for training.
    - snr_values: List of SNR values to use.
    - rate: List of arrival rates for interfering users.
    - model: Base model to be used for training.
    - optimizer: Optimizer choice for training.
    - num_epochs: Number of epochs for training.
    - learning_rate: Learning rate for training.
    - patience: Patience parameter for early stopping.
    - min_delta: Minimum change to qualify as an improvement for early stopping.
Output:
    - Trained models in the "models" subdirectory.
    - A log file with training details.
    - A copy of the configuration file used for training.
    - A .csv file with the estimated SER.
"""
import sys
import json
import os
import time
import logging
from numpy import savetxt
import numpy as np
import pandas as pd
import wandb
import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))
from model_includes.train_cnn import train_cnn
    
    
def sweep():
    wandb.init()
    logger.info(f"Starting sweep with config: {wandb.config}")

    # define hyperparameters
    num_symbols= wandb.config.num_symbols
    learning_rate = wandb.config.lr
    batch_size = wandb.config.batch_size
    num_epochs = wandb.config.num_epochs
    optimizer = wandb.config.optimizer
    
    # # define data parameters
    # snr = "-10.0"
    # rate = "0.0"
    
    # train model
    SERs=train_cnn(logger=logger,
                    train_dir=config["train_dir"],
                    img_size=config["img_size"],
                    output_folder=output_dir,
                    batch_size=batch_size,
                    snr_list=config['snr_values'],
                    rates=config['rate'],
                    base_model=config["model"],
                    M=num_symbols,
                    optimizer_choice=optimizer,
                    num_epochs=num_epochs,
                    learning_rate=learning_rate,
                    patience=config["patience"],
                    min_delta=config["min_delta"],
                    sweep=config["sweep"])
    
    pd.DataFrame(SERs, columns=config["snr_values"], index=config["rate"]).to_csv(os.path.join(data_dir, f"estimate_SER.csv"))
    logger.info("Finished training CNN models")
    
    
    
if __name__=="__main__":
    # Load config file and create output folders
    with open("cnn_bash/train_cnn_config.json") as f: #fix so dont have to be in root?
        config=json.load(f)
        
    
    if config["test_id"]!="" and config["sweep"] == False:
        test_id = config["test_id"]+"_"+time.strftime("%Y%m%d-%H%M%S")
    elif config["test_id"]=="" and config["sweep"] == False:
        test_id = time.strftime("%Y%m%d-%H%M%S")
    elif config["test_id"]!="" and config["sweep"] == True:
        test_id = "sweep_" + config["test_id"] + "_" + time.strftime("%Y%m%d-%H%M%S")
    elif config["test_id"]=="" and config["sweep"] == True:
        test_id = "sweep_" + time.strftime("%Y%m%d-%H%M%S") # cursed af
    
    output_dir = os.path.join("cnn_output", test_id)
    model_dir = os.path.join(output_dir, "models")
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Set up logging
    logfilename=test_id+".log"
    log_path=os.path.join(output_dir, logfilename)
    logger=logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_path,
        encoding="utf-8",
        level=logging.INFO,
    )
    # Save config file
    logger.info("save config file")
    config["test_id"] = test_id
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f)
    
    if config["sweep"] == False:
        # Train model with config parameters
        logger.info("Training CNN models...")
        SERs=train_cnn(logger=logger,
                            train_dir=config["train_dir"],
                            img_size=config["img_size"],
                            output_folder=output_dir,
                            batch_size=config["batch_size"],
                            snr_list=config["snr_values"],
                            rates=config["rate"],
                            base_model=config["model"],
                            M=2**config["spreading_factor"],
                            optimizer_choice=config["optimizer"],
                            num_epochs=config["num_epochs"],
                            learning_rate=config["learning_rate"],
                            patience=config["patience"],
                            min_delta=config["min_delta"],
                            sweep=config["sweep"])
        pd.DataFrame(SERs, columns=config["snr_values"], index=config["rate"]).to_csv(os.path.join(data_dir, f"estimate_SER.csv"))
        logger.info("Finished training CNN models")
        
    else:
        with open("cnn_bash/sweep_config.yaml") as f:
            sweep_config = yaml.safe_load(f)
            
        # dump config to output folder
        with open(os.path.join(output_dir, "sweep_config.yaml"), "w") as f:
            yaml.dump(sweep_config, f)
        
        #sweep_id = wandb.sweep(sweep_config, project="CNN")
        sweep_id = "zriwdki1"
        
        wandb.agent(sweep_id, function=sweep, count=10) # 10 sweeps
                        
    
    





