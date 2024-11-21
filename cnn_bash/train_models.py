import sys
import json
import os
import time
import logging
from numpy import savetxt
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))
from model_includes.ModelTrainAndEval import ModelTrainAndEval

"""
Description:
    Trains CNN models with parameters given in train_cnn_config.json
Usage:
    Edit the train_cnn_config.json file and run the train_models.sh
    script calling this file. The trained models will be saved in 
    the ~/cnn_output with the test_id.
"""

if __name__=="__main__":
    # Load config file and create output folders
    with open("cnn_bash/train_cnn_config.json") as f: #fix so dont have to be in root?
        config=json.load(f)
    if config["test_id"]!="":
        test_id = config["test_id"]+"_"+time.strftime("%Y%m%d-%H%M%S")
    else:
        test_id = time.strftime("%Y%m%d-%H%M%S")
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
    
    # Train model with config parameters
    logger.info("Training CNN models...")
    SERs=ModelTrainAndEval(logger=logger,
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
                        min_delta=config["min_delta"])
    pd.DataFrame(SERs, columns=config["snr_values"], index=config["rate"]).to_csv(os.path.join(data_dir, f"estimate_SER.csv"))
    logger.info("Finished training CNN models")
                        
    
    





