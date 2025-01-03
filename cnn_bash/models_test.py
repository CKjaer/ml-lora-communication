"""
This script tests CNN models with parameters specified in the test_cnn_config.json file.
The trained models will be saved in the ~/cnn_output directory with the test_id.
Usage:
    1. Edit the test_cnn_config.json file with the desired parameters.
    2. Run the test_models.sh script, which calls this file.
    3. The test results will be saved in the ~/cnn_output directory with the test_id.
Configuration:
The configuration file (test_cnn_config.json) should contain the following keys:
    - test_id: A unique identifier for the test. If empty, a timestamp will be used.
    - spreading_factor: Spreading factor of LoRa modulation.
    - snr_values: List of SNR values to test against (used if mixed_test is True).
    - rate: List of arrival rate values for interfering users to test against (used if mixed_test is True).
    - test_dir: Directory containing the test data.
    - output_dir: Directory to save the test results.
    - img_size: Size of the images used for testing.
    - model: The base model to use for testing.
    - trained_model_folder: Folder containing the trained models.
    - trained_model: A specific model or list of models to test. If empty, all models in the folder will be tested.
    - mixed_test:  indicating whether to test models across all SNR and rate combinations (True) or only on their own data (False).
Outputs:
The script generates the following outputs:
    - A log file containing the details of the test run.
    - A config.json file saved in the output directory with the test configuration.
    - CSV files containing the SERs (Symbol Error Rates) for each model tested.
"""
import json
import os
import time
import logging
import numpy as np
import pandas as pd
import sys
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))
from model_includes.cnn_test import test_model


if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config file and create output folders
    with open("cnn_bash/cnn_test_config.json") as f:
        init_config=json.load(f)
    resume=init_config["resume"] #if resume from previous run
    if resume:
        resume_dir=init_config["resume_dir"]
        with open(os.path.join(resume_dir, "config.json")) as file: 
            config=json.load(file)
        test_id="resume"+"_"+time.strftime("%Y%m%d-%H%M%S")
        output_dir=resume_dir
        data_dir = os.path.join(output_dir, "data")
    else:
        config=init_config
        if config["test_id"]!="":
            test_id = config["test_id"]+"_"+time.strftime("%Y%m%d-%H%M%S")
        else:
            test_id = time.strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join("cnn_output", test_id)
        data_dir = os.path.join(output_dir, "data")
        os.makedirs(output_dir, exist_ok=True)
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
    if not resume:
        logger.info("save config file")
        config["test_id"] = test_id
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f)
    
    # Models to be tested, if empty all models in the folder will be tested
    logger.info("Evaluating CNN models with test data...")
    trained_model_folder=config["trained_model_folder"]
    if config["trained_model"]!="":
        trained_models=[config["trained_model"]] if not isinstance(config["trained_model"], list) else config["trained_model"]
    else:
        trained_models=os.listdir(trained_model_folder)

    #initiate progress txt
    
    if resume and os.path.exists(os.path.join(output_dir, "progress.txt")):
        tested_models=open(os.path.join(output_dir, "progress.txt"), "r")
        progress_txt=open(os.path.join(output_dir, "progress.txt"), "a")
        tested_models=tested_models.readlines()
        data_txt=open(os.path.join(output_dir, "data.txt"), "a")
    else:
        progress_txt=open(os.path.join(output_dir, "progress.txt"), "x")
        data_txt=open(os.path.join(output_dir, "data.txt"), "x")
    

    
    # Test models only on trained SNR 
    if config["mixed_test"]==False: 
        ser_count=[]
        rate_count=[]
        
        # Count the instances of of different snr and rate to make list for model_SERs
        for Tmodel in trained_models: 
            ser_count.append(int(Tmodel.split("_")[2]))
            rate_count.append(float(Tmodel.split("_")[4].replace(".pth", "")))
        unique_snr=[]
        unique_rate=[]
        [unique_snr.append(i) for i in ser_count if i not in unique_snr]
        [unique_rate.append(i) for i in rate_count if i not in unique_rate]
        unique_snr=np.sort(unique_snr)
        unique_rate= np.sort(unique_rate)
        model_SERs=[[None] *len(unique_rate) for _ in range(len(unique_snr))]

        # Test models on their given snr and rate
        for Tmodel in range(len(trained_models)):
            snr=[int(trained_models[Tmodel].split("_")[2])] #find snr value from name
            rate=[float(trained_models[Tmodel].split("_")[4].replace(".pth", ""))] #find rate value from name
            if resume:
                if f"{snr}_{rate}\n" in tested_models:
                    continue
            print(snr, rate)
            SERs=test_model(logger=logger,
                                  test_dir=config["test_dir"],
                                  img_size=config["img_size"],
                                  trained_model=os.path.join(trained_model_folder, trained_models[Tmodel]),
                                  snr_list=snr,
                                  rates=rate,
                                  base_model=config["model"],
                                  M=2**config["spreading_factor"],
                                  device=device)

            # Sort the SERs into the model_SERs list
            data_txt.write(f"{snr}_{rate}_SER_{SERs[0][0]}\n")
            progress_txt.write(f"{snr}_{rate}\n")
            model_SERs[np.where(unique_snr==snr[0])[0][0]][np.where(unique_rate==np.float64(rate[0]))[0][0]]=SERs[0][0]
            print(f"Tested model snr:{snr} rate:{rate} Result SER:{SERs[0][0]}")
            logger.info(f"Tested model snr:{snr} rate:{rate} Result SER:{SERs[0][0]}")
        print(model_SERs)
        progress_txt.write("done\n")
        pd.DataFrame(model_SERs, columns=unique_snr, index=unique_rate).to_csv(os.path.join(data_dir, f"{test_id}.csv"))

    elif config["mixed_test"]==True: 
        for Tmodel in trained_models:
            if resume:
                if Tmodel+"\n" in tested_models:
                    continue
            SERs=test_model(logger=logger, 
                    test_dir=config["test_dir"],
                    img_size=config["img_size"],
                    trained_model=os.path.join(trained_model_folder, Tmodel),
                    snr_list=config["snr_values"],
                    rates=config["rate"],
                    base_model=config["model"],
                    M=2**config["spreading_factor"])
            print(SERs)
            frame=pd.DataFrame(SERs, columns=config["rate"], index=config["snr_values"])
            print(frame)
            frame.to_csv(os.path.join(data_dir, f"mixed_test_{Tmodel.replace('.pth', '', -1)}.csv"))
            progress_txt.write(Tmodel+"\n")
            logger.info(f'saved mixed_test_{Tmodel.replace(".pth", "", -1)}.csv')           
    else:
        logger.error("No test option chosen: mixtest")

    logger.info("Testing complete.")
    
    

        
                        
    
    





