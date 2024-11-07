import sys
import json
import os
import time
import logging
from numpy import savetxt
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))
from ML_modelFunctions.ModelTrainAndEval import ModelTrainAndEval
from ML_modelFunctions.LoadAndEval import loadAndevalModel

if __name__=="__main__":
    #generate unique folder
    with open("cnn_bash/ML_config.json") as f: #fix so dont have to be in root?
        config=json.load(f)
    if config["test_id"]!="":
        test_id = config["test_id"]
    else:
        test_id = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join("cnn_output", test_id)
    model_dir = os.path.join(output_dir, "models")
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

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
    logger.info("starting model evalulation")

    trained_model_folder=config["trained_model_folder"]
    #if the config file in line trained_model_folder is not empty, the program will run the load model and eval
    if trained_model_folder!="": 
        logger.info("Loading and evalulating models")
        SERs, trained_models=loadAndevalModel(logger=logger, 
                         img_dir=config["img_dir"],
                         output_dir=output_dir,
                         trained_model_folder=os.path.join(trained_model_folder),
                         batch_size=config["batch_size"],
                         snr_list=config["snr_values"],
                         rates=config["rate"],
                         base_model=config["model"],
                         M=2**config["spreading_factor"],
                         seed=config["seed"])

    else:
        logger.info("Training and evalulating models")
        SERs, trained_models=ModelTrainAndEval(logger=logger,
                          img_dir=config["img_dir"],
                          output_folder=output_dir,
                          batch_size=config["batch_size"],
                          snr_list=config["snr_values"],
                          rates=config["rate"],
                          base_model=config["model"],
                          M=2**config["spreading_factor"],
                          optimizer_choice=config["optimizer"],
                          num_epochs=config["num_epochs"],
                          learning_rate=config["learning_rate"],
                          seed=config["seed"])
    
    logger.info("Starting save data to csv")
    rates=config["rate"]
    SNRs=config["snr_values"]
    symbol_error_rates={rate: {} for rate in rates}
    for trained_model in range(len(trained_models)):
        for rate in range(len(rates)):
            for snr in range(len(SNRs)):
                symbol_error_rates[rates[rate]][SNRs[snr]] = SERs[trained_model][rate][snr]
                

        for rate, values in symbol_error_rates.items():
            snr_values = sorted(values.keys()) 
            ser_values = [values[snr] for snr in snr_values] # loop through to not mix up the order
            
            if rate == 0:
                zero_snr_values = snr_values
                zero_ser_values = ser_values
            
            savetxt(os.path.join(data_dir,f'{trained_models[trained_model]}snr_vs_ser_rate_{rate}.csv'), np.array([snr_values, ser_values]).T, delimiter=';', fmt='%d;%.6f')
            logger.info(f'saved {trained_models[trained_model]}snr_vs_ser_rate_{rate}.csv')
    
    logger.info("save config file")
    config["test_id"] = test_id
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f)





