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
    with open("cnn_bash/ML_config.json") as f: #fix so dont have to be in root
        config=json.load(f)
    if config["test_id"]!="":
        test_id = config["test_id"]
    else:
        test_id = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join("ML_output", test_id)
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

    trained_model=config["trained_model"]
    if trained_model!="": #make it so that it can loop through a list of trained models
        logger.info("Loading and evalulating models")
        SERs=loadAndevalModel(logger=logger, 
                         img_dir=config["img_dir"],
                         output_dir=output_dir,
                         trained_model=trained_model,
                         batch_size=config["batch_size"],
                         snr_list=config["snr_values"],
                         rates=config["rate"],
                         base_model=config["model"],
                         M=2**config["spreading_factor"],
                         seed=config["seed"])
    else:
        logger.info("Training and evalulating models")
        SERs=ModelTrainAndEval(logger=logger,
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
    
    symbol_error_rates={}
    for rate in config["rate"]:
        symbol_error_rates[rate]=[]

    rates=config["rate"]
    SNRs=config["snr_values"]
    for rate in range(len(rates)):
        for snr in range(len(SNRs)):
            symbol_error_rates[rates[rate]].append((SERs[rate][snr], SNRs[snr]))
    print(symbol_error_rates)

for rate, values in symbol_error_rates.items():
    snr_values = list(map(int, values.keys()))
    ser_values = list(values.values())
    snr_values = sorted(snr_values)
    
    if rate == 0:
        zero_snr_values = snr_values
        zero_ser_values = ser_values
    
    savetxt(os.path.join(data_dir,f'snr_vs_ser_rate_{rate}.csv'), np.array([snr_values, ser_values]).T, delimiter=';', fmt='%d;%.6f')
    # os.makedirs(os.path.join(output_dir, test_id), exist_ok=True)





