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
from model_includes.LoadAndEval import loadAndevalModel


if __name__=="__main__":
    """
    Tests the trained models from the trained model folder given in test_ML_config.json
    mix_test=True:  tests the trained models on all snr and rate combinations given by the config
    output is:              rates
                    SNRs    [][][]
                            [][][]
    mix_test=False: tests the trained models on the the given snr and rates (ignores snr and rates from config)
    output is:        model1.csv|        rates  |     model2.csv|       rates  |
                                |SNRs    [][][] |               |SNRs   [][][] |
                                |        [][][] |               |       [][][] |
    trained_model:  if left empty it will test all models in the trained_model_folder
                    if not empty it will only train on the given model(s) can be set in a list for multiple models - only needs <name>.pth
    """
    #generate output folders
    with open("cnn_bash/ML_config_test.json") as f: #fix so dont have to be in root?
        config=json.load(f)
    if config["test_id"]!="":
        test_id = config["test_id"]+"_"+time.strftime("%Y%m%d-%H%M%S")
    else:
        test_id = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join("cnn_output", test_id)
    data_dir = os.path.join(output_dir, "data")
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
    
    logger.info("save config file")
    config["test_id"] = test_id
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f)

    logger.info("starting model evalulation")

    
    #specify the models to tested, if nothing it will test all files in the folder
    trained_model_folder=config["trained_model_folder"]
    if config["trained_model"]!="":
        trained_models=[config["trained_model"]] if not isinstance(config["trained_model"], list) else config["trained_model"]

    else:
        trained_models=os.listdir(trained_model_folder)

    if config["mixed_test"]==False: #state whether the models should be tested across all SNR or only the one it is trained on

        """
        SNR and Rate from config are not needed for eval models on only their own data
        model_SERs gives SERs for models tested on the intended dataset (so one model for each SER)
                rates
        SNRs    [][][]
                [][][]
        """
        #count the instances of of different snr and rate to make list for Model_Sers
        ser_count=[]
        rate_count=[]
        for Tmodel in trained_models: 
            ser_count.append(int(Tmodel.split("_")[2]))
            rate_count.append(float(Tmodel.split("_")[4].replace(".pth", "")))
        uniqe_snr=[]
        uniqe_rate=[]
        [uniqe_snr.append(i) for i in ser_count if i not in uniqe_snr]
        [uniqe_rate.append(i) for i in rate_count if i not in uniqe_rate]
        uniqe_snr=np.sort(uniqe_snr)
        unique_rate= np.sort(uniqe_rate)
        model_SERs=[[None] *len(uniqe_rate) for _ in range(len(uniqe_snr))]

        #test models on their given snr and rate
        for Tmodel in range(len(trained_models)):
            snr=[int(trained_models[Tmodel].split("_")[2])] #find snr value from name
            rate=[float(trained_models[Tmodel].split("_")[4].replace(".pth", ""))] #find rate value from name
            SERs=loadAndevalModel(logger=logger,
                                  test_dir=config["test_dir"],
                                  img_size=config["img_size"],
                                  trained_model=os.path.join(trained_model_folder, trained_models[Tmodel]),
                                  snr_list=snr,
                                  rates=rate,
                                  base_model=config["model"],
                                  M=2**config["spreading_factor"])
            model_SERs[np.where(uniqe_snr==snr[0])[0][0]][np.where(uniqe_rate==np.float64(rate[0]))[0][0]]=SERs[0][0] # this line if fucked but it works

        print(model_SERs)
        pd.DataFrame(model_SERs, columns=uniqe_snr, index=uniqe_rate).to_csv(os.path.join(data_dir, f"{test_id}.csv"))#save to csv

    elif config["mixed_test"]==True: 
        """
        test each trained model on all snr and rate values, rate and snr values are needed from config
        saves csv file in the form of

        model1.csv|        rates  |     model2.csv|       rates  |
                  |SNRs    [][][] |               |SNRs   [][][] |
                  |        [][][] |               |       [][][] |

        """
        for Tmodel in trained_models:
            SERs=loadAndevalModel(logger=logger, 
                    test_dir=config["test_dir"],
                    img_size=config["img_size"],
                    trained_model=os.path.join(trained_model_folder, Tmodel),
                    snr_list=config["snr_values"],
                    rates=config["rate"],
                    base_model=config["model"],
                    M=2**config["spreading_factor"])
            pd.DataFrame(SERs, columns=config["snr_values"], index=config["rate"]).to_csv(os.path.join(data_dir, f"mixed_test_{Tmodel.replace(".pth", "", -1)}.csv"))
            logger.info(f'saved mixed_test_{Tmodel.replace(".pth", "", -1)}.csv')
            
    else:
        logger.error("No test option chosen: mixtest")
    

    logger.info("script done")
    
    

        
                        
    
    





