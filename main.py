import os
import json
from plot_stuff.generate_plots import generate_plots
from plot_stuff.load_files import load_data, find_max
from training_sims.data_generator.generate_training_data import create_data_csvs
import training_sims.data_generator.lora_phy #Must be imported for create_data_csvs to work
import logging
import time

if __name__ == "__main__":
    # generate a unique
    with open('config.json') as f:
        config = json.load(f)
    test_id = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join("output")
    os.makedirs(os.path.join(output_dir, test_id), exist_ok=True)
    csv_dir = os.path.join(output_dir, test_id, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    # setup logging
    logfilename = "test_log.log"
    log_path = os.path.join(output_dir,test_id,logfilename)
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=log_path, encoding='utf-8', level=logging.INFO)
    logger.info("Starting the program")

    ################# data simulation #################
    try:
        create_data_csvs(logger, config["number_of_samples"], config["snr_values"], config["spreading_factor"], csv_dir, config["lambda"])
    except Exception as e:
        logger.error(f"Error creating data: {e}")
        print(f"Error creating data: {e}")
    
    
    ################# plot data #################
    try:
        plot_data = load_data(csv_dir, logger)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        print(f"Error loading data: {e}")
    
    # Debugging to check for max value in FFT magnitude
    try:
        find_max(plot_data, logger)
        generate_plots(plot_data, logger, config["spreading_factor"], config["number_of_samples"], os.path.join(output_dir, test_id))
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
        print(f"Error generating plots: {e}")
    
    
    ################# train model #################
    # function calls to train the model go here
    
    # save config
    config["test_id"] = test_id
    with open(os.path.join(output_dir, test_id, "config.json"), 'w') as f:
        json.dump(config, f)
