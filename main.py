import os
import json
from plot_stuff.generate_plots import generate_plots, find_max
from plot_stuff.load_files import load_data
from training_sims.data_generator.generate_training_data import create_data_csvs
import training_sims.data_generator.lora_phy #Must be imported for create_data_csvs to work
import logging
import time

if __name__ == "__main__":
    # Generate a unique
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
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',datefmt='%Y-%m-%d %H:%M:%S', filename=log_path, encoding='utf-8', level=logging.INFO)
    logger.info("Starting the program")

    N_samples = config["number_of_samples"]
    snr_values = config["snr_values"]
    if len(N_samples) == 1:
            N_samp_array = N_samples * len(snr_values)
    elif len(N_samples) != len(snr_values):
        print(f"BAD DIMENSION: {len(N_samples)}!")
        raise TypeError(f"number_of_samples has invalid dimensions: Must be either 1, or the same as snr_values ({len(snr_values)})")
    else:
        print(f"GOOD SIZE: {len(N_samples)}!")
        N_samp_array = N_samples

    ################# data simulation #################
    try:
        create_data_csvs(logger, N_samp_array, config["snr_values"], config["spreading_factor"], csv_dir, config["rate"], config["SIR_random"], config["SIR_span_and_steps"])
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
        max_vals = find_max(plot_data, logger)
        generate_plots(data = plot_data, logger = logger, spreading_factor=config["spreading_factor"], num_samples=N_samp_array, directory = os.path.join(output_dir, test_id), max_vals=max_vals, line_plot=config["line_plot"])
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
        print(f"Error generating plots: {e}")
    
    ################# train model #################
    # function calls to train the model go here
    
    # save config
    config["test_id"] = test_id
    with open(os.path.join(output_dir, test_id, "config.json"), 'w') as f:
        json.dump(config, f)
