import os
import json
import uuid
from plot_stuff.generate_plots import generate_plots
from plot_stuff.load_files import load_data
from training_sims.data_generator.generate_training_data import create_data_csvs
import training_sims.data_generator.lora_phy #Must be imported for create_data_csvs to work
import logging

if __name__ == "__main__":
    # generate a unique
    with open('config.json') as f:
        config = json.load(f)
    test_id = str(uuid.uuid4())
    os.makedirs(os.path.join("output", test_id), exist_ok=True)
    csv_dir = os.path.join("output", test_id, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    # setup logging
    logfilename = "test_log.log"
    log_path = os.path.join("output",test_id,logfilename)
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=log_path, encoding='utf-8', level=logging.DEBUG)
    logger.info("Starting the program")

    # generate data
    create_data_csvs(logger, config["number_of_samples"], config["snr_values"], config["spreading_factor"], csv_dir, config["lambda"])
    
    # plot data
    plot_data = load_data(csv_dir, logger)
    # generate plots
    generate_plots(plot_data, logger, config["spreading_factor"], config["number_of_samples"], os.path.join("output", test_id))
    
    # save config
    config["test_id"] = test_id
    with open(os.path.join("output", test_id, "config.json"), 'w') as f:
        json.dump(config, f)