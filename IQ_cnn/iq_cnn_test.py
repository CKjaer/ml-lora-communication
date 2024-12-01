from iq_cnn import IQCNN, RealValuedCNN, ComplexValuedCNN, evaluate_and_calculate_ser
from iq_dataset import IQDataset, CustomIQTransform
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import os
import time
import json

# load config
with open("cnn_bash/iq_test_config.json", "r") as f:
    config = json.load(f)

# create output directory
name = config['model_name']
output_dir = "iq_cnn_output"
os.makedirs(output_dir, exist_ok=True) # make general output folder if it doesn't exist
try:
    output_dir = os.path.join(output_dir, "test" + "_" + name)
    os.makedirs(output_dir) # make a new folder for the current run
except FileExistsError:
    output_dir = "iq_cnn_output" # reset the output directory
    output_dir = os.path.join(output_dir, "test" + "_" + name + "_" + time.strftime("%Y%m%d-%H%M%S")) # if one folder already exists, create a new one with timestamp
    os.makedirs(output_dir)


# dump the config file to the output directory
with open(os.path.join(output_dir, "config.json"), "w") as f:
    json.dump(config, f)
    
# create a logger
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename=os.path.join(output_dir, "iq_cnn_test.log"), encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if GPU is available otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
logger.info(f"Using device: {device}")

# create dataset
dataset = IQDataset(config['input_dir'], transform=CustomIQTransform(), logger=logger)

# define hyperparameters
M = 2**config['spreading_factor']
batch_size = config['batch_size']

# define data parameters
snr_list = config['snr_values']
rate_list = config['rate_values']

# define model architecture
model_choice = config['model_name']
logger.info(f"Using model: {model_choice}")

# dictionary to store the symbol error rates
symbol_error_rates = {rate: {} for rate in rate_list}

for rate in rate_list:
    for snr in snr_list:
        start_time = time.time()
        logger.info(f"Calculating SER for snr: {snr}, rate {rate}")
        
        # subset the data
        dataset.subset_data(snr=snr, rate_param=rate)
        try:
            test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=64)
        except ValueError:
            logger.error(f"No samples found for snr: {snr}, rate: {rate}. Skipping...")
            continue
        
        # define the model
        match model_choice:
            case "IQCNN":
                model = IQCNN(M).to(device) # create the model
                model_dir = "iq_cnn_output/IQCNN/models" # get path to model state dicts
            case "RealValuedCNN":
                model = RealValuedCNN(M).to(device)
                model_dir = "iq_cnn_output/RealValuedCNN/models"
            case "ComplexValuedCNN":
                model = ComplexValuedCNN(M).to(device)
                model_dir = "iq_cnn_output/ComplexValuedCNN/models"
            case _: # default case
                logger.error(f"Model {model_choice} not recognized.")
                raise ValueError(f"Model {model_choice} not recognized.")
                
        
        # load the model and setup criterion
        try:
            model.load_state_dict(torch.load(os.path.join(model_dir, f"model_snr_{snr}_rate_{rate}.pth")))
        except FileNotFoundError:
            logger.error(f"model_snr_{snr}_rate_{rate}.pth not found in path: {model_dir}")
            break # stop the program
        criterion = nn.CrossEntropyLoss()
        
        # evaluate the model
        ser = evaluate_and_calculate_ser(model, test_loader, criterion, device, logger, start_time=start_time)
        
        # store the symbol error rate
        symbol_error_rates[rate][snr] = ser
        
        # dump the symbol error rates to a JSON file
        with open(os.path.join(output_dir, "symbol_error_rates.json"), "w") as f:
            json.dump(symbol_error_rates, f)