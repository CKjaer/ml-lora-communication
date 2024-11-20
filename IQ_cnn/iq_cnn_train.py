from iq_cnn import IQCNN, otherArticleIQCNN, train
from iq_dataset import IQDataset, CustomIQTransform
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import logging
import os
import time
import json
from torch.utils.tensorboard import SummaryWriter

# create output directory
output_dir = "iq_cnn_output"
os.makedirs(output_dir, exist_ok=True) # make general output folder if it doesn't exist
output_dir = os.path.join(output_dir, time.strftime("%Y%m%d-%H%M%S"))
os.makedirs(output_dir) # make a new folder for the current run

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))

# create a logger
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename=os.path.join(output_dir, "iq_cnn_train.log"), encoding='utf-8', level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if GPU is available otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
logger.info(f"Using device: {device}")

# create dataset
dataset = IQDataset("output/20241120-085757/csv", transform=CustomIQTransform(), logger=logger)

# define hyperparameters
M = 128
learning_rate = 0.01
batch_size = 10
epochs = 50
optimizer_choice = "Adam"
criterion_choice = "cross_entropy"

# define data parameters
# snr_list =[-8]
snr_list = [i for i in range(-12, -6, 2)] # NOTE: use -16 to -4 for full range #TODO: make this load from a config file
rate_list = [0.0, 0.25] # NOTE: use 0.0, 0.25, 0.5, 0.7, 1.0 for full range #TODO: make this load from a config file

# dictionary to store the symbol error rates
symbol_error_rates = {rate: {} for rate in rate_list}

for rate in rate_list:
    for snr in snr_list:
        logger.info(f"Calculating SER for snr: {snr}, rate {rate}")
        
        # create datasets and dataloaders
        dataset.subset_data(snr=snr, rate_param=rate) # filter the dataset based on snr and rate
        train_set, val_set = random_split(dataset, [int(0.8*len(dataset)), len(dataset) - int(0.8*len(dataset))]) # 80-20 split
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        
        # define the model
        #model = IQCNN(M).to(device)
        model = otherArticleIQCNN(M).to(device)
        
        # define the optimizer and criterion
        if optimizer_choice == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_choice == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
            
        if criterion_choice == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()
        else:
            logger.error("Only cross entropy loss is supported")
            raise ValueError("Only cross entropy loss is supported")
        
        # train the model
        ser = train(model, train_loader, val_loader, epochs, criterion, optimizer, device, logger, writer)
        
        # Save model and optimizer
        torch.save(model.state_dict(), os.path.join(output_dir, f'model_snr_{snr}_rate_{rate}.pth'))
        torch.save(optimizer.state_dict(), os.path.join(output_dir, f'optimizer_snr_{snr}_rate_{rate}.pth'))
        
        # store the symbol error rate
        symbol_error_rates[rate][snr] = ser
        
        # dump the symbol error rates to a JSON file
        with open(os.path.join(output_dir, "symbol_error_rates.json"), "w") as f:
            json.dump(symbol_error_rates, f)
        
logger.info("All SER values have been calculated.")
writer.close()

# def debugging_stuff():
#     # Checking data/label alignment in validation set
#         for batch_idx, (inputs, labels) in enumerate(val_loader):
#             print(f"Batch {batch_idx + 1} - Eval Data")
#             print(f"Inputs shape: {inputs.shape}")
#             print(f"First input: {inputs[0][0]}")
#             print(f"Labels: {labels[:10]}")
#             break
        
#         # check label distribution across train and val set
#         from collections import Counter
        
#         ################################################################
        
#         # Check label distribution
#         train_labels = [dataset[i][1].item() for i in train_set.indices]
#         val_labels = [dataset[i][1].item() for i in val_set.indices]

#         import matplotlib.pyplot as plt
#         plt.subplot(1, 2, 1)
#         plt.hist(train_labels, bins=128)
#         plt.title("Train Label Distribution")
#         plt.subplot(1, 2, 2)
#         plt.hist(val_labels, bins=128)
#         plt.title("Validation Label Distribution")
#         plt.show()