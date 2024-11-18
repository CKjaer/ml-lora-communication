import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import logging
from load_iq_data import load_data

class IQDataset(Dataset):
    def __init__(self, data_dir, transform=None, logger: logging.Logger = None):
        self.data = load_data(data_dir, logger)
        self.data_subset = None # subset of data based on snr and rate_param
        self.transform = transform
        self.snr = None
        self.rate_param = None
    
    def subset_data(self, snr, rate_param):
        self.snr = snr
        self.rate_param = rate_param
        if self.snr is not None and self.rate_param is not None:
            self.data_subset = self.data.loc[(self.data['snr'] == self.snr) & (self.data['rate'] == self.rate_param)] # filter to load only current snr and rate
        
        self.data_subset.reset_index(drop=True, inplace=True)
        self.labels = self.data_subset['symbol'].values
    
    def __len__(self):
        return len(self.data_subset)
    
    def __getitem__(self, idx):
        data_subset = self.data_subset.loc[idx, 'iq_data']
        label = torch.tensor(self.labels[idx])
        
        if self.transform:
            data_subset = self.transform(data_subset) # apply transform to data (to tensor and normalize)
        
        return data_subset, label
        
class CustomIQTransform:
    def __call__(self, data):
        data = torch.tensor([complex(value) for value in data], dtype=torch.cfloat)
        #data = (data - data.mean()) / data.std() # normalize
        
        # split the complex data into real and imaginary parts (remove this if you want to use complex numbers)
        real = torch.tensor([value.real for value in data], dtype=torch.float) # [1, M]
        imag = torch.tensor([value.imag for value in data], dtype=torch.float) # [1, M]
        data = torch.stack([real, imag], dim=0) # stack to [2, M] tensor (real and imaginary parts)
        
        return data
    
if __name__ == "__main__":
    # create a logger
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename="load_data.log", encoding='utf-8', level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    dataset = IQDataset("output/20241114-115337/csv", transform=CustomIQTransform(), logger=logger)
    snrs = [-10, -8, -6]
    for snr in snrs:
        dataset.subset_data(snr=snr, rate_param=0.0)
        test, label = dataset.__getitem__(0)
        print(test[0][0], test[1][0], label)