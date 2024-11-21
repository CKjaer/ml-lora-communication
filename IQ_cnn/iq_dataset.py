import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import logging
from load_iq_data import load_data

class IQDataset(Dataset):
    """
    A custom Dataset class for loading and processing IQ data.

    Attributes:
        data (pd.DataFrame): The loaded data from CSV files.
        data_subset (pd.DataFrame): A subset of the data filtered by SNR and rate parameters.
        transform (callable, optional): A function/transform to apply to the data.
        snr (float, optional): The signal-to-noise ratio for filtering the data.
        rate_param (float, optional): The rate parameter for filtering the data.
        labels (np.ndarray): The labels corresponding to the data subset.
    """
    def __init__(self, data_dir: str, transform=None, logger: logging.Logger = None):
        """
        Initializes the IQDataset with the data directory, transform, and logger.

        Args:
            data_dir (str): The directory containing the CSV files with IQ data.
            transform (callable, optional): A function/transform to apply to the data.
            logger (logging.Logger, optional): A logger for logging information.
        """
        self.data = load_data(data_dir, logger)
        self.data_subset = None  # Subset of data based on SNR and rate_param
        self.transform = transform
    
    def subset_data(self, snr=None, rate_param=None):
        """
        Filters the data based on the given SNR and rate parameters.\n
        If no parameters are given, the entire dataset is loaded.

        Args:
            snr (float, optional): The signal-to-noise ratio for filtering the data.
            rate_param (float, optional): The rate parameter for filtering the data.
        """
        self.snr = snr
        self.rate_param = rate_param
        if self.snr is not None and self.rate_param is not None:
            self.data_subset = self.data.loc[(self.data['snr'] == self.snr) & (self.data['rate'] == self.rate_param)]  # Filter to load only current SNR and rate
        else:
            self.data_subset = self.data  # Load all data (useful for testing)
        
        self.data_subset.reset_index(drop=True, inplace=True)
        self.labels = self.data_subset['symbol'].values
    
    def __len__(self):
        """
        Returns the length of the data subset.

        Returns:
            int: The number of samples in the data subset.
        """
        return len(self.data_subset)
    
    def __getitem__(self, idx):
        """
        Retrieves the data and label at the given index.\n
        Note that this grabs items from the data subset.

        Args:
            idx (int): The index of the data to retrieve.

        Returns:
            tuple: A tuple containing the data and the corresponding label.
        """
        data_subset = self.data_subset.loc[idx, 'iq_data']
        label = torch.tensor(self.labels[idx])
        
        if self.transform:
            data_subset = self.transform(data_subset)  # Apply transform to data (to tensor and normalize)
        
        return data_subset, label
        
class CustomIQTransform:
    """
    A custom transform class for processing IQ data.
    """
    def __call__(self, data):
        """
        Applies the transform to the given data.

        Args:
            data (list): A list of complex IQ data values.

        Returns:
            torch.Tensor: A tensor containing the real and imaginary parts of the IQ data. [2, M]
        """
        data = torch.tensor([complex(value) for value in data], dtype=torch.cfloat)
        data = (data - data.mean(dim=0)) / data.std(dim=0)  # Normalize
        
        # Split the complex data into real and imaginary parts
        real = torch.tensor([value.real for value in data], dtype=torch.float)  # [1, M]
        imag = torch.tensor([value.imag for value in data], dtype=torch.float)  # [1, M]
        data = torch.stack([real, imag], dim=0)  # Stack to [2, M] tensor (real and imaginary parts)
        
        return data
    
if __name__ == "__main__":
    # Create a logger
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename="load_data.log", encoding='utf-8', level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    dataset = IQDataset("output/20241114-115337/csv", transform=CustomIQTransform(), logger=logger)
    snrs = [-10, -8, -6]
    for snr in snrs:
        dataset.subset_data(snr=snr, rate_param=0.0)
        test, label = dataset.__getitem__(0)
        print(test[0][0], test[1][0], label)