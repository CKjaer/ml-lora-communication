import numpy as np
import os
import pandas as pd
import logging
import time
from tqdm import tqdm

def load_data(directory, logger):
    '''   
    Load all CSV files in the given directory into a single pandas DataFrame.
    This function will only work with the naming convention snr_{snr_value}_symbol_{symbol_value}.csv
    '''
    
    data_list = []
    logger.name = "load_data"
    start_time = time.time()
    # Iterate through all files in the given directory
    try:
        for filename in tqdm(os.listdir(directory), desc="Loading data"):
            if filename.endswith(".csv"):
                try:
                    parts = filename.split('_')
                    snr_value = float(parts[1])
                    symbol_value = int(parts[3].split('.')[0])
                except ValueError:
                    logger.error(f"Invalid filename {filename}. Expected format: snr_{snr_value}_symbol_{symbol_value}.csv")
                    continue

                # Read the CSV file into a pandas DataFrame
                file_path = os.path.join(directory, filename)
                data = pd.read_csv(file_path, header=None)

                # create a new column for SNR and Symbol as labels
                data['snr'] = snr_value
                data['symbol'] = symbol_value
                data_list.append(data)
    except FileNotFoundError:
        logger.error(f"Directory {directory} not found. Check the path and try again.")
        exit()

    # Concatenate all dataframes into one large dataframe
    combined_data = pd.concat(data_list, ignore_index=True)
    combined_data.rename(columns={0: "freqs"}, inplace=True)
    print("HERE")
    combined_data['freqs'] = combined_data['freqs'].apply(lambda x: list(map(float, x.split(';'))))
    print("Done")
    
    # Sort the data by SNR and Symbol
    combined_data.sort_values(by=['snr','symbol'], ascending=True, inplace=True)

    # Reset index after sorting
    combined_data = combined_data.reset_index(drop=True)

    logger.info(f"Loaded {len(combined_data)} samples from {len(data_list)} files in {time.time() - start_time:.4f} seconds")
    return combined_data

# def split_freqs_column(df):
#     # Initialize an empty list to store the split frequency lists
#     split_freqs = []

#     # Iterate over the 'freqs' column and split the strings
#     for freqs in df['freqs']:
#         split_freqs.append(list(map(float, freqs.split(';'))))

#     # Convert the list of lists to a numpy array for better memory management
#     df['freqs'] = np.array(split_freqs, dtype=object)
#     return df

def find_max(df, logger):
    flattened_freqs = [(value, row_idx, col_idx) 
                   for row_idx, sublist in enumerate(df['freqs']) 
                   for col_idx, value in enumerate(sublist)]

    # Find the maximum value and its location
    max_value, max_row_idx, max_col_idx = max(flattened_freqs, key=lambda x: x[0])


    logger.info(f"Maximum value: {max_value} found at row {max_row_idx}, column {max_col_idx}, in symbol {df['symbol'][max_row_idx]}, snr {df['snr'][max_row_idx]}")

if __name__ == "__main__":
    ...
