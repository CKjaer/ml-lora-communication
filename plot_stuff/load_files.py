import numpy as np
import os
import pandas as pd

def load_data(directory):
    '''   
    Load all CSV files in the given directory into a single pandas DataFrame.
    This function will only work with the naming convention snr_{snr_value}_symbol_{symbol_value}.csv
    '''
    
    data_list = []

    # Iterate through all files in the given directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            
            parts = filename.split('_')
            snr_value = float(parts[1])
            symbol_value = int(parts[3].split('.')[0])

            # Read the CSV file into a pandas DataFrame
            file_path = os.path.join(directory, filename)
            data = pd.read_csv(file_path, header=None)

            # create a new column for SNR and Symbol as labels
            data['snr'] = snr_value
            data['symbol'] = symbol_value


            data_list.append(data)

    # Concatenate all dataframes into one large dataframe
    combined_data = pd.concat(data_list, ignore_index=True)
    combined_data.rename(columns={0: "freqs"}, inplace=True)
    
    return combined_data

if __name__ == "__main__":
    ...
