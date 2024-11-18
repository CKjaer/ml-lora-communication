import numpy as np
import os
import pandas as pd
import logging
import time
from tqdm import tqdm

# Load all CSV files in the given directory into a single pandas DataFrame
def load_data(directory, logger: logging.Logger, header="iq"):
    data_list = []
    start_time = time.time()

    # Iterate through all files in the given directory
    try:
        for filename in tqdm(os.listdir(directory), desc="Loading data"):
            if filename.endswith(".csv") and filename.startswith(header):
                try:
                    parts = filename.split('_')
                    snr_value = float(parts[1])
                    symbol_value = int(parts[3])
                    rate = float(parts[5].removesuffix(".csv"))
                except (ValueError, IndexError):
                    logger.error(f"Invalid filename {filename}. Expected format: snr_{snr_value}_symbol_{symbol_value}_rate{rate}.csv")
                    continue

                # Read the CSV file into a pandas DataFrame
                file_path = os.path.join(directory, filename)
                data = pd.read_csv(file_path, header=None)
                data.rename(columns={0: "iq_data"}, inplace=True)

                # Split and convert all complex values in one step
                try:
                    data['iq_data'] = data['iq_data'].str.split(';').apply(
                        lambda row: [complex(x.strip()) for x in row]
                    )
                except ValueError as e:
                    logger.error(f"Error processing file {filename}: {e}")
                    continue


                # # DEBUGGING
                # # Check for rows with 0 imaginary part
                # zero_imag_rows = data[data['iq_data'].apply(lambda x: all(abs(y.imag) < 1e-10 for y in x))].index
                # non_zero_imag_rows = data[~(data['iq_data'].apply(lambda x: all(abs(y.imag) < 1e-10 for y in x)))].index
                # if len(zero_imag_rows) > 0:
                #     logger.warning(f"File {filename} contains {len(zero_imag_rows)} rows with 0 imaginary part")
                # logger.info(f"File {filename} contains {len(non_zero_imag_rows)} rows with non-zero imaginary part")

                # Add SNR, symbol, and rate as columns
                data['rate'] = rate
                data['snr'] = snr_value
                data['symbol'] = symbol_value
                data_list.append(data)
    except FileNotFoundError:
        logger.error(f"Directory {directory} not found. Check the path and try again.")
        exit()

    # Concatenate all dataframes into one large dataframe
    combined_data = pd.concat(data_list, ignore_index=True)

    # Sort the data by SNR and Symbol
    combined_data.sort_values(by=['rate', 'snr', 'symbol'], ascending=True, inplace=True)
    combined_data = combined_data.reset_index(drop=True)

    logger.info(f"Loaded {len(combined_data)} samples from {len(data_list)} files in {time.time() - start_time:.4f} seconds")
    return combined_data

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename="load_data.log", encoding='utf-8', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting the program")
    dataset = load_data("output/20241114-115337/csv", logger)
    print("finished loading data")
    # save the dataset to a CSV file
    #dataset.to_csv("output/20241114-115337/combined_data.csv", index=False)