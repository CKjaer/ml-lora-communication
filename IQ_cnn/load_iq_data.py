import os
import pandas as pd
import logging
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def process_file(file_info):
    """This file processes a single CSV file containing IQ data.

    Args:
        file_info (list): The file_info list contains the filename, directory, and header. It is generated by the load_data function.

    Returns:
        (data, error): A tuple containing the data and an error message. If the data is None, the error message will contain the reason.
    """
    filename, directory, header = file_info
    if filename.endswith(".csv") and filename.startswith(header):
        try:
            parts = filename.split('_')
            snr_value = float(parts[1])
            symbol_value = int(parts[3])
            rate = float(parts[5].removesuffix(".csv"))
        except (ValueError, IndexError):
            # If the filename is not in the expected format, return None data and an error message (data, error)
            return None, f"Invalid filename {filename}. Expected format: snr_{snr_value}_symbol_{symbol_value}_rate{rate}.csv"

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
            # returns (data, error)
            return None, f"Error processing file {filename}: {e}"

        # Add SNR, symbol, and rate as columns
        data['rate'] = rate
        data['snr'] = snr_value
        data['symbol'] = symbol_value
        return data, None
    return None, None

def load_data(directory, logger: logging.Logger, header="iq"):
    data_list = []
    start_time = time.time()

    # Create a list of file information tuples
    file_info_list = [(filename, directory, header) for filename in os.listdir(directory)]

    # Use ProcessPoolExecutor to read files in parallel
    # store all data and error messages in a list
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, file_info_list), total=len(file_info_list), desc="Loading data"))

    # Process the results
    for data, error in results:
        if data is not None:
            data_list.append(data)
        if error is not None:
            logger.error(error) # log all error messages

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
    dataset = load_data("output/autoscaling_training_set_20241121-134229/csv", logger)
    print("finished loading data")