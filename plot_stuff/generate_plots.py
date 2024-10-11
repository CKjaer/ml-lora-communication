import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import logging
import time

def generate_plots(data, logger, spreading_factor: int, num_samples: int, directory: str, max_vals: dict = None):
    
    sample_idx = 0
    start_time = time.time()
    num_symbols = 2**spreading_factor
    plt.switch_backend('agg')
    
    for i in tqdm(range(len(data)), desc="Generating plots"):
        # create a lin space for the frequency values
        freqs_idx = np.arange(0, num_symbols, 1)
        
        fig = plt.figure(figsize=(1,1), dpi=num_symbols)
        ax = fig.add_subplot(111)
        
        # find the upper limit for current snr condition
        upper_y_lim = max_vals[data['snr'][i]]
        ax.set_ylim(0, upper_y_lim)
        
        # make the plot binary
        plt.axis('off')
        fig.set_facecolor('black')
        plt.plot(freqs_idx, data['freqs'][i], color = 'white', linewidth=0.5)
        plt.close(fig)
        
        # find the index of the current sample
        sample_idx = i % num_samples
        
        plots_dir = os.path.join(directory, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        #save images to folder
        try:
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fig.savefig(os.path.join(plots_dir, f"snr_{data['snr'][i]}_symbol_{data['symbol'][i]}_{sample_idx}.png"), dpi=num_symbols)
        except Exception as e:
            logger.error(f"Error generating plot for sample {sample_idx} in file snr_{data['snr'][i]}_symbol_{data['symbol'][i]}. Error: {e}")
        
        if (i + 1) % 5000 == 0:
            logger.info(f"Generated {i + 1} plots in {time.time() - start_time:.4f} seconds") 

    logger.info(f"Finished generating {len(data)} plots in {time.time() - start_time:.4f} seconds")

def find_max(df, logger):
    snr_values = df['snr'].unique()

    # store the maximum value for each snr in a dictionary
    max_vals = {}
    for snr in snr_values:
        # create subset of data for current snr
        snr_df = df[df['snr'] == snr]
        
        # find maximum frequency in every sample and then find the maximum of those
        max_value = snr_df['freqs'].apply(lambda x: np.max(x)).max()
        
        # find the maximum within each sample, and then find the row with the highest value
        max_row_idx = snr_df['freqs'].apply(lambda x: np.argmax(x)).idxmax()
        
        # find index within the sample with the highest value
        max_col_idx = np.argmax(snr_df['freqs'][max_row_idx])
        
        max_vals[snr] = max_value
        logger.info(f"Maximum value: {max_value} found at row {max_row_idx}, column {max_col_idx}, in symbol {df['symbol'][max_row_idx]}, in snr {df['snr'][max_row_idx]}")

    # logger.info(f"Maximum value: {max_value} found at row {max_row_idx}, column {max_col_idx}, in symbol {df['symbol'][max_row_idx]}, snr {df['snr'][max_row_idx]}")
    return max_vals


if __name__ == "__main__":
    from load_files import load_data
    logfilename = os.path.join("plot_stuff","generate_plots.log")
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=logfilename, encoding='utf-8', level=logging.INFO)
    logger.info("Starting the program")
    
    data = load_data("plot_stuff/test_data_fft", logger=logger) # change directory when running test
    max_vals = find_max(data, logger=logger)
    # generate_plots(data, logger=logger, spreading_factor=7, num_samples=1000, directory="/home/clyholm/ml-lora-communication/plot_stuff", max_vals=max_vals) # change directory when running test
    
    