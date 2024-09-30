import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import logging
import time

def generate_plots(data, logger, spreading_factor: int, num_samples: int, directory: str):
    
    sample_idx = 0
    start_time = time.time()
    num_symbols = 2**spreading_factor
    plt.switch_backend('agg')
    
    for i in tqdm(range(len(data)), desc="Generating plots"):
        freqs_idx = np.arange(0, num_symbols, 1)
        
        fig = plt.figure(figsize=(1,1), dpi=num_symbols)
        ax = fig.add_subplot(111)
        ax.set_ylim(0, 130)
        plt.axis('off')
        fig.set_facecolor('black')
        plt.plot(freqs_idx, data['freqs'][i], color = 'white', linewidth=0.5)
        plt.close(fig)
        
        sample_idx += 1
        if sample_idx > num_samples:
            sample_idx = 1
        
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


if __name__ == "__main__":
    from load_files import load_data
    logfilename = "test_log.log"
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=logfilename, encoding='utf-8', level=logging.DEBUG)
    logger.info("Starting the program")
    
    data = load_data("/home/clyholm/ml-lora-communication/test_data_for_plots", logger=logger) # change directory when running test
    generate_plots(data, logger=logger, spreading_factor=7, num_samples=1000, directory="/home/clyholm/ml-lora-communication/plot_stuff") # change directory when running test
    
    