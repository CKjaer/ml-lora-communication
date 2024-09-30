import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from tqdm import tqdm
import logging
import time

def generate_plots(data, logger, spreading_factor: int, num_samples: int, directory: str):
    """This function takes a list of data of form ['freqs', 'snr', 'symbol'] and generated binary FFT modulus plots.
    
        Use load_data from load_data.py to get data from csv to dataframe of this form

    Args:
        data (_type_): list containing absolute values of the generated FFT.\n
        spreading_factor (int): spreading factor is used to determine the number of symbols, and the size of the plots.\n
        num_samples (int): number of samples used for naming the plots.\n
        directory (str): Directory to where the plots folder should be created.
    """    
    sample_idx = 0
    num_symbols = 2**spreading_factor
    start_time = time.time()
    plt.switch_backend('agg')
    
    for i in tqdm(range(len(data)), desc="Generating plots"):
        freqs = list(map(float, data["freqs"].iloc[i].split(';'))) # split string and convert to float
        freqs_idx = np.arange(0, len(freqs), 1)
        #print(data["symbol"].iloc[i])

        fig = plt.figure(figsize=(1,1), dpi=num_symbols)
        ax = fig.add_subplot(111)
        plt.axis('off')
        fig.set_facecolor('black')
        ax.set_ylim(0, 130)
        plt.plot(freqs_idx, freqs, color = 'white', linewidth=0.5)
        plt.close(fig)
        
        # this counter is used to create a unique name for each plot
        sample_idx += 1
        if sample_idx == num_samples: # hard coded as we have 1000 representations
            sample_idx = 1
        
        # create plots folder in given directory
        plots_dir = os.path.join(directory, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        #save images to folder
        try:
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fig.savefig(os.path.join(plots_dir, f"snr_{data['snr'].iloc[i]}_symbol_{data['symbol'].iloc[i]}_{sample_idx}.png"), dpi=num_symbols)
        except Exception as e:
            logger.error(f"Error generating plot for sample {sample_idx} in file snr_{data['snr'].iloc[i]}_symbol_{data['symbol'].iloc[i]}. Error: {e}")
        
        if (i + 1) % 5000 == 0:
            logger.info(f"Generated {i + 1} plots in {time.time() - start_time:.4f} seconds") 

    logger.info(f"Finished generating {len(data)} plots in {time.time() - start_time:.4f} seconds")


if __name__ == "__main__":
    from load_files import load_data
    data = load_data("test")
    generate_plots(data)
    
    