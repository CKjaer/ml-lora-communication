import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import logging
import time
import tfplot
import tensorflow as tf

@tfplot.autowrap
def plot_fft(freqs_idx, freqs, num_symbols, sample_idx, plots_dir, snr, symbol):
    fig, ax = tfplot.subplots(figsize=(1,1), dpi=num_symbols)
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.set_facecolor('black')
    fig.set_facecolor('black')
    ax.plot(freqs_idx, freqs, color = 'white', linewidth=0.5)
    #save images to folder
    try:
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(os.path.join(plots_dir, f"snr_{snr}_symbol_{symbol}_{sample_idx}.png"), dpi=num_symbols, facecolor='black', transparent=True)
    except Exception as e:
        logger.error(f"Error generating plot for sample {sample_idx} in file snr_{snr}_symbol_{symbol}. Error: {e}")
    return fig

def generate_plots(data, logger, spreading_factor: int, num_samples: int, directory: str):
    
    sample_idx = 0
    start_time = time.time()

 # create plots folder in given directory
    plots_dir = os.path.join(directory, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    num_symbols = 2**spreading_factor
    num_samples = tf.constant(num_samples,dtype=tf.int32)
    
    with tf.device('/cpu:0'):
        for i in tqdm(range(len(data)), desc="Generating plots"):
            freqs_idx = tf.range(0, num_symbols, 1, dtype=tf.int32)
            freqs = tf.convert_to_tensor(data['freqs'][i], dtype=tf.float32)
            snr = data['snr'][i]
            symbol = data['symbol'][i]
            # this counter is used to create a unique name for each plot
            sample_idx += 1
            if sample_idx > num_samples: # hard coded as we have 1000 representations
                sample_idx = 1

            plot_fft(freqs_idx, freqs, num_symbols, sample_idx, plots_dir, snr,symbol)
        
          if (i + 1) % 5000 == 0:
              logger.info(f"Generated {i + 1} plots in {time.time() - start_time:.4f} seconds") 

        logger.info(f"Finished generating {len(data)} plots in {time.time() - start_time:.4f} seconds")


if __name__ == "__main__":
    from load_files import load_data
    #Setup a simple logger
    logfilename = "test_log.log"
    log_path = os.path.join(os.getcwd(),logfilename)
    logger = logging.getLogger(__name__)
    direc = "test"
    data = load_data(direc,logger)
    SF = 7
    N_samps = 10
    generate_plots(data, logger,SF,N_samps,os.path.join(os.getcwd(),"out"))
    