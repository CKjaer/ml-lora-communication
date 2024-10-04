import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import logging
import time
import tfplot
import tensorflow as tf

def generate_plots(data, logger, spreading_factor: int, num_samples: int, directory: str):
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
    
    logger.name = "Plot generator"
    logger.debug("Starting the plot generation")
    logger.debug(f"Available logical devices: {tf.config.list_logical_devices('GPU')}")

    gpus = tf.config.list_logical_devices('GPU')
    if gpus:
        logger.debug('Found GPU, using that')
        device = tf.device(gpus[0].name)
    else:
        logger.debug('GPU device not found, using CPU')
        device = tf.device('/device:CPU:0')

    sample_idx = 0
    start_time = time.time()

 # create plots folder in given directory
    plots_dir = os.path.join(directory, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    num_symbols = tf.constant(2**spreading_factor,dtype=tf.int32)
    num_samples = tf.constant(num_samples,dtype=tf.int32)
    data_len = tf.constant(len(data),dtype=tf.int32)
    snr_list = tf.convert_to_tensor(data['snr'],dtype=tf.int32)
    symbol_list = tf.convert_to_tensor(data['symbol'],dtype=tf.int32)
    freqs_list = np.array(data['freqs'])
    freqs_data = np.zeros((data_len,num_symbols))
    for i in range(data_len):
        freqs_data[i] = freqs_list[i]
    freqs_list = tf.convert_to_tensor(freqs_data,dtype=tf.float32)

    with device:
        for i in tf.range(data_len):
            freqs_idx = tf.range(0, num_symbols, 1, dtype=tf.int32)
            freqs = tf.gather(freqs_list,i,axis=0)
            snr = tf.slice(snr_list,[i],[1])
            symbol = tf.slice(symbol_list,[i],[1])
            # this counter is used to create a unique name for each plot
            sample_idx = tf.math.floormod(i,num_samples)

            plot_fft(freqs_idx, freqs, num_symbols, sample_idx, plots_dir, snr,symbol)
        
            if tf.math.floormod(i+1, 5000) == 0:
                print(f"Current time: {time.time() - start_time}, current i:{i}")
                logger.info(f"Generated {i + 1} plots in {time.time() - start_time:.4f} seconds") 

        logger.info(f"Finished generating {len(data)} plots in {time.time() - start_time:.4f} seconds")


if __name__ == "__main__":
    from load_files import load_data
    #Setup a simple logger
    logfilename = "test_log.log"
    log_path = os.path.join(os.getcwd(),logfilename)
    logger = logging.getLogger(__name__)
    direc = "output\\5a4a7784-8552-49d3-a105-cc248da13d71\\csv"
    direc = os.path.join(os.getcwd(),direc)
    print(direc)
    data = load_data(direc,logger)
    SF = 7
    N_samps = 10
    generate_plots(data, logger,SF,N_samps,os.path.join(os.getcwd(),"out"))
    