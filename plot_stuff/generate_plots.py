import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import logging
import time

def generate_plots(data, logger, spreading_factor: int, num_samples: int, directory: str, max_vals: dict = None, line_plot: bool = True):
    sample_idx = 0
    start_time = time.time()
    num_symbols = 2**spreading_factor
    
    # create a lin space for the frequency values
    #freqs_idx = np.arange(0, num_symbols, 1)
    
    #plt.switch_backend('agg')
    #fig = plt.figure(figsize=(1,1), dpi=num_symbols)
    plots_dir = os.path.join(directory, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.figure(figsize=(1, 1), dpi=num_symbols)  # Create a figure with 128x128 pixels
    plt.gcf().patch.set_facecolor('black')  # Set the figure background to black
    plt.axis('off')
    plt.xlim(0,num_symbols-1)
    plt.gca().set_facecolor('black')  # Set background color to black
    #Set prev_symbol to a symbol it can never be
    prev_symbol = num_symbols+1

    for i in tqdm(range(len(data)), desc="Generating plots"):
        #order: RateParam -> SNR -> Symbol -> index
        snr = data['snr'][i]
        symbol = data['symbol'][i]
        rate = data['rate'][i]

        #List is ordered - thus a new symbol requires sample index to start over :)
        if symbol != prev_symbol:
            sample_idx = 0
        # find the upper limit for current snr condition
        upper_y_lim = max_vals[data['snr'][i]]
        plt.ylim(0, upper_y_lim)

        # find the data that should be used for this plot
        data_freqs_i = data['freqs'][i]
        
        # set to true to plot line_plot
        if line_plot:
            # 0.75 seems like a fine size :)
            line_width = 0.75
            linp = plt.plot(data_freqs_i,
                    color = 'white',
                    linewidth=line_width,
                    )
        else:
            line_width = 0.5
            stems = plt.stem(data_freqs_i,
                            linefmt='w-',
                            markerfmt=' ',
                            basefmt=' ')
            
            # Set the line width for each stem
            for stem in stems:
                stem.set_linewidth(line_width)  # Set line width to 3 pixels (adjust as needed)
                stem.set_aa(False)  # Disable anti-aliasing
        
        # find the index of the current sample
        #sample_idx = i % num_samples[0]
        
        #save images to folder
        try:
            # removes padding
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            filename = f"snr_{snr}_symbol_{symbol}_rate_{rate}_{sample_idx}.png"
            plt.savefig(os.path.join(
                            plots_dir,
                            filename
                        ),
                        pad_inches=0, # Probably not nessecary
                        dpi=num_symbols # Also probably not nessecary
                        )
        except Exception as e:
            logger.error(f"Error generating plot for sample {sample_idx} in file snr_{data['snr'][i]}_symbol_{data['symbol'][i]}. Error: {e}")
        
        if line_plot:
            linp.pop().remove()
        else:
            stems.remove()
        
        if (i + 1) % 5000 == 0:
            logger.info(f"Generated {i + 1} plots in {time.time() - start_time:.4f} seconds") 
        prev_symbol = symbol
        sample_idx = sample_idx + 1

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
    
    csv_dir = "C:/Users/rdybs/Desktop/gitnstuff/ml-lora-communication/output/20241023-134010/csv"
    data = load_data(csv_dir, logger=logger) # change directory when running test
    max_vals = find_max(data, logger=logger)

    import uuid
    rand = uuid.uuid1()
    print(rand)
    outdir = "C:/Users/rdybs/Desktop/gitnstuff/ml-lora-communication/output/example"
    outerdir = os.path.join(outdir,str(rand))
    os.makedirs(outerdir,exist_ok=True)
    generate_plots(data, logger=logger, spreading_factor=7, num_samples=[10,10,10,10,10,10,10], directory=outerdir, max_vals=max_vals, line_plot=True) # change directory when running test