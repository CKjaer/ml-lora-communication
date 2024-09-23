import numpy as np
import matplotlib.pyplot as plt
from load_files import load_data
import os
from PIL import Image

def generate_plots(data, directory=""):
    """This function takes a list of data of form ['freqs', 'snr', 'symbol'] and generated binary FFT modulus plots.
    
        Use load_data from load_data.py to get data from csv to dataframe of this form

    Args:
        data (_type_): list containing absolute values of the generated FFT
        directory (str, optional): Directory to where the plots folder should be created. Defaults to current working directory.
    """    
    sample_idx = 0
    plt.switch_backend('agg')
    
    for i in range(len(data)):
        freqs = list(map(float, data["freqs"].iloc[i].split(';'))) # split string and convert to float
        freqs_idx = np.arange(0, len(freqs), 1)
        #print(data["symbol"].iloc[i])

        fig = plt.figure(figsize=(1,1), dpi=128)
        ax = fig.add_subplot(111)
        plt.axis('off')
        fig.set_facecolor('black')
        plt.plot(freqs_idx, freqs, color = 'white', linewidth=0.5)
        plt.close(fig)
        
        # this counter is used to create a unique name for each plot
        sample_idx += 1
        if sample_idx == 1001: # hard coded as we have 1000 representations
            sample_idx = 1
        
        # create plots folder in given directory
        plots_dir = os.path.join(directory, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        #save images to folder
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(os.path.join(plots_dir, f"snr_{data['snr'].iloc[i]}_symbol_{data['symbol'].iloc[i]}_{sample_idx}.png"), dpi=128)
        # resize_plot(128, directory=plots_dir)

def resize_plot(num_symbols, directory="plots"):
    """this function resizes the plots to MxM where M is the number of symbols
    Args:
        num_symbols (_type_): number of symbols
        directory (str, optional): Directory to where the plots are located. Defaults to "plots".
    """
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            file_path = os.path.join(directory, filename)
            img = Image.open(file_path)
            img = img.resize((num_symbols, num_symbols))
            img.save(file_path)

if __name__ == "__main__":
    data = load_data("test")
    generate_plots(data)
    
    